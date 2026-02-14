"""
torch.compile 加速版 Duplex 评估脚本。

对模型的核心子模块（ViT、LLM backbone、Resampler）进行 torch.compile，
并启用 FX Graph Cache 实现跨进程缓存，第二次运行无需重新编译。

用法:
    # 第一次运行（会触发编译，较慢，约 1-3 分钟）
    python eval_duplex_compiled.py --video_dir /root/omni_duplex_eval/omni_demo_zh_duplex

    # 第二次运行（命中缓存，几乎零编译时间）
    python eval_duplex_compiled.py --video_dir /root/omni_duplex_eval/omni_demo_zh_duplex

对比不带 compile 的版本，观察 wall 时间差异。
"""

import json
import os
import time
import argparse
from pathlib import Path

import librosa
import torch
import torch._inductor.config
import numpy as np
from transformers import AutoModel

from minicpmo.utils import generate_duplex_video, get_video_frame_audio_segments


# ─── 启用 FX Graph Cache，编译结果持久化到磁盘 ───
torch._inductor.config.fx_graph_cache = True
# 可选：指定缓存目录
COMPILE_CACHE_DIR = os.environ.get(
    "TORCHINDUCTOR_CACHE_DIR",
    "/root/.cache/torch_compile_cache"
)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = COMPILE_CACHE_DIR


# ─── 所有要追踪的耗时指标（同 eval_duplex_batch.py）───
PREFILL_COST_KEYS = [
    "cost_vision_process", "cost_vision_embed", "cost_vision_feed",
    "cost_audio_process", "cost_audio_embed", "cost_audio_feed",
    "cost_prefill_all",
]
GENERATE_COST_KEYS = [
    "cost_llm", "cost_tts_prep", "cost_tts", "cost_token2wav",
    "cost_generate_all",
]
TOKEN_KEYS = ["n_tokens", "n_tts_tokens"]
WALL_KEYS = ["wall_prefill", "wall_generate", "wall_chunk_total"]
ALL_METRIC_KEYS = PREFILL_COST_KEYS + GENERATE_COST_KEYS + TOKEN_KEYS + ["cost_chunk_total"] + WALL_KEYS


def load_and_compile_model(model_path: str, compile_mode: str = "reduce-overhead"):
    """
    加载模型 → 转 duplex → 对核心子模块进行 torch.compile。

    compile 的子模块：
      - model.model.vpm          : ViT 视觉编码器 (SiglipVisionTransformer)
      - model.model.llm.model    : LLM backbone (Qwen3)
      - model.model.resampler    : 视觉 resampler
      - model.model.tts.model    : TTS 内部 LLM backbone (LlamaModel)

    不 compile：
      - model.model.apm          : Whisper 音频编码器（动态 shape + 流式 KV cache，compile 反而增加开销）

    不 compile 的部分：
      - MiniCPMODuplex 外层（streaming 状态管理，Python 控制流多，compile 收益低）
      - TTS 外层（流式解码控制逻辑）
      - Token2wav（扩散模型，动态 shape 多）
    """
    print(f"[INFO] 正在加载模型: {model_path}")
    raw_model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    raw_model.eval().cuda()

    # ─── 对核心子模块进行 torch.compile ───
    print(f"[INFO] 正在 compile 子模块 (mode={compile_mode})...")
    print(f"[INFO] FX Graph Cache 目录: {COMPILE_CACHE_DIR}")

    t0 = time.time()

    # 1. ViT 视觉编码器
    print(f"  → compile vpm (SiglipVisionTransformer)...")
    raw_model.vpm = torch.compile(raw_model.vpm, mode=compile_mode)

    # 2. LLM backbone（compile 内部 model，保留外层 lm_head + generate 逻辑）
    print(f"  → compile llm.model (Qwen3 backbone)...")
    raw_model.llm.model = torch.compile(raw_model.llm.model, mode=compile_mode)

    # 3. Resampler
    print(f"  → compile resampler...")
    raw_model.resampler = torch.compile(raw_model.resampler, mode=compile_mode)

    # 4. TTS 内部的 LLM backbone（LlamaModel）
    print(f"  → compile tts.model (Llama backbone for TTS)...")
    raw_model.tts.model = torch.compile(raw_model.tts.model, mode=compile_mode)

    compile_wrap_time = time.time() - t0
    print(f"[INFO] compile 包装完成 (耗时 {compile_wrap_time:.2f}s，实际编译在首次 forward 时触发)")

    # 启用 TF32 提升矩阵乘性能
    torch.set_float32_matmul_precision('high')

    # 转为 duplex 模式
    model = raw_model.as_duplex()
    print("[INFO] 模型加载 + compile 完成")

    return model


def print_chunk_detail(chunk_idx, prefill_result, gen_result, wall_prefill, wall_generate):
    """流式打印每个 chunk 的完整耗时信息"""
    is_listen = gen_result["is_listen"]
    text = gen_result["text"]
    status = "LISTEN" if is_listen else "SPEAK "

    pf = prefill_result
    vp = pf.get("cost_vision_process", 0)
    ve = pf.get("cost_vision_embed", 0)
    vf = pf.get("cost_vision_feed", 0)
    ap = pf.get("cost_audio_process", 0)
    ae = pf.get("cost_audio_embed", 0)
    af = pf.get("cost_audio_feed", 0)
    pa = pf.get("cost_all", 0)
    vision_total = vp + ve + vf
    audio_total = ap + ae + af

    g = gen_result
    gl = g["cost_llm"]
    gtp = g["cost_tts_prep"]
    gt = g["cost_tts"]
    gtw = g["cost_token2wav"]
    ga = g["cost_all"]

    wall_total = wall_prefill + wall_generate

    text_display = f'text="{text}"' if text else ''
    print(f"  ┌─ chunk {chunk_idx:3d} [{status}] {text_display}")
    print(f"  │  PREFILL  internal={pa*1000:7.1f}ms  wall={wall_prefill*1000:7.1f}ms │ "
          f"vision: proc={vp*1000:.1f} embed={ve*1000:.1f} feed={vf*1000:.1f} (Σ={vision_total*1000:.1f}ms) │ "
          f"audio: proc={ap*1000:.1f} embed={ae*1000:.1f} feed={af*1000:.1f} (Σ={audio_total*1000:.1f}ms)")
    print(f"  │  GENERATE internal={ga*1000:7.1f}ms  wall={wall_generate*1000:7.1f}ms │ "
          f"llm={gl*1000:.1f} tts_prep={gtp*1000:.1f} tts={gt*1000:.1f} token2wav={gtw*1000:.1f} │ "
          f"tokens={g['n_tokens']} tts_tokens={g['n_tts_tokens']}")
    print(f"  └─ WALL TOTAL {wall_total*1000:7.1f}ms"
          f"{'  eot=Y' if g['end_of_turn'] else ''}")


def evaluate_single_video(model, video_path: str, ref_audio_path: str, output_video_path: str):
    """对单个视频运行 duplex 推理"""
    ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

    video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
        video_path, stack_frames=1, use_ffmpeg=True, adjust_audio_length=True
    )

    model.prepare(
        prefix_system_prompt="Streaming Omni Conversation.",
        ref_audio=ref_audio,
        prompt_wav_path=ref_audio_path,
    )

    chunk_results = []
    results_log = []
    timed_output_audio = []

    for chunk_idx in range(len(audio_segments)):
        audio_chunk = audio_segments[chunk_idx] if chunk_idx < len(audio_segments) else None
        frame = video_frames[chunk_idx] if chunk_idx < len(video_frames) else None
        frame_list = []
        if frame is not None:
            frame_list.append(frame)
            if (stacked_frames is not None
                    and chunk_idx < len(stacked_frames)
                    and stacked_frames[chunk_idx] is not None):
                frame_list.append(stacked_frames[chunk_idx])

        t_pf_start = time.time()
        prefill_result = model.streaming_prefill(
            audio_waveform=audio_chunk,
            frame_list=frame_list,
            max_slice_nums=1,
            batch_vision_feed=False,
        )
        wall_prefill = time.time() - t_pf_start

        t_gen_start = time.time()
        gen_result = model.streaming_generate(
            prompt_wav_path=ref_audio_path,
            max_new_speak_tokens_per_chunk=20,
            decode_mode="sampling",
        )
        wall_generate = time.time() - t_gen_start

        print_chunk_detail(chunk_idx, prefill_result, gen_result, wall_prefill, wall_generate)

        if gen_result["audio_waveform"] is not None:
            timed_output_audio.append((chunk_idx, gen_result["audio_waveform"]))

        results_log.append({
            "chunk_idx": chunk_idx,
            "is_listen": gen_result["is_listen"],
            "text": gen_result["text"],
            "end_of_turn": gen_result["end_of_turn"],
            "current_time": gen_result["current_time"],
            "audio_length": len(gen_result["audio_waveform"]) if gen_result["audio_waveform"] is not None else 0,
        })

        prefill_all = prefill_result.get("cost_all", 0)
        generate_all = gen_result["cost_all"]

        chunk_results.append({
            "chunk_idx": chunk_idx,
            "is_listen": gen_result["is_listen"],
            "text": gen_result["text"],
            "end_of_turn": gen_result["end_of_turn"],
            "current_time": gen_result["current_time"],
            "audio_length": len(gen_result["audio_waveform"]) if gen_result["audio_waveform"] is not None else 0,
            "cost_vision_process": prefill_result.get("cost_vision_process", 0),
            "cost_vision_embed": prefill_result.get("cost_vision_embed", 0),
            "cost_vision_feed": prefill_result.get("cost_vision_feed", 0),
            "cost_audio_process": prefill_result.get("cost_audio_process", 0),
            "cost_audio_embed": prefill_result.get("cost_audio_embed", 0),
            "cost_audio_feed": prefill_result.get("cost_audio_feed", 0),
            "cost_prefill_all": prefill_all,
            "wall_prefill": wall_prefill,
            "cost_llm": gen_result["cost_llm"],
            "cost_tts_prep": gen_result["cost_tts_prep"],
            "cost_tts": gen_result["cost_tts"],
            "cost_token2wav": gen_result["cost_token2wav"],
            "cost_generate_all": generate_all,
            "wall_generate": wall_generate,
            "n_tokens": gen_result["n_tokens"],
            "n_tts_tokens": gen_result["n_tts_tokens"],
            "cost_chunk_total": prefill_all + generate_all,
            "wall_chunk_total": wall_prefill + wall_generate,
        })

    print(f"  [渲染] 正在生成输出视频: {output_video_path}")
    generate_duplex_video(
        video_path=video_path,
        output_video_path=output_video_path,
        results_log=results_log,
        timed_output_audio=timed_output_audio,
        output_sample_rate=24000,
    )
    print(f"  [渲染] 输出视频已保存")

    return chunk_results


def compute_averages(chunk_results, metric_keys):
    if not chunk_results:
        return {k: 0.0 for k in metric_keys}

    averages = {}
    for key in metric_keys:
        values = [c[key] for c in chunk_results]
        averages[key] = float(np.mean(values))

    speak_chunks = [c for c in chunk_results if not c["is_listen"]]
    for key in metric_keys:
        if speak_chunks:
            averages[f"{key}_speak_only"] = float(np.mean([c[key] for c in speak_chunks]))
        else:
            averages[f"{key}_speak_only"] = 0.0

    averages["total_chunks"] = len(chunk_results)
    averages["speak_chunks"] = len(speak_chunks)
    averages["listen_chunks"] = len(chunk_results) - len(speak_chunks)
    return averages


def print_video_summary(video_avg, total_time):
    print(f"\n  ╔{'═'*62}╗")
    print(f"  ║  视频汇总统计 [torch.compile]                              ║")
    print(f"  ╠{'═'*62}╣")
    print(f"  ║  总耗时: {total_time:8.2f}s   "
          f"chunks: {video_avg['total_chunks']:3d} "
          f"(speak: {video_avg['speak_chunks']}, listen: {video_avg['listen_chunks']})")
    print(f"  ╟{'─'*62}╢")
    print(f"  ║  PREFILL 平均:")
    print(f"  ║    vision_process: {video_avg['cost_vision_process']*1000:6.1f}ms  "
          f"vision_embed(ViT): {video_avg['cost_vision_embed']*1000:6.1f}ms  "
          f"vision_feed: {video_avg['cost_vision_feed']*1000:6.1f}ms")
    print(f"  ║    audio_process:  {video_avg['cost_audio_process']*1000:6.1f}ms  "
          f"audio_embed:       {video_avg['cost_audio_embed']*1000:6.1f}ms  "
          f"audio_feed:  {video_avg['cost_audio_feed']*1000:6.1f}ms")
    print(f"  ║    prefill_all:    {video_avg['cost_prefill_all']*1000:6.1f}ms")
    print(f"  ╟{'─'*62}╢")
    print(f"  ║  GENERATE 平均:")
    print(f"  ║    llm:       {video_avg['cost_llm']*1000:6.1f}ms  "
          f"tts_prep: {video_avg['cost_tts_prep']*1000:6.1f}ms  "
          f"tts: {video_avg['cost_tts']*1000:6.1f}ms  "
          f"token2wav: {video_avg['cost_token2wav']*1000:6.1f}ms")
    print(f"  ║    generate_all:   {video_avg['cost_generate_all']*1000:6.1f}ms")
    print(f"  ╟{'─'*62}╢")
    print(f"  ║  WALL 平均:")
    print(f"  ║    wall_prefill:  {video_avg['wall_prefill']*1000:6.1f}ms  "
          f"wall_generate: {video_avg['wall_generate']*1000:6.1f}ms  "
          f"wall_total: {video_avg['wall_chunk_total']*1000:6.1f}ms")
    print(f"  ╟{'─'*62}╢")
    print(f"  ║  INTERNAL TOTAL: {video_avg['cost_chunk_total']*1000:6.1f}ms  "
          f"(prefill {video_avg['cost_prefill_all']*1000:.1f} + generate {video_avg['cost_generate_all']*1000:.1f})")
    print(f"  ╚{'═'*62}╝")


def main():
    parser = argparse.ArgumentParser(description="torch.compile 加速版 Duplex 评估")
    parser.add_argument("--model_path", type=str, default="/root/MiniCPM-o-4_5")
    parser.add_argument("--video_dir", type=str,
                        default="/root/omni_duplex_eval/omni_demo_zh_duplex")
    parser.add_argument("--ref_audio", type=str,
                        default="/root/MiniCPM-o-4_5/assets/HT_ref_audio.wav")
    parser.add_argument("--output_dir", type=str,
                        default="/root/test/speed_test/eval_report_compiled")
    parser.add_argument("--videos", type=str, default=None,
                        help="指定视频文件名，逗号分隔。不指定则全部")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile 模式")
    parser.add_argument("--no_compile", action="store_true",
                        help="禁用 compile，用于 A/B 对比")
    args = parser.parse_args()

    # 扫描视频
    video_dir = Path(args.video_dir)
    if args.videos:
        video_files = []
        for name in args.videos.split(","):
            p = video_dir / name.strip()
            if p.exists():
                video_files.append(p)
            else:
                print(f"[WARN] 视频不存在: {p}")
    else:
        video_files = sorted(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"[ERROR] 在 {video_dir} 中未找到 mp4 文件")
        return

    print(f"[INFO] 找到 {len(video_files)} 个视频")
    print(f"[INFO] compile 模式: {'DISABLED' if args.no_compile else args.compile_mode}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = output_dir / "rendered"
    rendered_dir.mkdir(exist_ok=True)
    source_videos_dir = output_dir / "videos"
    source_videos_dir.mkdir(exist_ok=True)

    # 加载模型（带或不带 compile）
    total_load_start = time.time()
    if args.no_compile:
        print("[INFO] 不使用 torch.compile（对比基线）")
        raw = AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        )
        raw.eval().cuda()
        model = raw.as_duplex()
    else:
        model = load_and_compile_model(args.model_path, args.compile_mode)
    total_load_time = time.time() - total_load_start
    print(f"[INFO] 模型加载总耗时: {total_load_time:.2f}s")

    # ─── Warmup：第一个视频触发编译 ───
    warmup_video = video_files[0]
    print(f"\n{'='*70}")
    print(f"  [WARMUP] 使用 {warmup_video.name} 触发编译（首次会较慢）")
    print(f"{'='*70}")

    warmup_start = time.time()
    warmup_results = evaluate_single_video(
        model, str(warmup_video), args.ref_audio,
        str(rendered_dir / f"output_{warmup_video.stem}.mp4")
    )
    warmup_time = time.time() - warmup_start
    warmup_avg = compute_averages(warmup_results, ALL_METRIC_KEYS)
    print_video_summary(warmup_avg, warmup_time)
    print(f"\n  ⚡ WARMUP 总耗时: {warmup_time:.2f}s（含编译时间）")

    # 软链接 warmup 视频
    src_link = source_videos_dir / warmup_video.name
    if src_link.exists() or src_link.is_symlink():
        src_link.unlink()
    src_link.symlink_to(warmup_video.resolve())

    all_results = [{
        "video_name": warmup_video.name,
        "video_path": str(warmup_video.resolve()),
        "source_video_rel": f"videos/{warmup_video.name}",
        "rendered_video_rel": f"rendered/output_{warmup_video.stem}.mp4",
        "total_inference_time": warmup_time,
        "chunk_results": warmup_results,
        "averages": warmup_avg,
        "model_output_text": "".join(c["text"] for c in warmup_results),
        "is_warmup": True,
    }]
    all_chunks_flat = list(warmup_results)

    # ─── 正式评估：剩余视频 ───
    remaining = video_files[1:]
    if remaining:
        print(f"\n{'='*70}")
        print(f"  [EVAL] 开始评估剩余 {len(remaining)} 个视频（编译已缓存）")
        print(f"{'='*70}")

    for vid_idx, video_path in enumerate(remaining):
        video_name = video_path.name
        output_video_name = f"output_{video_path.stem}.mp4"
        output_video_path = str(rendered_dir / output_video_name)

        print(f"\n{'='*70}")
        print(f"  [{vid_idx+2}/{len(video_files)}] {video_name}")
        print(f"{'='*70}")

        t0 = time.time()
        chunk_results = evaluate_single_video(
            model, str(video_path), args.ref_audio, output_video_path
        )
        total_time = time.time() - t0

        video_avg = compute_averages(chunk_results, ALL_METRIC_KEYS)
        print_video_summary(video_avg, total_time)

        src_link = source_videos_dir / video_name
        if src_link.exists() or src_link.is_symlink():
            src_link.unlink()
        src_link.symlink_to(video_path.resolve())

        all_results.append({
            "video_name": video_name,
            "video_path": str(video_path.resolve()),
            "source_video_rel": f"videos/{video_name}",
            "rendered_video_rel": f"rendered/{output_video_name}",
            "total_inference_time": total_time,
            "chunk_results": chunk_results,
            "averages": video_avg,
            "model_output_text": "".join(c["text"] for c in chunk_results),
            "is_warmup": False,
        })
        all_chunks_flat.extend(chunk_results)

    # 总体统计（排除 warmup）
    non_warmup_chunks = []
    for r in all_results:
        if not r.get("is_warmup", False):
            non_warmup_chunks.extend(r["chunk_results"])

    overall_avg = compute_averages(all_chunks_flat, ALL_METRIC_KEYS)
    overall_avg_no_warmup = compute_averages(non_warmup_chunks, ALL_METRIC_KEYS) if non_warmup_chunks else overall_avg

    output_data = {
        "metadata": {
            "model_path": args.model_path,
            "video_dir": str(video_dir.resolve()),
            "ref_audio": args.ref_audio,
            "total_videos": len(all_results),
            "total_chunks": len(all_chunks_flat),
            "eval_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "compile_mode": "disabled" if args.no_compile else args.compile_mode,
            "compile_cache_dir": COMPILE_CACHE_DIR,
        },
        "overall_averages": overall_avg,
        "overall_averages_no_warmup": overall_avg_no_warmup,
        "video_results": all_results,
    }

    json_path = output_dir / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 打印总结
    print(f"\n{'='*70}")
    print(f"  [DONE] torch.compile 评估完成！")
    print(f"{'='*70}")
    print(f"  compile 模式: {'DISABLED' if args.no_compile else args.compile_mode}")
    print(f"  结果 JSON: {json_path}")
    print(f"  总视频数: {len(all_results)}  总chunks: {len(all_chunks_flat)}")
    print(f"\n  ── 总体平均（含 warmup）──")
    print(f"    wall_prefill:  {overall_avg['wall_prefill']*1000:6.1f}ms")
    print(f"    wall_generate: {overall_avg['wall_generate']*1000:6.1f}ms")
    print(f"    wall_total:    {overall_avg['wall_chunk_total']*1000:6.1f}ms")
    if non_warmup_chunks:
        print(f"\n  ── 总体平均（不含 warmup）──")
        print(f"    wall_prefill:  {overall_avg_no_warmup['wall_prefill']*1000:6.1f}ms")
        print(f"    wall_generate: {overall_avg_no_warmup['wall_generate']*1000:6.1f}ms")
        print(f"    wall_total:    {overall_avg_no_warmup['wall_chunk_total']*1000:6.1f}ms")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
