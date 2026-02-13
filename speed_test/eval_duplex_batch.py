"""
批量评估 MiniCPM-o Duplex 模型。
直接扫描视频目录中的 mp4 文件进行推理，不依赖 JSON 元数据。
每个视频会：
  1. 记录每个 chunk 的完整推理耗时（prefill + generate 全阶段）
  2. 渲染输出带字幕+AI语音的结果视频 mp4
  3. 计算平均值，保存到 JSON

用法:
    python eval_duplex_batch.py --video_dir /root/omni_duplex_eval/omni_demo_zh_duplex
    python eval_duplex_batch.py --video_dir /path/to/new_videos
"""

import glob
import json
import os
import time
import argparse
from pathlib import Path

import librosa
import torch
import numpy as np
from transformers import AutoModel

from minicpmo.utils import generate_duplex_video, get_video_frame_audio_segments


# ─── 所有要追踪的耗时指标 ───
# prefill 阶段（streaming_prefill 返回）
PREFILL_COST_KEYS = [
    "cost_vision_process",   # 图像预处理
    "cost_vision_embed",     # ViT 编码
    "cost_vision_feed",      # 视觉嵌入 feed 到 decoder
    "cost_audio_process",    # 音频预处理
    "cost_audio_embed",      # 音频编码
    "cost_audio_feed",       # 音频嵌入 feed 到 decoder
    "cost_prefill_all",      # prefill 总耗时
]

# generate 阶段（streaming_generate 返回）
GENERATE_COST_KEYS = [
    "cost_llm",              # LLM 解码
    "cost_tts_prep",         # TTS 准备
    "cost_tts",              # TTS 推理
    "cost_token2wav",        # token 转波形
    "cost_generate_all",     # generate 总耗时
]

# token 统计
TOKEN_KEYS = [
    "n_tokens",              # LLM 输出 token 数
    "n_tts_tokens",          # TTS token 数
]

# 汇总
COST_CHUNK_TOTAL = "cost_chunk_total"  # prefill_all + generate_all

# 外部墙钟时间
WALL_KEYS = [
    "wall_prefill",          # streaming_prefill 整体墙钟时间
    "wall_generate",         # streaming_generate 整体墙钟时间
    "wall_chunk_total",      # wall_prefill + wall_generate
]

ALL_METRIC_KEYS = PREFILL_COST_KEYS + GENERATE_COST_KEYS + TOKEN_KEYS + [COST_CHUNK_TOTAL] + WALL_KEYS


def load_model(model_path: str):
    """加载模型并转为 duplex 模式"""
    print(f"[INFO] 正在加载模型: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.eval().cuda()
    model = model.as_duplex()
    print("[INFO] 模型加载完成")
    return model


def print_chunk_detail(chunk_idx, prefill_result, gen_result, wall_prefill, wall_generate):
    """流式打印每个 chunk 的完整耗时信息"""
    is_listen = gen_result["is_listen"]
    text = gen_result["text"]
    status = "LISTEN" if is_listen else "SPEAK "

    # prefill 耗时
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

    # generate 耗时
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
    """对单个视频运行 duplex 推理，渲染输出视频，返回所有 chunk 的结果列表"""
    ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)

    # 提取视频帧和音频片段
    video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
        video_path, stack_frames=1, use_ffmpeg=True, adjust_audio_length=True
    )

    # 准备 duplex 会话
    model.prepare(
        prefix_system_prompt="Streaming Omni Conversation.",
        ref_audio=ref_audio,
        prompt_wav_path=ref_audio_path,
    )

    chunk_results = []
    results_log = []          # 给 generate_duplex_video 用
    timed_output_audio = []   # 给 generate_duplex_video 用

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

        # Step 1: Streaming prefill（返回 vision + audio 各阶段耗时）
        t_pf_start = time.time()
        prefill_result = model.streaming_prefill(
            audio_waveform=audio_chunk,
            frame_list=frame_list,
            max_slice_nums=1,
            batch_vision_feed=False,
        )
        wall_prefill = time.time() - t_pf_start

        # Step 2: Streaming generate（返回 llm + tts 各阶段耗时）
        t_gen_start = time.time()
        gen_result = model.streaming_generate(
            prompt_wav_path=ref_audio_path,
            max_new_speak_tokens_per_chunk=20,
            decode_mode="sampling",
        )
        wall_generate = time.time() - t_gen_start

        # 流式打印完整耗时
        print_chunk_detail(chunk_idx, prefill_result, gen_result, wall_prefill, wall_generate)

        # 收集音频用于渲染输出视频
        if gen_result["audio_waveform"] is not None:
            timed_output_audio.append((chunk_idx, gen_result["audio_waveform"]))

        # results_log 给 generate_duplex_video 用
        results_log.append({
            "chunk_idx": chunk_idx,
            "is_listen": gen_result["is_listen"],
            "text": gen_result["text"],
            "end_of_turn": gen_result["end_of_turn"],
            "current_time": gen_result["current_time"],
            "audio_length": len(gen_result["audio_waveform"]) if gen_result["audio_waveform"] is not None else 0,
        })

        # chunk_results 记录完整耗时指标
        prefill_all = prefill_result.get("cost_all", 0)
        generate_all = gen_result["cost_all"]

        chunk_results.append({
            "chunk_idx": chunk_idx,
            "is_listen": gen_result["is_listen"],
            "text": gen_result["text"],
            "end_of_turn": gen_result["end_of_turn"],
            "current_time": gen_result["current_time"],
            "audio_length": len(gen_result["audio_waveform"]) if gen_result["audio_waveform"] is not None else 0,
            # prefill 阶段（内部计时）
            "cost_vision_process": prefill_result.get("cost_vision_process", 0),
            "cost_vision_embed": prefill_result.get("cost_vision_embed", 0),
            "cost_vision_feed": prefill_result.get("cost_vision_feed", 0),
            "cost_audio_process": prefill_result.get("cost_audio_process", 0),
            "cost_audio_embed": prefill_result.get("cost_audio_embed", 0),
            "cost_audio_feed": prefill_result.get("cost_audio_feed", 0),
            "cost_prefill_all": prefill_all,
            # prefill 外部墙钟时间
            "wall_prefill": wall_prefill,
            # generate 阶段（内部计时）
            "cost_llm": gen_result["cost_llm"],
            "cost_tts_prep": gen_result["cost_tts_prep"],
            "cost_tts": gen_result["cost_tts"],
            "cost_token2wav": gen_result["cost_token2wav"],
            "cost_generate_all": generate_all,
            # generate 外部墙钟时间
            "wall_generate": wall_generate,
            # token 统计
            "n_tokens": gen_result["n_tokens"],
            "n_tts_tokens": gen_result["n_tts_tokens"],
            # 总耗时
            "cost_chunk_total": prefill_all + generate_all,
            "wall_chunk_total": wall_prefill + wall_generate,
        })

    # 渲染输出视频（带 AI 语音 + 字幕）
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


def compute_averages(chunk_results: list, metric_keys: list):
    """计算一组 chunk 结果的平均耗时指标"""
    if not chunk_results:
        return {k: 0.0 for k in metric_keys}

    averages = {}
    for key in metric_keys:
        values = [c[key] for c in chunk_results]
        averages[key] = float(np.mean(values))

    # 额外统计：只统计 speak 的 chunk
    speak_chunks = [c for c in chunk_results if not c["is_listen"]]
    for key in metric_keys:
        if speak_chunks:
            values = [c[key] for c in speak_chunks]
            averages[f"{key}_speak_only"] = float(np.mean(values))
        else:
            averages[f"{key}_speak_only"] = 0.0

    averages["total_chunks"] = len(chunk_results)
    averages["speak_chunks"] = len(speak_chunks)
    averages["listen_chunks"] = len(chunk_results) - len(speak_chunks)

    return averages


def print_video_summary(video_avg: dict, total_time: float):
    """打印单个视频的汇总统计"""
    print(f"\n  ╔{'═'*58}╗")
    print(f"  ║  视频汇总统计                                          ║")
    print(f"  ╠{'═'*58}╣")
    print(f"  ║  总耗时: {total_time:8.2f}s   "
          f"chunks: {video_avg['total_chunks']:3d} "
          f"(speak: {video_avg['speak_chunks']}, listen: {video_avg['listen_chunks']})")
    print(f"  ╟{'─'*58}╢")
    print(f"  ║  PREFILL 平均耗时:")
    print(f"  ║    vision_process: {video_avg['cost_vision_process']*1000:6.1f}ms  "
          f"vision_embed(ViT): {video_avg['cost_vision_embed']*1000:6.1f}ms  "
          f"vision_feed: {video_avg['cost_vision_feed']*1000:6.1f}ms")
    print(f"  ║    audio_process:  {video_avg['cost_audio_process']*1000:6.1f}ms  "
          f"audio_embed:       {video_avg['cost_audio_embed']*1000:6.1f}ms  "
          f"audio_feed:  {video_avg['cost_audio_feed']*1000:6.1f}ms")
    print(f"  ║    prefill_all:    {video_avg['cost_prefill_all']*1000:6.1f}ms")
    print(f"  ╟{'─'*58}╢")
    print(f"  ║  GENERATE 平均耗时:")
    print(f"  ║    llm:       {video_avg['cost_llm']*1000:6.1f}ms  "
          f"tts_prep: {video_avg['cost_tts_prep']*1000:6.1f}ms  "
          f"tts: {video_avg['cost_tts']*1000:6.1f}ms  "
          f"token2wav: {video_avg['cost_token2wav']*1000:6.1f}ms")
    print(f"  ║    generate_all:   {video_avg['cost_generate_all']*1000:6.1f}ms")
    print(f"  ╟{'─'*58}╢")
    print(f"  ║  WALL 平均耗时 (外部计时):")
    print(f"  ║    wall_prefill:  {video_avg['wall_prefill']*1000:6.1f}ms  "
          f"wall_generate: {video_avg['wall_generate']*1000:6.1f}ms  "
          f"wall_total: {video_avg['wall_chunk_total']*1000:6.1f}ms")
    print(f"  ╟{'─'*58}╢")
    print(f"  ║  INTERNAL TOTAL 平均: {video_avg['cost_chunk_total']*1000:6.1f}ms  "
          f"(prefill {video_avg['cost_prefill_all']*1000:.1f} + generate {video_avg['cost_generate_all']*1000:.1f})")
    print(f"  ║  tokens/chunk: {video_avg['n_tokens']:.2f}   tts_tokens/chunk: {video_avg['n_tts_tokens']:.2f}")
    print(f"  ╚{'═'*58}╝")


def main():
    parser = argparse.ArgumentParser(description="批量评估 MiniCPM-o Duplex 模型（直接扫描视频目录）")
    parser.add_argument("--model_path", type=str, default="/root/MiniCPM-o-4_5",
                        help="模型路径")
    parser.add_argument("--video_dir", type=str,
                        default="/root/omni_duplex_eval/omni_demo_zh_duplex",
                        help="视频所在目录，会扫描其中所有 .mp4 文件")
    parser.add_argument("--ref_audio", type=str,
                        default="/root/MiniCPM-o-4_5/assets/HT_ref_audio.wav",
                        help="参考音频路径")
    parser.add_argument("--output_dir", type=str,
                        default="/root/test/eval_report",
                        help="输出目录（JSON + 渲染视频 + HTML）")
    parser.add_argument("--videos", type=str, default=None,
                        help="指定视频文件名列表，逗号分隔。不指定则运行目录下全部 mp4")
    args = parser.parse_args()

    # 扫描视频文件
    video_dir = Path(args.video_dir)
    if args.videos:
        video_files = []
        for name in args.videos.split(","):
            name = name.strip()
            p = video_dir / name
            if p.exists():
                video_files.append(p)
            else:
                print(f"[WARN] 指定的视频不存在: {p}")
    else:
        video_files = sorted(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"[ERROR] 在 {video_dir} 中未找到任何 mp4 文件")
        return

    print(f"[INFO] 找到 {len(video_files)} 个视频文件")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = output_dir / "rendered"
    rendered_dir.mkdir(exist_ok=True)
    source_videos_dir = output_dir / "videos"
    source_videos_dir.mkdir(exist_ok=True)

    # 加载模型
    model = load_model(args.model_path)

    all_results = []
    all_chunks_flat = []

    for vid_idx, video_path in enumerate(video_files):
        video_name = video_path.name
        output_video_name = f"output_{video_path.stem}.mp4"
        output_video_path = str(rendered_dir / output_video_name)

        print(f"\n{'='*70}")
        print(f"  [{vid_idx+1}/{len(video_files)}] {video_name}")
        print(f"{'='*70}")

        t0 = time.time()
        chunk_results = evaluate_single_video(
            model, str(video_path), args.ref_audio, output_video_path
        )
        total_time = time.time() - t0

        # 计算该视频的平均指标
        video_avg = compute_averages(chunk_results, ALL_METRIC_KEYS)

        # 打印汇总
        print_video_summary(video_avg, total_time)

        # 软链接原始视频到 videos/ 目录
        src_link = source_videos_dir / video_name
        if src_link.exists() or src_link.is_symlink():
            src_link.unlink()
        src_link.symlink_to(video_path.resolve())

        video_result = {
            "video_name": video_name,
            "video_path": str(video_path.resolve()),
            "source_video_rel": f"videos/{video_name}",
            "rendered_video_rel": f"rendered/{output_video_name}",
            "total_inference_time": total_time,
            "chunk_results": chunk_results,
            "averages": video_avg,
            "model_output_text": "".join(c["text"] for c in chunk_results),
        }
        all_results.append(video_result)
        all_chunks_flat.extend(chunk_results)

    # 计算所有数据的总体平均值
    overall_avg = compute_averages(all_chunks_flat, ALL_METRIC_KEYS)

    output_data = {
        "metadata": {
            "model_path": args.model_path,
            "video_dir": str(video_dir.resolve()),
            "ref_audio": args.ref_audio,
            "total_videos": len(all_results),
            "total_chunks": len(all_chunks_flat),
            "eval_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "overall_averages": overall_avg,
        "video_results": all_results,
    }

    # 保存 JSON
    json_path = output_dir / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 打印总体统计
    print(f"\n{'='*70}")
    print(f"  [DONE] 全部评估完成！")
    print(f"{'='*70}")
    print(f"  结果 JSON: {json_path}")
    print(f"  渲染视频:  {rendered_dir}/ ({len(all_results)} 个)")
    print(f"  总视频数:  {len(all_results)}")
    print(f"  总chunk数: {len(all_chunks_flat)}")
    print(f"\n  ── 总体平均耗时 ──")
    print(f"  PREFILL:")
    print(f"    vision_process: {overall_avg['cost_vision_process']*1000:6.1f}ms  "
          f"vision_embed(ViT): {overall_avg['cost_vision_embed']*1000:6.1f}ms  "
          f"vision_feed: {overall_avg['cost_vision_feed']*1000:6.1f}ms")
    print(f"    audio_process:  {overall_avg['cost_audio_process']*1000:6.1f}ms  "
          f"audio_embed:       {overall_avg['cost_audio_embed']*1000:6.1f}ms  "
          f"audio_feed:  {overall_avg['cost_audio_feed']*1000:6.1f}ms")
    print(f"    prefill_all:    {overall_avg['cost_prefill_all']*1000:6.1f}ms")
    print(f"  GENERATE:")
    print(f"    llm: {overall_avg['cost_llm']*1000:6.1f}ms  "
          f"tts_prep: {overall_avg['cost_tts_prep']*1000:6.1f}ms  "
          f"tts: {overall_avg['cost_tts']*1000:6.1f}ms  "
          f"token2wav: {overall_avg['cost_token2wav']*1000:6.1f}ms")
    print(f"    generate_all:   {overall_avg['cost_generate_all']*1000:6.1f}ms")
    print(f"  WALL (外部计时):")
    print(f"    wall_prefill:   {overall_avg['wall_prefill']*1000:6.1f}ms  "
          f"wall_generate: {overall_avg['wall_generate']*1000:6.1f}ms  "
          f"wall_total: {overall_avg['wall_chunk_total']*1000:6.1f}ms")
    print(f"  INTERNAL TOTAL:")
    print(f"    chunk_total:    {overall_avg['cost_chunk_total']*1000:6.1f}ms  "
          f"(prefill {overall_avg['cost_prefill_all']*1000:.1f} + generate {overall_avg['cost_generate_all']*1000:.1f})")
    print(f"    tokens/chunk: {overall_avg['n_tokens']:.2f}   tts_tokens/chunk: {overall_avg['n_tts_tokens']:.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
