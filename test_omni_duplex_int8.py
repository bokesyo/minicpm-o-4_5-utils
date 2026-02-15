"""
MiniCPM-o Duplex 模型 - bitsandbytes INT8 量化测速脚本
功能：
  1. 使用 bitsandbytes 进行 INT8 量化加载模型
  2. 记录每个 chunk 的 prefill + generate 耗时
  3. 记录峰值 GPU 显存占用
  4. 输出汇总统计

CUDA_VISIBLE_DEVICES=1 python /root/test/test_omni_duplex_int8.py
"""

import time
import librosa
import torch
import numpy as np
from transformers import AutoModel, BitsAndBytesConfig
from minicpmo.utils import generate_duplex_video, get_video_frame_audio_segments


# ─── 配置 ───
MODEL_PATH = "/root/MiniCPM-o-4_5"
VIDEO_PATH = "/root/MiniCPM-o-4_5/assets/omni_duplex1.mp4"
REF_AUDIO_PATH = "/root/MiniCPM-o-4_5/assets/HT_ref_audio.wav"
OUTPUT_VIDEO_PATH = "duplex_output_int8.mp4"


def get_gpu_memory_mb():
    """获取当前 GPU 已分配显存（MB）"""
    return torch.cuda.memory_allocated() / (1024 ** 2)


def get_gpu_memory_reserved_mb():
    """获取当前 GPU 保留显存（MB）"""
    return torch.cuda.memory_reserved() / (1024 ** 2)


def get_gpu_peak_memory_mb():
    """获取 GPU 峰值已分配显存（MB）"""
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def get_gpu_peak_reserved_mb():
    """获取 GPU 峰值保留显存（MB）"""
    return torch.cuda.max_memory_reserved() / (1024 ** 2)


def print_gpu_memory(label=""):
    """打印当前 GPU 显存状态"""
    allocated = get_gpu_memory_mb()
    reserved = get_gpu_memory_reserved_mb()
    peak_alloc = get_gpu_peak_memory_mb()
    peak_reserved = get_gpu_peak_reserved_mb()
    print(f"  [GPU {label}] "
          f"已分配: {allocated:.1f}MB  保留: {reserved:.1f}MB  "
          f"峰值分配: {peak_alloc:.1f}MB  峰值保留: {peak_reserved:.1f}MB")


# ─── Step 1: 加载模型（bitsandbytes INT8 量化）───
print("=" * 70)
print("  MiniCPM-o Duplex - bitsandbytes INT8 量化测速")
print("=" * 70)

# 重置 GPU 显存统计
torch.cuda.reset_peak_memory_stats()
print_gpu_memory("模型加载前")

print(f"\n[INFO] 正在加载模型（INT8 量化）: {MODEL_PATH}")
t_load_start = time.time()

# bitsandbytes INT8 量化配置（仅对 LLM 部分量化，跳过 vision/audio/TTS 模块）
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=[
        "vpm",                    # SiglipVisionTransformer（视觉编码器）
        "resampler",              # Resampler（视觉重采样器）
        "apm",                    # MiniCPMWhisperEncoder（音频编码器）
        "audio_avg_pooler",       # AvgPool1d（音频池化）
        "audio_projection_layer", # MultiModalProjector（音频投影层）
        "tts",                    # MiniCPMTTS（TTS 模块）
    ],
)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto",
)

t_load_end = time.time()
model_load_time = t_load_end - t_load_start
print(f"[INFO] 模型加载完成，耗时: {model_load_time:.2f}s")
print_gpu_memory("模型加载后")

model.eval()
model = model.as_duplex()
print_gpu_memory("转为 duplex 模式后")

# ─── Step 2: 加载视频和参考音频 ───
print(f"\n[INFO] 加载视频: {VIDEO_PATH}")
ref_audio, _ = librosa.load(REF_AUDIO_PATH, sr=16000, mono=True)

video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
    VIDEO_PATH, stack_frames=1, use_ffmpeg=True, adjust_audio_length=True
)
print(f"[INFO] 视频帧数: {len(video_frames)}, 音频片段数: {len(audio_segments)}")

# ─── Step 3: 准备 duplex 会话 ───
model.prepare(
    prefix_system_prompt="Streaming Omni Conversation.",
    ref_audio=ref_audio,
    prompt_wav_path=REF_AUDIO_PATH,
)

# ─── Step 4: 逐 chunk 推理并计时 ───
results_log = []
timed_output_audio = []
chunk_timings = []

print(f"\n{'─'*70}")
print(f"  开始逐 chunk 推理（共 {len(audio_segments)} 个 chunks）")
print(f"{'─'*70}")

# 重置峰值统计，只统计推理阶段的峰值显存
torch.cuda.reset_peak_memory_stats()
mem_before_inference = get_gpu_memory_mb()

total_inference_start = time.time()

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

    # Step 4a: Streaming prefill
    t_pf_start = time.time()
    prefill_result = model.streaming_prefill(
        audio_waveform=audio_chunk,
        frame_list=frame_list,
        max_slice_nums=1,
        batch_vision_feed=False,
    )
    t_pf_end = time.time()
    wall_prefill = t_pf_end - t_pf_start

    # Step 4b: Streaming generate
    t_gen_start = time.time()
    result = model.streaming_generate(
        prompt_wav_path=REF_AUDIO_PATH,
        max_new_speak_tokens_per_chunk=20,
        decode_mode="sampling",
    )
    t_gen_end = time.time()
    wall_generate = t_gen_end - t_gen_start

    wall_total = wall_prefill + wall_generate

    # 记录耗时
    timing = {
        "chunk_idx": chunk_idx,
        "wall_prefill_ms": wall_prefill * 1000,
        "wall_generate_ms": wall_generate * 1000,
        "wall_total_ms": wall_total * 1000,
        "is_listen": result["is_listen"],
        "n_tokens": result.get("n_tokens", 0),
        "n_tts_tokens": result.get("n_tts_tokens", 0),
    }
    chunk_timings.append(timing)

    # 打印单 chunk 信息
    status = "LISTEN" if result["is_listen"] else "SPEAK "
    text_display = f'"{result["text"]}"' if result["text"] else ""
    print(f"  chunk {chunk_idx:3d} [{status}] "
          f"prefill={wall_prefill*1000:7.1f}ms  "
          f"generate={wall_generate*1000:7.1f}ms  "
          f"total={wall_total*1000:7.1f}ms  "
          f"tokens={result.get('n_tokens', 0)}  "
          f"{text_display}")

    if result["audio_waveform"] is not None:
        timed_output_audio.append((chunk_idx, result["audio_waveform"]))

    chunk_result = {
        "chunk_idx": chunk_idx,
        "is_listen": result["is_listen"],
        "text": result["text"],
        "end_of_turn": result["end_of_turn"],
        "current_time": result["current_time"],
        "audio_length": len(result["audio_waveform"]) if result["audio_waveform"] is not None else 0,
    }
    results_log.append(chunk_result)

total_inference_end = time.time()
total_inference_time = total_inference_end - total_inference_start

# ─── Step 5: 显存统计 ───
peak_allocated_mb = get_gpu_peak_memory_mb()
peak_reserved_mb = get_gpu_peak_reserved_mb()
final_allocated_mb = get_gpu_memory_mb()

# ─── Step 6: 汇总统计 ───
prefill_times = [t["wall_prefill_ms"] for t in chunk_timings]
generate_times = [t["wall_generate_ms"] for t in chunk_timings]
total_times = [t["wall_total_ms"] for t in chunk_timings]

speak_timings = [t for t in chunk_timings if not t["is_listen"]]
listen_timings = [t for t in chunk_timings if t["is_listen"]]

speak_totals = [t["wall_total_ms"] for t in speak_timings]
listen_totals = [t["wall_total_ms"] for t in listen_timings]

print(f"\n{'='*70}")
print(f"  INT8 量化测速结果汇总")
print(f"{'='*70}")

print(f"\n  ── 模型加载 ──")
print(f"  模型路径:       {MODEL_PATH}")
print(f"  量化方式:       bitsandbytes INT8")
print(f"  模型加载耗时:   {model_load_time:.2f}s")

print(f"\n  ── 显存占用 ──")
print(f"  推理前显存:     {mem_before_inference:.1f} MB")
print(f"  推理后显存:     {final_allocated_mb:.1f} MB")
print(f"  峰值已分配显存: {peak_allocated_mb:.1f} MB  ({peak_allocated_mb/1024:.2f} GB)")
print(f"  峰值保留显存:   {peak_reserved_mb:.1f} MB  ({peak_reserved_mb/1024:.2f} GB)")

print(f"\n  ── 推理耗时 ──")
print(f"  总 chunk 数:    {len(chunk_timings)}")
print(f"  speak chunks:   {len(speak_timings)}")
print(f"  listen chunks:  {len(listen_timings)}")
print(f"  总推理耗时:     {total_inference_time:.2f}s")

print(f"\n  ── 全部 chunk 平均耗时 ──")
print(f"  prefill:   {np.mean(prefill_times):7.1f}ms  (min={np.min(prefill_times):.1f}, max={np.max(prefill_times):.1f}, std={np.std(prefill_times):.1f})")
print(f"  generate:  {np.mean(generate_times):7.1f}ms  (min={np.min(generate_times):.1f}, max={np.max(generate_times):.1f}, std={np.std(generate_times):.1f})")
print(f"  total:     {np.mean(total_times):7.1f}ms  (min={np.min(total_times):.1f}, max={np.max(total_times):.1f}, std={np.std(total_times):.1f})")

if speak_totals:
    print(f"\n  ── SPEAK chunk 平均耗时 ──")
    print(f"  prefill:   {np.mean([t['wall_prefill_ms'] for t in speak_timings]):7.1f}ms")
    print(f"  generate:  {np.mean([t['wall_generate_ms'] for t in speak_timings]):7.1f}ms")
    print(f"  total:     {np.mean(speak_totals):7.1f}ms")

if listen_totals:
    print(f"\n  ── LISTEN chunk 平均耗时 ──")
    print(f"  prefill:   {np.mean([t['wall_prefill_ms'] for t in listen_timings]):7.1f}ms")
    print(f"  generate:  {np.mean([t['wall_generate_ms'] for t in listen_timings]):7.1f}ms")
    print(f"  total:     {np.mean(listen_totals):7.1f}ms")

print(f"\n{'='*70}")

# ─── Step 7: 生成输出视频 ───
print(f"\n[INFO] 正在生成输出视频: {OUTPUT_VIDEO_PATH}")
generate_duplex_video(
    video_path=VIDEO_PATH,
    output_video_path=OUTPUT_VIDEO_PATH,
    results_log=results_log,
    timed_output_audio=timed_output_audio,
    output_sample_rate=24000,
)
print(f"[INFO] 输出视频已保存: {OUTPUT_VIDEO_PATH}")
print_gpu_memory("最终状态")
