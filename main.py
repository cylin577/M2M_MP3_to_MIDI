import argparse
import os
from pydub import AudioSegment
# 使用 numpy.fft 代替 scipy.fft，以兼容 numba
import numpy as np
from numpy.fft import fft
from mido import Message, MidiFile, MidiTrack, MetaMessage
from multiprocessing import Pool
import numba

# -----------------------------------------------------------------------------
# 固定参数
# -----------------------------------------------------------------------------
FRAME_SIZE      = 4096         # FFT 大小
HOP_SIZE        = 256          # 帧跳长
SAMPLE_RATE     = 44100        # 采样率
NUM_PEAKS       = 8            # 每帧只保留最强的 8 个峰值
MAG_THRESHOLD   = 30000        # 能量阈值：FFT 幅度必须大于此值才算“峰”
VELOCITY_SCALE  = 127_000      # FFT 幅度 -> MIDI velocity 缩放

# MIDI 时间参数
TICKS_PER_BEAT     = 480
BPM                = 120
MICROSEC_PER_BEAT  = int(60_000_000 / BPM)
FRAME_DURATION_MS  = (HOP_SIZE / SAMPLE_RATE) * 1000
TICK_DURATION_MS   = MICROSEC_PER_BEAT / TICKS_PER_BEAT / 1000
TICKS_PER_FRAME    = round(FRAME_DURATION_MS / TICK_DURATION_MS)

# 全局变量，用于多进程子进程访问
_SAMPLES = None
_WINDOW = None
_HALF_FFT = FRAME_SIZE // 2

# -----------------------------------------------------------------------------
# 工具函数：三点抛物线拟合，用于提升峰值频率的精度（ numba 加速 + 缓存 ）
# -----------------------------------------------------------------------------
@numba.njit(cache=True)
def parabolic_interpolation(mag_spectrum, k):
    if k <= 0 or k >= mag_spectrum.shape[0] - 1:
        return 0.0
    alpha = mag_spectrum[k - 1]
    beta = mag_spectrum[k]
    gamma = mag_spectrum[k + 1]
    denom = (alpha - 2 * beta + gamma)
    if denom == 0.0:
        return 0.0
    return 0.5 * (alpha - gamma) / denom

# -----------------------------------------------------------------------------
# 工具函数：Hz 转换到最接近的 MIDI 音符号号（ numba 加速 + 缓存 ）
# -----------------------------------------------------------------------------
@numba.njit(cache=True)
def hz_to_midi(freq):
    return int(np.round(69.0 + 12.0 * np.log2(freq / 440.0)))

# -----------------------------------------------------------------------------
# 帧级预扫描函数：返回第 i 帧中满足 MAG_THRESHOLD 的所有峰值频率
# 供多进程并行调用，访问全局 _SAMPLES、_WINDOW、_HALF_FFT
# -----------------------------------------------------------------------------
def scan_frame_for_freqs(i):
    start = i * HOP_SIZE
    frame = _SAMPLES[start : start + FRAME_SIZE] * _WINDOW
    spectrum = np.abs(fft(frame)[:_HALF_FFT])
    # 取前 NUM_PEAKS 峰值
    peak_indices = np.argsort(spectrum)[-NUM_PEAKS:][::-1]
    freqs = []
    for k in peak_indices:
        mag = spectrum[k]
        if mag < MAG_THRESHOLD:
            continue
        delta = parabolic_interpolation(spectrum, k)
        true_bin = k + delta
        freq = true_bin * SAMPLE_RATE / FRAME_SIZE
        if freq > 0:
            freqs.append(freq)
    return freqs

# -----------------------------------------------------------------------------
# 帧级正式提取函数：返回第 i 帧中符合动态频率范围的音符及力度字典
# 供多进程并行调用，访问全局 _SAMPLES、_WINDOW、_HALF_FFT
# -----------------------------------------------------------------------------
def scan_frame_for_notes(args_tuple):
    i, min_freq, max_freq = args_tuple
    start = i * HOP_SIZE
    frame = _SAMPLES[start : start + FRAME_SIZE] * _WINDOW
    spectrum = np.abs(fft(frame)[:_HALF_FFT])
    peak_indices = np.argsort(spectrum)[-NUM_PEAKS:][::-1]
    notes_dict = {}
    for k in peak_indices:
        mag = spectrum[k]
        if mag < MAG_THRESHOLD:
            continue
        delta = parabolic_interpolation(spectrum, k)
        true_bin = k + delta
        freq = true_bin * SAMPLE_RATE / FRAME_SIZE
        # 应用动态频率上下限过滤
        if freq < min_freq or freq > max_freq:
            continue
        midi_note = hz_to_midi(freq)
        if midi_note < 0 or midi_note > 127:
            continue
        vel = int(min(127, (mag / VELOCITY_SCALE) * 127))
        if vel > 0:
            notes_dict[midi_note] = vel
    return notes_dict

# -----------------------------------------------------------------------------
# 核心：根据音频文件自动计算频率上下限、再生成 MIDI（含并行化）
# -----------------------------------------------------------------------------
def convert_to_piano_clone(input_path, output_path):
    global _SAMPLES, _WINDOW
    # 1) 载入 MP3，强制为单声道并重采样到 SAMPLE_RATE
    audio = AudioSegment.from_mp3(input_path).set_channels(1).set_frame_rate(SAMPLE_RATE)
    _SAMPLES = np.array(audio.get_array_of_samples(), dtype=np.float32)
    _WINDOW = np.hanning(FRAME_SIZE)

    total_frames = (len(_SAMPLES) - FRAME_SIZE) // HOP_SIZE + 1

    # -----------------------------------------------------------------------------
    # 2) 第一遍并行预扫描：遍历每一帧，收集所有峰值频率，计算 dynamic_min_freq / dynamic_max_freq
    # -----------------------------------------------------------------------------
    with Pool() as pool:
        # pool.map：把 0..total_frames-1 分发到多个进程
        all_freq_lists = pool.map(scan_frame_for_freqs, range(total_frames))
    # 将每帧频率合并成一个大列表
    all_freqs = []
    for sublist in all_freq_lists:
        all_freqs.extend(sublist)

    # 计算动态频率上下限
    if len(all_freqs) == 0:
        dynamic_min_freq = 50.0
        dynamic_max_freq = 4000.0
        print("[!] 警告：未检测到任何峰值。使用默认频率范围 50 Hz - 4000 Hz")
    else:
        dynamic_min_freq = min(all_freqs)
        dynamic_max_freq = max(all_freqs)
        print(f"[✓] 自动计算频率范围: {dynamic_min_freq:.1f} Hz ～ {dynamic_max_freq:.1f} Hz")

    # -----------------------------------------------------------------------------
    # 3) 第二遍并行正式提取：遍历每一帧，返回每帧的 {note: velocity} 字典
    # -----------------------------------------------------------------------------
    # 为 pool.map 传递参数 (i, min_freq, max_freq)
    task_args = [(i, dynamic_min_freq, dynamic_max_freq) for i in range(total_frames)]
    with Pool() as pool:
        frames_notes = pool.map(scan_frame_for_notes, task_args)
    # frames_notes 是长度 total_frames 的列表，每项是当前帧的 {note: velocity}

    # -----------------------------------------------------------------------------
    # 4) 简单“连续两帧才触发”去噪：只有当某音符在当前帧和上一帧都出现，才视为真正出现
    # -----------------------------------------------------------------------------
    filtered_frames = []
    for idx, curr in enumerate(frames_notes):
        if idx == 0:
            filtered_frames.append({})
            continue
        prev = frames_notes[idx - 1]
        sustained = {note: curr[note] for note in curr if note in prev}
        filtered_frames.append(sustained)

    # -----------------------------------------------------------------------------
    # 5) 根据 filtered_frames 生成 note_on / note_off 事件
    # -----------------------------------------------------------------------------
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=MICROSEC_PER_BEAT, time=0))

    active_notes = set()
    events = []
    for idx, curr in enumerate(filtered_frames):
        t_tick = idx * TICKS_PER_FRAME
        prev = filtered_frames[idx - 1] if idx > 0 else {}
        # note_off: prev 有、curr 没、有激活状态
        for note in prev:
            if note not in curr and note in active_notes:
                vel = prev[note]
                events.append((t_tick, 'off', note, vel))
                active_notes.remove(note)
        # note_on: curr 有、prev 没
        for note, vel in curr.items():
            if note not in prev:
                events.append((t_tick, 'on', note, vel))
                active_notes.add(note)
    # 结束时强制关闭所有还未关闭的音符
    end_tick = total_frames * TICKS_PER_FRAME
    for note in list(active_notes):
        events.append((end_tick, 'off', note, 0))

    # -----------------------------------------------------------------------------
    # 6) 排序并写入 MIDI Track，保存文件
    # -----------------------------------------------------------------------------
    events.sort(key=lambda e: e[0])
    last_tick = 0
    for abs_tick, etype, note, vel in events:
        delta = abs_tick - last_tick
        if etype == 'on':
            track.append(Message('note_on', note=note, velocity=vel, time=delta))
        else:
            track.append(Message('note_off', note=note, velocity=vel, time=delta))
        last_tick = abs_tick

    mid.save(output_path)
    print(f"[✓] MIDI saved to: {output_path}")

# -----------------------------------------------------------------------------
# CLI 部分：只接受 input 和 output，MIN_FREQ/MAX_FREQ 全自动计算
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MP3 to MIDI"
    )
    parser.add_argument("input", help="Path to MP3 file")
    parser.add_argument("-o", "--output", default="out.mid", help="Output MIDI path，default is ./out.mid")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input file not found。")
        return

    convert_to_piano_clone(args.input, args.output)

if __name__ == "__main__":
    main()
