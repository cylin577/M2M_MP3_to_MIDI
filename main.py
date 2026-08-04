import argparse
import json
import os
from pydub import AudioSegment
# Use numpy.fft instead of scipy.fft for numba compatibility
import numpy as np
from numpy.fft import fft
from mido import Message, MidiFile, MidiTrack, MetaMessage
from multiprocessing import Pool
import numba

DEFAULT_PARAMS_PATH = "best_params.json"

# -----------------------------------------------------------------------------
# Default parameters (mutable for tuning). Also define scanning functions so
# they may read parameters at call-time via globals updated by apply_params().
# -----------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "FRAME_SIZE":     4096,
    "HOP_SIZE":       256,
    "SAMPLE_RATE":    44100,
    "NUM_PEAKS":      8,
    "MAG_THRESHOLD":  30000,
    "VELOCITY_SCALE": 127_000,
    "TICKS_PER_BEAT": 480,
    "BPM":            120,
}

# Active parameters (mutable at runtime). To avoid recompiling numba jitted
# helpers, only non-jitted scan functions read these globals.
FRAME_SIZE      = DEFAULT_PARAMS["FRAME_SIZE"]
HOP_SIZE        = DEFAULT_PARAMS["HOP_SIZE"]
SAMPLE_RATE     = DEFAULT_PARAMS["SAMPLE_RATE"]
NUM_PEAKS       = DEFAULT_PARAMS["NUM_PEAKS"]
MAG_THRESHOLD   = DEFAULT_PARAMS["MAG_THRESHOLD"]
VELOCITY_SCALE  = DEFAULT_PARAMS["VELOCITY_SCALE"]
TICKS_PER_BEAT  = DEFAULT_PARAMS["TICKS_PER_BEAT"]
BPM             = DEFAULT_PARAMS["BPM"]
MICROSEC_PER_BEAT = int(60_000_000 / BPM)
FRAME_DURATION_MS  = (HOP_SIZE / SAMPLE_RATE) * 1000
TICK_DURATION_MS   = MICROSEC_PER_BEAT / TICKS_PER_BEAT / 1000
TICKS_PER_FRAME    = round(FRAME_DURATION_MS / TICK_DURATION_MS)

# Global samples for multiprocessing
_SAMPLES = None
_WINDOW = None

def apply_params(p):
    """Update module-level parameters from a dict. Must be called before each
    conversion so scan_frame_for_freqs / scan_frame_for_notes see the new
    values. Recomputes derived timing globals as well."""
    global FRAME_SIZE, HOP_SIZE, SAMPLE_RATE, NUM_PEAKS
    global MAG_THRESHOLD, VELOCITY_SCALE, TICKS_PER_BEAT, BPM
    global MICROSEC_PER_BEAT, FRAME_DURATION_MS, TICK_DURATION_MS
    global TICKS_PER_FRAME, _HALF_FFT

    FRAME_SIZE      = int(p["FRAME_SIZE"])
    HOP_SIZE        = int(p["HOP_SIZE"])
    SAMPLE_RATE     = int(p["SAMPLE_RATE"])
    NUM_PEAKS       = int(p["NUM_PEAKS"])
    MAG_THRESHOLD   = float(p["MAG_THRESHOLD"])
    VELOCITY_SCALE  = float(p["VELOCITY_SCALE"])
    TICKS_PER_BEAT  = int(p["TICKS_PER_BEAT"])
    BPM             = int(p["BPM"])

    MICROSEC_PER_BEAT = int(60_000_000 / BPM)
    FRAME_DURATION_MS = (HOP_SIZE / SAMPLE_RATE) * 1000
    TICK_DURATION_MS  = MICROSEC_PER_BEAT / TICKS_PER_BEAT / 1000
    TICKS_PER_FRAME   = round(FRAME_DURATION_MS / TICK_DURATION_MS)
    _HALF_FFT          = FRAME_SIZE // 2

# -----------------------------------------------------------------------------
# Utility: Parabolic interpolation to improve peak frequency precision (numba)
# Note: shape[0] only, no params dependency -> safe to keep njit.
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
# Utility: Convert Hz to nearest MIDI note number (numba)
# -----------------------------------------------------------------------------
@numba.njit(cache=True)
def hz_to_midi(freq):
    return int(np.round(69.0 + 12.0 * np.log2(freq / 440.0)))

# Module-level globals read by multiprocessing workers
_G_FRAME_SIZE   = None
_G_HOP_SIZE     = None
_G_NUM_PEAKS    = None
_G_MAG_THRESH   = None
_G_VEL_SCALE    = None
_G_SAMPLE_RATE  = None
_G_HALF_FFT     = None
_G_SAMPLES      = None
_G_WINDOW       = None

def _sync_worker_locals():
    """Push current module params into the worker-side globals used by
    scan_* functions. Called in the parent before each Pool."""
    global _G_FRAME_SIZE, _G_HOP_SIZE, _G_NUM_PEAKS, _G_MAG_THRESH
    global _G_VEL_SCALE, _G_SAMPLE_RATE, _G_HALF_FFT, _G_SAMPLES, _G_WINDOW
    _G_FRAME_SIZE  = FRAME_SIZE
    _G_HOP_SIZE    = HOP_SIZE
    _G_NUM_PEAKS   = NUM_PEAKS
    _G_MAG_THRESH  = MAG_THRESHOLD
    _G_VEL_SCALE   = VELOCITY_SCALE
    _G_SAMPLE_RATE = SAMPLE_RATE
    _G_HALF_FFT    = FRAME_SIZE // 2
    _G_SAMPLES     = _SAMPLES
    _G_WINDOW      = _WINDOW

# -----------------------------------------------------------------------------
# Frame pre-scan: Return all peak frequencies in frame i that meet MAG_THRESHOLD
# Called in parallel by multiprocessing; reads worker-side globals.
# -----------------------------------------------------------------------------
def scan_frame_for_freqs(i):
    start = i * _G_HOP_SIZE
    frame = _G_SAMPLES[start : start + _G_FRAME_SIZE] * _G_WINDOW
    spectrum = np.abs(fft(frame)[:_G_HALF_FFT])
    # Keep top NUM_PEAKS peaks
    peak_indices = np.argsort(spectrum)[-_G_NUM_PEAKS:][::-1]
    freqs = []
    for k in peak_indices:
        mag = spectrum[k]
        if mag < _G_MAG_THRESH:
            continue
        delta = parabolic_interpolation(spectrum, k)
        true_bin = k + delta
        freq = true_bin * _G_SAMPLE_RATE / _G_FRAME_SIZE
        if freq > 0:
            freqs.append(freq)
    return freqs

def _scan_frame_for_freqs_init(samples, window, frame_size, hop_size,
                                num_peaks, mag_thresh, vel_scale, sample_rate):
    """Pool initializer: copy shared arrays + current params into worker globals.
    Workers are separate processes (fork), so globals must be set inside them."""
    global _G_SAMPLES, _G_WINDOW, _G_FRAME_SIZE, _G_HOP_SIZE
    global _G_NUM_PEAKS, _G_MAG_THRESH, _G_VEL_SCALE, _G_SAMPLE_RATE, _G_HALF_FFT
    _G_SAMPLES      = samples
    _G_WINDOW       = window
    _G_FRAME_SIZE   = frame_size
    _G_HOP_SIZE     = hop_size
    _G_NUM_PEAKS    = num_peaks
    _G_MAG_THRESH   = mag_thresh
    _G_VEL_SCALE    = vel_scale
    _G_SAMPLE_RATE  = sample_rate
    _G_HALF_FFT     = frame_size // 2

# -----------------------------------------------------------------------------
# Frame extraction: Return {note: velocity} dict for frame i within dynamic freq range
# Called in parallel by multiprocessing; reads worker-side globals.
# -----------------------------------------------------------------------------
def scan_frame_for_notes(args_tuple):
    i, min_freq, max_freq = args_tuple
    start = i * _G_HOP_SIZE
    frame = _G_SAMPLES[start : start + _G_FRAME_SIZE] * _G_WINDOW
    spectrum = np.abs(fft(frame)[:_G_HALF_FFT])
    peak_indices = np.argsort(spectrum)[-_G_NUM_PEAKS:][::-1]
    notes_dict = {}
    for k in peak_indices:
        mag = spectrum[k]
        if mag < _G_MAG_THRESH:
            continue
        delta = parabolic_interpolation(spectrum, k)
        true_bin = k + delta
        freq = true_bin * _G_SAMPLE_RATE / _G_FRAME_SIZE
        # Apply dynamic frequency range filter
        if freq < min_freq or freq > max_freq:
            continue
        midi_note = hz_to_midi(freq)
        if midi_note < 0 or midi_note > 127:
            continue
        vel = int(min(127, (mag / _G_VEL_SCALE) * 127))
        if vel > 0:
            notes_dict[midi_note] = vel
    return notes_dict

# -----------------------------------------------------------------------------
# Core: Auto-calculate frequency bounds from audio, then generate MIDI (parallelized)
# Accepts optional params dict; if None, uses DEFAULT_PARAMS.
# -----------------------------------------------------------------------------
def convert_to_piano_clone(input_path, output_path, params=None):
    """Convert an MP3 to MIDI using the given params dict (or DEFAULT_PARAMS).

    A fresh multiprocessing.Pool is created each call because (a) Pool workers
    inherit samples/window via an initializer, and those change per file, and
    (b) FRAME_SIZE/HOP_SIZE/etc. must be re-broadcast to workers each run.
    """
    if params is None:
        params = DEFAULT_PARAMS
    apply_params(params)

    global _SAMPLES, _WINDOW
    # 1) Load MP3, force mono and resample to SAMPLE_RATE
    audio = AudioSegment.from_mp3(input_path).set_channels(1).set_frame_rate(SAMPLE_RATE)
    _SAMPLES = np.array(audio.get_array_of_samples(), dtype=np.float32)
    _WINDOW = np.hanning(FRAME_SIZE)

    total_frames = (len(_SAMPLES) - FRAME_SIZE) // HOP_SIZE + 1

    # Push current params into worker-side globals (not strictly needed since
    # we use a Pool initializer, but kept for any synchronous inspection).
    _sync_worker_locals()

    # -----------------------------------------------------------------------------
    # 2) First parallel pre-scan: Iterate all frames, collect peak frequencies,
    #    compute dynamic_min_freq / dynamic_max_freq
    # -----------------------------------------------------------------------------
    with Pool(initializer=_scan_frame_for_freqs_init,
              initargs=(_SAMPLES, _WINDOW, FRAME_SIZE, HOP_SIZE,
                        NUM_PEAKS, MAG_THRESHOLD, VELOCITY_SCALE, SAMPLE_RATE)) as pool:
        # pool.map: Distribute 0..total_frames-1 across processes
        all_freq_lists = pool.map(scan_frame_for_freqs, range(total_frames))

    # Merge all frame frequencies into a single list
    all_freqs = []
    for sublist in all_freq_lists:
        all_freqs.extend(sublist)

    # Calculate dynamic frequency bounds
    if len(all_freqs) == 0:
        dynamic_min_freq = 50.0
        dynamic_max_freq = 4000.0
        print("[!] Warning: No peaks detected. Using default range 50 Hz - 4000 Hz")
    else:
        dynamic_min_freq = min(all_freqs)
        dynamic_max_freq = max(all_freqs)
        print(f"[OK] Auto-calculated frequency range: {dynamic_min_freq:.1f} Hz - {dynamic_max_freq:.1f} Hz")

    # -----------------------------------------------------------------------------
    # 3) Second parallel extraction: Iterate all frames, return {note: velocity} dict
    # -----------------------------------------------------------------------------
    # Prepare args (i, min_freq, max_freq) for pool.map
    task_args = [(i, dynamic_min_freq, dynamic_max_freq) for i in range(total_frames)]
    with Pool(initializer=_scan_frame_for_freqs_init,
              initargs=(_SAMPLES, _WINDOW, FRAME_SIZE, HOP_SIZE,
                        NUM_PEAKS, MAG_THRESHOLD, VELOCITY_SCALE, SAMPLE_RATE)) as pool:
        frames_notes = pool.map(scan_frame_for_notes, task_args)
    # frames_notes is a list of length total_frames, each item is {note: velocity} for that frame

    # -----------------------------------------------------------------------------
    # 4) Simple denoising: Only trigger note when it appears in both current and previous frame
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
    # 5) Generate note_on / note_off events from filtered_frames
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
        # note_off: Note in prev, not in curr, and active
        for note in prev:
            if note not in curr and note in active_notes:
                vel = prev[note]
                events.append((t_tick, 'off', note, vel))
                active_notes.remove(note)
        # note_on: Note in curr, not in prev
        for note, vel in curr.items():
            if note not in prev:
                events.append((t_tick, 'on', note, vel))
                active_notes.add(note)
    # Force-close any remaining active notes at the end
    end_tick = total_frames * TICKS_PER_FRAME
    for note in list(active_notes):
        events.append((end_tick, 'off', note, 0))

    # -----------------------------------------------------------------------------
    # 6) Sort and write events to MIDI track, save file
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
    print(f"[OK] MIDI saved to: {output_path}")

def _noop_init(*args, **kwargs):
    pass

# -----------------------------------------------------------------------------
# Helpers: load/save tuned params
# -----------------------------------------------------------------------------
def load_params(path):
    """Load a params dict from JSON. Missing keys fall back to DEFAULT_PARAMS.
    Returns None if file does not exist (caller may apply DEFAULT_PARAMS)."""
    if not os.path.isfile(path):
        return None
    with open(path, "r") as fh:
        p = json.load(fh)
    merged = dict(DEFAULT_PARAMS)
    merged.update(p)
    return merged

# -----------------------------------------------------------------------------
# CLI: Accepts input and output only, MIN_FREQ/MAX_FREQ are auto-calculated
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MP3 to MIDI"
    )
    parser.add_argument("input", help="Path to MP3 file")
    parser.add_argument("-o", "--output", default="out.mid", help="Output MIDI path, default is ./out.mid")
    parser.add_argument("--params", default=DEFAULT_PARAMS_PATH,
                        help=f"Params JSON path (default: {DEFAULT_PARAMS_PATH}). If file is missing, falls back to built-in defaults.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input file not found.")
        return

    params = load_params(args.params)
    if params is None:
        print(f"[!] Params file '{args.params}' not found. Using built-in defaults.")
        params = DEFAULT_PARAMS
    else:
        print(f"[*] Loaded params from {args.params}")

    convert_to_piano_clone(args.input, args.output, params=params)

if __name__ == "__main__":
    main()
