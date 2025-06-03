import argparse
import os
from pydub import AudioSegment
from scipy.fft import fft
import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage

# PARAMETERS 
FRAME_SIZE      = 4096        # FFT size; larger → finer pitch resolution
HOP_SIZE        = 1024        # Frame hop (≈23 ms at 44.1 kHz)
SAMPLE_RATE     = 44100
NUM_PEAKS       = 8           # How many top peaks to pick each frame
MIN_FREQ        = 50          # Hz; drop anything below
MAX_FREQ        = 4000        # Hz; drop anything above
MAG_THRESHOLD   = 15000       # Drop very low-energy bins
VELOCITY_SCALE  = 127_000     # Scale raw FFT magnitude → [0..127]

# MIDI TIMING (match real time) 
TICKS_PER_BEAT     = 480
BPM                = 120
MICROSEC_PER_BEAT  = int(60_000_000 / BPM)      # 120 BPM → 500 000 µs per beat
FRAME_DURATION_MS  = (HOP_SIZE / SAMPLE_RATE) * 1000
TICK_DURATION_MS   = MICROSEC_PER_BEAT / TICKS_PER_BEAT / 1000
TICKS_PER_FRAME    = round(FRAME_DURATION_MS / TICK_DURATION_MS)

# UTILS
def parabolic_interpolation(mag_spectrum, k):
    """
    If bin k is a peak, use bins (k-1, k, k+1) to fit a parabola and return Δ.
    Δ = 0.5 * (mag[k-1] - mag[k+1]) / (mag[k-1] - 2*mag[k] + mag[k+1])
    """
    if k <= 0 or k >= len(mag_spectrum) - 1:
        return 0.0
    α = mag_spectrum[k - 1]
    β = mag_spectrum[k]
    γ = mag_spectrum[k + 1]
    denom = (α - 2 * β + γ)
    if denom == 0:
        return 0.0
    return 0.5 * (α - γ) / denom

def hz_to_midi(freq):
    """Convert Hz → nearest MIDI note (0–127)."""
    return int(round(69 + 12 * np.log2(freq / 440.0)))

# CORE CONVERSION
def convert_to_piano_clone(input_path, output_path):
    # 1) Load MP3, force to mono @ SAMPLE_RATE
    audio = AudioSegment.from_mp3(input_path).set_channels(1).set_frame_rate(SAMPLE_RATE)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # 2) Prepare a new MIDI file & track, insert tempo (120 BPM)
    mid   = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=MICROSEC_PER_BEAT, time=0))

    half_FFT = FRAME_SIZE // 2
    window   = np.hanning(FRAME_SIZE)

    # 3) For each frame, compute top-NUM_PEAKS peaks → build a dict {midi_note: velocity}
    frames = []
    for i in range(0, len(samples) - FRAME_SIZE, HOP_SIZE):
        frame    = samples[i : i + FRAME_SIZE] * window
        spectrum = np.abs(fft(frame)[:half_FFT])

        # pick the bins of the top-NUM_PEAKS magnitudes
        peak_bins = spectrum.argsort()[-NUM_PEAKS:][::-1]
        notes_this_frame = {}

        for k in peak_bins:
            mag = spectrum[k]
            if mag < MAG_THRESHOLD:
                continue

            δ = parabolic_interpolation(spectrum, k)
            true_bin = k + δ
            freq     = true_bin * SAMPLE_RATE / FRAME_SIZE
            if freq < MIN_FREQ or freq > MAX_FREQ:
                continue

            midi_note = hz_to_midi(freq)
            if midi_note < 0 or midi_note > 127:
                continue

            vel = int(min(127, (mag / VELOCITY_SCALE) * 127))
            if vel == 0:
                continue

            notes_this_frame[midi_note] = vel

        frames.append(notes_this_frame)

    # 4) Now emit note_on/note_off by comparing each frame’s note‐set to the previous
    active_notes = set()   # which MIDI notes are currently “on” in our track
    events = []            # list of (absolute_tick, 'on'|'off', note, velocity)

    for idx, curr in enumerate(frames):
        t_tick = idx * TICKS_PER_FRAME
        prev   = frames[idx - 1] if idx > 0 else {}

        # a) any note in prev that is NOT in curr → emit note_off at t_tick
        for note in prev:
            if note not in curr and note in active_notes:
                vel = prev[note]
                events.append((t_tick, 'off', note, vel))
                active_notes.remove(note)

        # b) any note in curr that is NOT in prev → emit note_on at t_tick
        for note, vel in curr.items():
            if note not in prev:
                events.append((t_tick, 'on', note, vel))
                active_notes.add(note)

    # 5) Turn off any notes still active at very end
    end_tick = len(frames) * TICKS_PER_FRAME
    for note in list(active_notes):
        # we can reuse the last frame’s velocity or just set 64
        events.append((end_tick, 'off', note, 0))
        # not strictly needed to remove from active_notes anymore

    # 6) Sort events by absolute tick, then convert to delta‐time and write to track
    events.sort(key=lambda e: e[0])
    last_tick = 0
    for abs_tick, etype, note, vel in events:
        delta = abs_tick - last_tick
        if etype == 'on':
            track.append(Message('note_on', note=note, velocity=vel, time=delta))
        else:
            track.append(Message('note_off', note=note, velocity=vel, time=delta))
        last_tick = abs_tick

    # 7) Save the MIDI
    mid.save(output_path)
    print(f"[✓] MIDI saved to: {output_path}")

# CLI HANDLER 
def main():
    parser = argparse.ArgumentParser(
        description="MP3 to MIDI"
    )
    parser.add_argument("input",  help="Path to input MP3 file")
    parser.add_argument("-o", "--output", default="out.mid",
                        help="Path to output MIDI file")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input file not found.")
        return

    convert_to_piano_clone(args.input, args.output)

if __name__ == "__main__":
    main()
