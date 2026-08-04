"""Hill-climbing parameter tuner for Audio2Midi.

Layout:
    orignal/  -> input MP3 files  (note: folder name as the user spelled it)
    sample/   -> reference MID files, matched by base name (ignoring extension)
                e.g. orignal/foo.mp3 <-> sample/foo.mid

Run:
    python tune.py
    python tune.py --rounds 50 --patience 5 --tol 0.05 --output best_params.json
    python tune.py --dry-run         # evaluate default params only, no climb

The hill climber perturbs the 6 tunable parameters (FRAME_SIZE, HOP_SIZE,
NUM_PEAKS, MAG_THRESHOLD, VELOCITY_SCALE, BPM), converts every mp3 in
orignal/ to MIDI with the candidate params, scores each against its
reference MID using an onset-aware metric, and moves to the best neighbor
if it beats the current best mean score. Search stops after `patience`
consecutive rounds with no improvement.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from copy import deepcopy

import numpy as np
from mido import MidiFile

# Silence pydub's invalid-escape SyntaxWarnings (upstream bug, harmless).
warnings.filterwarnings("ignore", category=SyntaxWarning)

import main

ORIGINAL_DIR = "orignal"
SAMPLE_DIR  = "sample"

# Parameters tuned and their search strategy.
# Each entry: (key, [(step, factor), ...]) where a neighbor perturbs the
# current value by `step` for integer params, or multiplies by `factor` for
# float-ish params. We provide both up and down perturbations implicitly by
# using signed steps / both <1 and >1 factors.
TUNABLE = {
    "FRAME_SIZE":     {"type": "int",   "steps": [512, 1024, 2048], "min": 1024, "max": 16384},
    "HOP_SIZE":       {"type": "int",   "steps": [64, 128, 256, 512], "min": 32,  "max": 2048},
    "NUM_PEAKS":      {"type": "int",   "steps": [1, 2, 4],           "min": 1,   "max": 32},
    "MAG_THRESHOLD":  {"type": "float", "factors": [0.5, 0.8, 1.25, 2.0], "min": 100, "max": 1e7},
    "VELOCITY_SCALE": {"type": "float", "factors": [0.5, 0.8, 1.25, 2.0], "min": 100, "max": 1e8},
    "BPM":            {"type": "int",   "steps": [10, 20, 40],        "min": 40,  "max": 300},
}

# Fixed (non-tunable) defaults carried through.
FIXED = {
    "SAMPLE_RATE":    main.DEFAULT_PARAMS["SAMPLE_RATE"],
    "TICKS_PER_BEAT": main.DEFAULT_PARAMS["TICKS_PER_BEAT"],
}

# -----------------------------------------------------------------------------
# MIDI loading: extract note_on onsets as (time_seconds, note, velocity)
# -----------------------------------------------------------------------------
def load_onsets(mid_path):
    """Return a list of (time_seconds, note, velocity) for every note_on event
    with velocity > 0. Handles tempo changes via mido.tick2second."""
    mid = MidiFile(mid_path)
    onsets = []
    abs_tick = 0
    # mido's MidiFile merges all tracks when iterated, with delta times.
    # We need per-track tempo handling, but the merged iterator already
    # applies tempo correctly via mido's built-in tempo tracking only if
    # using mid.tracks manually. Use the simple, robust path:
    current_tempo = 500000  # microseconds per beat, default 120 BPM
    ticks_per_beat = mid.ticks_per_beat
    for msg in mid:
        abs_tick += msg.time
        if msg.type == 'set_tempo':
            current_tempo = msg.tempo
        elif msg.type == 'note_on' and msg.velocity > 0:
            sec = mido_tick2second(abs_tick, ticks_per_beat, current_tempo)
            onsets.append((sec, msg.note, msg.velocity))
    return onsets

def mido_tick2second(tick, ticks_per_beat, tempo_usec):
    return tick * (tempo_usec / 1_000_000.0) / ticks_per_beat

# -----------------------------------------------------------------------------
# Fitness: compare generated onsets to reference onsets.
# Returns a score in [0, 1]. Higher is better.
#
# Components:
#   - Pitch+onset match: for each generated onset, find nearest reference
#     onset with the same note within `tol` seconds. Reward = 1 - dt/tol.
#     Each reference onset can be matched at most once (greedy).
#   - Velocity match: for matched pairs, add (1 - |v1-v2|/127) * 0.3 weight.
#   - Precision/recall: penalize unmatched generated (FP) and unmatched
#     reference (FN) onsets.
# -----------------------------------------------------------------------------
def fitness(gen_onsets, ref_onsets, tol=0.05):
    gen = sorted(gen_onsets)
    ref = sorted(ref_onsets)
    n_gen = len(gen)
    n_ref = len(ref)
    if n_ref == 0 and n_gen == 0:
        return 1.0
    if n_ref == 0 or n_gen == 0:
        return 0.0

    # Group reference by note for fast lookup.
    ref_by_note = {}
    for idx, (t, note, v) in enumerate(ref):
        ref_by_note.setdefault(note, []).append((t, v, idx))

    matched_ref = set()
    pitch_score = 0.0
    vel_score   = 0.0
    matched = 0

    for (t_g, note_g, v_g) in gen:
        cands = ref_by_note.get(note_g, [])
        best_idx = -1
        best_dt = None
        for (t_r, v_r, ridx) in cands:
            if ridx in matched_ref:
                continue
            dt = abs(t_g - t_r)
            if dt > tol:
                continue
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_idx = ridx
            # velocity candidate stored alongside
            best_v = v_r
        if best_idx >= 0:
            # re-read v for the chosen candidate
            for (t_r, v_r, ridx) in cands:
                if ridx == best_idx:
                    best_v = v_r
                    break
            matched_ref.add(best_idx)
            matched += 1
            pitch_score += 1.0 - (best_dt / tol)
            vel_score   += 1.0 - (abs(v_g - best_v) / 127.0)

    # Normalize components
    precision = matched / max(n_gen, 1)            # fraction of gen that hit
    recall    = matched / max(n_ref, 1)            # fraction of ref covered
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    pitch_norm = pitch_score / max(n_ref, 1)
    vel_norm   = vel_score   / max(matched, 1) if matched > 0 else 0.0

    # Weights: F1 dominates, then onset tightness, then velocity.
    score = 0.6 * f1 + 0.25 * pitch_norm + 0.15 * vel_norm
    return float(max(0.0, min(1.0, score)))

# -----------------------------------------------------------------------------
# Pair discovery
# -----------------------------------------------------------------------------
def discover_pairs():
    pairs = []
    if not os.path.isdir(ORIGINAL_DIR):
        print(f"[!] Missing folder: {ORIGINAL_DIR}/")
        return pairs
    if not os.path.isdir(SAMPLE_DIR):
        print(f"[!] Missing folder: {SAMPLE_DIR}/")
        return pairs
    for fname in sorted(os.listdir(ORIGINAL_DIR)):
        if not fname.lower().endswith(".mp3"):
            continue
        base = os.path.splitext(fname)[0]
        ref_path = os.path.join(SAMPLE_DIR, base + ".mid")
        if not os.path.isfile(ref_path):
            print(f"[!] No reference for {fname} (expected {ref_path}); skipping.")
            continue
        pairs.append((os.path.join(ORIGINAL_DIR, fname), ref_path))
    return pairs

# -----------------------------------------------------------------------------
# Evaluate a parameter set across all pairs. Returns mean fitness.
# -----------------------------------------------------------------------------
def evaluate(params, pairs, tol=0.05, verbose=False):
    if not pairs:
        return 0.0
    scores = []
    tmp = tempfile.mkdtemp(prefix="a2m_tune_")
    for (mp3, ref) in pairs:
        out_mid = os.path.join(tmp, os.path.splitext(os.path.basename(mp3))[0] + ".mid")
        # Suppress convert's own prints if verbose=False
        if not verbose:
            with _suppress_stdout():
                try:
                    main.convert_to_piano_clone(mp3, out_mid, params=params)
                except Exception as e:
                    print(f"[!] Conversion failed for {mp3}: {e}")
                    scores.append(0.0)
                    continue
        else:
            try:
                main.convert_to_piano_clone(mp3, out_mid, params=params)
            except Exception as e:
                print(f"[!] Conversion failed for {mp3}: {e}")
                scores.append(0.0)
                continue
        try:
            gen = load_onsets(out_mid)
            ref = load_onsets(ref)
            s = fitness(gen, ref, tol=tol)
        except Exception as e:
            print(f"[!] Scoring failed for {mp3}: {e}")
            s = 0.0
        scores.append(s)
        if verbose:
            print(f"    {os.path.basename(mp3)}: score={s:.4f}")
    return float(np.mean(scores)) if scores else 0.0

# -----------------------------------------------------------------------------
# Neighbor generation
# -----------------------------------------------------------------------------
def neighbors(params):
    """Yield (label, params_dict) for every perturbation of every tunable key."""
    for key, spec in TUNABLE.items():
        cur = params[key]
        if spec["type"] == "int":
            for step in spec["steps"]:
                for delta in (step, -step):
                    new = cur + delta
                    if new < spec["min"] or new > spec["max"]:
                        continue
                    p = deepcopy(params)
                    p[key] = int(new)
                    yield (f"{key}{delta:+d}", p)
        else:  # float
            for factor in spec["factors"]:
                new = cur * factor
                if new < spec["min"] or new > spec["max"]:
                    continue
                p = deepcopy(params)
                p[key] = float(new)
                # Ensure power-of-2-ish FFT sizes can still be valid: skip
                # any FRAME_SIZE / HOP_SIZE perturbation that produces a
                # combination where FRAME_SIZE < HOP_SIZE.
                if p["FRAME_SIZE"] <= p["HOP_SIZE"]:
                    continue
                yield (f"{key}x{factor:g}", p)

# -----------------------------------------------------------------------------
# Hill climber
# -----------------------------------------------------------------------------
def hill_climb(pairs, max_rounds=200, patience=10, tol=0.05, verbose=True):
    current = deepcopy(main.DEFAULT_PARAMS)
    # Sanity: ensure FRAME_SIZE > HOP_SIZE
    if current["FRAME_SIZE"] <= current["HOP_SIZE"]:
        current["HOP_SIZE"] = current["FRAME_SIZE"] // 4

    print("[*] Evaluating default params...")
    best_score = evaluate(current, pairs, tol=tol, verbose=verbose)
    print(f"[*] Default score: {best_score:.4f}\n")

    history = [(deepcopy(current), best_score)]
    no_improve = 0
    for rnd in range(1, max_rounds + 1):
        t0 = time.time()
        best_neighbor = None
        best_neighbor_score = best_score
        best_neighbor_label = None
        n_tried = 0
        for (label, cand) in neighbors(current):
            n_tried += 1
            s = evaluate(cand, pairs, tol=tol, verbose=False)
            if s > best_neighbor_score + 1e-6:
                best_neighbor_score = s
                best_neighbor = cand
                best_neighbor_label = label
        dt = time.time() - t0
        if best_neighbor is not None:
            current = best_neighbor
            best_score = best_neighbor_score
            no_improve = 0
            history.append((deepcopy(current), best_score))
            print(f"[round {rnd}] IMPROVE via {best_neighbor_label} -> score={best_score:.4f}  ({n_tried} neighbors, {dt:.1f}s)")
        else:
            no_improve += 1
            print(f"[round {rnd}] no improvement ({n_tried} neighbors, {dt:.1f}s). patience {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"[*] Stopping after {patience} rounds with no improvement.")
                break
    return current, best_score, history

# -----------------------------------------------------------------------------
# stdout suppression helper
# -----------------------------------------------------------------------------
import contextlib, io

@contextlib.contextmanager
def _suppress_stdout():
    saved = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="Hill-climb Audio2Midi parameters against reference MIDIs.")
    parser.add_argument("--rounds", type=int, default=200, help="Max hill-climb rounds.")
    parser.add_argument("--patience", type=int, default=10, help="Stop after this many rounds with no improvement.")
    parser.add_argument("--tol", type=float, default=0.05, help="Onset match tolerance in seconds.")
    parser.add_argument("--output", default="best_params.json", help="Where to save best params.")
    parser.add_argument("--dry-run", action="store_true", help="Score default params and exit (no climbing).")
    parser.add_argument("--verbose", action="store_true", help="Print per-file scores during eval.")
    args = parser.parse_args()

    pairs = discover_pairs()
    if not pairs:
        print(f"[!] No usable pairs found. Put MP3s in ./{ORIGINAL_DIR}/ and matching .mid files in ./{SAMPLE_DIR}/.")
        return 1
    print(f"[*] Found {len(pairs)} pair(s).")
    for (mp3, ref) in pairs:
        print(f"    {os.path.basename(mp3)}  <->  {os.path.basename(ref)}")

    if args.dry_run:
        s = evaluate(main.DEFAULT_PARAMS, pairs, tol=args.tol, verbose=True)
        print(f"[*] Default params score: {s:.4f}")
        return 0

    best_params, best_score, history = hill_climb(
        pairs,
        max_rounds=args.rounds,
        patience=args.patience,
        tol=args.tol,
        verbose=args.verbose,
    )
    print("\n[*] Best score:", f"{best_score:.4f}")
    print("[*] Best params:")
    print(json.dumps(best_params, indent=2))
    with open(args.output, "w") as fh:
        json.dump(best_params, fh, indent=2)
    print(f"[*] Saved to {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main_cli())
