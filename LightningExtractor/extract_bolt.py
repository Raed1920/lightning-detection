#!/usr/bin/env python3
"""
extract_flash.py - For each lightning event, export the exact video frame at
the moment the lightning line appears, completely unmodified.

The enhanced clips are RIFE 2x interpolated: original camera frames sit at
even frame indices, AI-interpolated ones at odd. We pick the earliest
ORIGINAL camera frame where the flash is visible - the true instant of
capture - and save it untouched (no background removal, no masking).

Outputs: bolt_output/<video>/strike_NNN.png (lossless, unmodified frame)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
ENHANCED_DIR = BASE_DIR / "enhanced_output"
OUTPUT_DIR = BASE_DIR / "bolt_output"

FLASH_FRACTION = 0.35


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def read_frames(clip_path: Path):
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def estimate_background(frames):
    arr = np.stack(frames)
    g = arr.mean(axis=(1, 2, 3))
    n = max(3, len(frames) // 8)
    dark = np.argsort(g)[:n]
    return np.median(arr[dark].astype(np.float32), axis=0)


def flash_frames(frames, bg):
    g_bg = float(bg.mean())
    g = np.stack([f.astype(np.float32).mean() for f in frames])
    peak = g.max()
    if peak - g_bg <= 0.01:
        return []
    return list(np.nonzero((g - g_bg) > FLASH_FRACTION * (peak - g_bg))[0])


def capture_frame(frames, flashes):
    if not flashes:
        return int(np.argmax([f.mean() for f in frames]))
    originals = [i for i in flashes if i % 2 == 0]
    if originals:
        return min(originals)
    return min(flashes)


def extract(clip_path: Path, out_dir: Path):
    frames = read_frames(clip_path)
    if not frames:
        return None, "no frames decoded"
    bg = estimate_background(frames)
    flashes = flash_frames(frames, bg)
    idx = capture_frame(frames, flashes)
    cv2.imwrite(str(out_dir / f"{clip_path.stem}.png"), frames[idx])
    return idx, None


def process_video(video: str, out_root: Path, limit: int = 0):
    clips_dir = ENHANCED_DIR / video / "clips"
    clips = sorted(clips_dir.glob("*.mp4"))
    if not clips:
        return 0, 0
    if limit > 0:
        clips = clips[:limit]
    out_dir = out_root / video
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = []
    ok = 0
    for c in tqdm(clips, desc=video):
        _, err = extract(c, out_dir)
        if err:
            skipped.append((c.stem, err))
        else:
            ok += 1
    for s, e in skipped:
        eprint(f"  SKIP {video}/{s}: {e}")
    return ok, len(clips)


def main():
    ap = argparse.ArgumentParser(
        description="Export the untouched frame at the lightning instant"
    )
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Video names to process (default: all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N clips per video (0 = all)")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = args.videos or sorted(
        d.name for d in ENHANCED_DIR.iterdir()
        if (d / "clips").exists()
    )

    total_ok = 0
    total = 0
    for v in videos:
        ok, n = process_video(v, OUTPUT_DIR, args.limit)
        total_ok += ok
        total += n
        print(f"  {v}: {ok}/{n}")

    print(f"\nDone! {total_ok}/{total} frames exported")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
