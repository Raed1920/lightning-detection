#!/usr/bin/env python3
"""
extract_trail.py - Isolate the lightning channel (trail) from enhanced clips.

Approach (validated against simple thresholding, which fragmented the trail):
  1. background = per-pixel median of the darkest frames
  2. flash frames = frames significantly brighter than the background
  3. delta MIP = per-pixel maximum of (flash - background) across flash frames
     -> recovers every branch of the strike, static sky removed
  4. grayscale top-hat: trail = delta_MIP - morphological_open(delta_MIP)
     The opening (disk kernel larger than the trail width) leaves only the
     broad fog envelope; subtracting it keeps the full channel (including its
     saturated interior) while removing the diffuse fog.
  5. threshold + speckle removal + small close to bridge channel gaps
  6. render the channel in its natural colors on a pure black background

Outputs (bolt_output/<video>/strike_NNN_*.png):
  - _trail.png  : the lightning channel only (natural colors, black bg)
  - _mip.png    : the full delta-MIP frame (reference)
Also writes report.csv with per-clip quality metrics.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
ENHANCED_DIR = BASE_DIR / "enhanced_output"
OUTPUT_DIR = BASE_DIR / "bolt_output"

FLASH_FRACTION = 0.35
DISK_RADIUS = 25
THRESH_MULT = 3.0
MIN_AREA = 150
CLOSE_KERNEL = 9


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


def clean_mask(mask, min_area, close_kernel):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k)
    return keep


def process_clip(clip_path: Path, out_dir: Path,
                 disk_r, thresh_mult, min_area, close_kernel):
    frames = read_frames(clip_path)
    if not frames:
        return None, "no frames decoded"
    bg = estimate_background(frames)
    flashes = flash_frames(frames, bg)
    if not flashes:
        return None, "no flash"

    delta_mip = np.max(
        np.stack([frames[i].astype(np.float32) - bg for i in flashes]), axis=0
    )
    gray = delta_mip.mean(axis=2)

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_r, disk_r))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, disk)
    top = np.clip(gray - opened, 0, None)

    mask = (top > thresh_mult * float(top.std())).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = clean_mask(mask, min_area, close_kernel)

    if mask.sum() == 0:
        return None, "no trail detected"

    mip_u8 = delta_mip.astype(np.uint8)
    trail = np.zeros_like(mip_u8)
    trail[mask > 0] = mip_u8[mask > 0]

    cv2.imwrite(str(out_dir / f"{clip_path.stem}_trail.png"), trail)
    cv2.imwrite(str(out_dir / f"{clip_path.stem}_mip.png"), mip_u8)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = sorted(stats[1:, 4], reverse=True)
    metrics = {
        "video": clip_path.parent.parent.name,
        "clip": clip_path.stem,
        "flash_frames": len(flashes),
        "trail_area_pct": round(100.0 * mask.mean(), 3),
        "components": n - 1,
        "largest_px": int(areas[0]) if areas else 0,
        "mean_brightness": round(float(top[mask > 0].mean()), 1),
    }
    return metrics, None


def main():
    ap = argparse.ArgumentParser(
        description="Top-hat based lightning trail extraction"
    )
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--disk-r", type=int, default=DISK_RADIUS)
    ap.add_argument("--thresh-mult", type=float, default=THRESH_MULT)
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--close-kernel", type=int, default=CLOSE_KERNEL)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = args.videos or sorted(
        d.name for d in ENHANCED_DIR.iterdir() if (d / "clips").exists()
    )

    rows = []
    total_ok = total = 0
    for v in videos:
        clips = sorted((ENHANCED_DIR / v / "clips").glob("*.mp4"))
        if args.limit > 0:
            clips = clips[:args.limit]
        out_dir = OUTPUT_DIR / v
        out_dir.mkdir(parents=True, exist_ok=True)
        skipped = []
        ok = 0
        for c in tqdm(clips, desc=v):
            m, err = process_clip(c, out_dir, args.disk_r, args.thresh_mult,
                                  args.min_area, args.close_kernel)
            if err:
                skipped.append((c.stem, err))
            else:
                rows.append(m)
                ok += 1
        total_ok += ok
        total += len(clips)
        for s, e in skipped:
            eprint(f"  SKIP {v}/{s}: {e}")

    report = OUTPUT_DIR / "report.csv"
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video", "clip", "flash_frames", "trail_area_pct",
            "components", "largest_px", "mean_brightness",
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone! {total_ok}/{total} trails extracted")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
