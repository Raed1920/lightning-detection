#!/usr/bin/env python3
"""
make_contact_sheet.py - Build per-video contact sheets of all extracted
trails so the whole batch can be reviewed at a glance.

Output: bolt_output/<video>_contact.png (grid of labeled trail thumbnails)
"""

import csv
import sys
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "bolt_output"

COLS = 5
THUMB_W, THUMB_H = 320, 180
MARGIN = 12
LABEL_H = 24
BG = (10, 10, 10)
FG = (220, 220, 220)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():
    report_path = OUTPUT_DIR / "report.csv"
    if not report_path.exists():
        eprint("no report.csv - run extract_trail.py first")
        sys.exit(1)

    metrics = {}
    with open(report_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            metrics[(r["video"], r["clip"])] = r

    for sheet in sorted(OUTPUT_DIR.glob("*_contact.png")):
        sheet.unlink()

    videos = sorted({v for v, _ in metrics})
    for video in videos:
        entries = []
        for (v, clip), m in metrics.items():
            if v != video:
                continue
            p = OUTPUT_DIR / video / f"{clip}_trail.png"
            if p.exists():
                entries.append((clip, m, p))
        entries.sort()

        if not entries:
            continue

        rows = (len(entries) + COLS - 1) // COLS
        cell_w = THUMB_W + MARGIN
        cell_h = THUMB_H + LABEL_H + MARGIN
        sheet_w = COLS * cell_w + MARGIN
        sheet_h = rows * cell_h + MARGIN
        canvas = np_full = None
        import numpy as np
        canvas = np.full((sheet_h, sheet_w, 3), BG, np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (clip, m, p) in enumerate(entries):
            r, c = divmod(i, COLS)
            x0 = MARGIN + c * cell_w
            y0 = MARGIN + r * cell_h
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (THUMB_W, THUMB_H))
            canvas[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = img
            label = f'{clip}  {m["largest_px"]}px x{m["components"]}'
            cv2.putText(canvas, label, (x0 + 2, y0 + THUMB_H + 16),
                        font, 0.5, FG, 1, cv2.LINE_AA)

        out = OUTPUT_DIR / f"{video}_contact.png"
        cv2.imwrite(str(out), canvas)
        eprint(f"wrote {out.name} ({len(entries)} trails, {rows}x{COLS})")


if __name__ == "__main__":
    main()
