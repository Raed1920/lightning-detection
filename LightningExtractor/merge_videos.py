#!/usr/bin/env python3
"""
merge_videos.py - Merge all enhanced (and all original) lightning clips into
two single videos, separated by capture date.

For each merged video:
  - All clips normalized to a 1920x1080 landscape canvas (portrait clips get
    side bars), uniform fps, uniform H.264 encode.
  - A 3s title card per capture date separates the date groups.
  - Enhanced clips get the per-strike timestamp burned in (same style as the
    original clips); original clips already carry their timestamp.

Outputs in merged_output/:
  - enhanced_merged_by_date.mp4  (60 fps)
  - original_merged_by_date.mp4  (30 fps)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
FFMPEG = BASE_DIR / "ffmpeg.EXE"
OUTPUT_DIR = BASE_DIR / "output"
ENHANCED_DIR = BASE_DIR / "enhanced_output"
MERGED_DIR = BASE_DIR / "merged_output"

W, H = 1920, 1080
FONT = "C\\:/Windows/Fonts/arial.ttf"
TITLE_CARD_SEC = 3.0


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def video_date(video_name: str) -> str:
    stamp = video_name.split("_")[1]
    return f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"


def run(cmd, desc="") -> bool:
    if desc:
        eprint(f"  {desc}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        eprint(f"  FAILED (code {r.returncode}): {r.stderr.strip()[:300]}")
        return False
    return True


def load_timestamps(video_name: str):
    report = OUTPUT_DIR / video_name / "report.json"
    if not report.exists():
        return {}
    data = json.loads(report.read_text(encoding="utf-8"))
    by_index = {}
    for d in data.get("detections", []):
        by_index[d["index"]] = d["timestamp"]
    return by_index


def make_normalized_clip(src: Path, dst: Path, fps: int, text: str = None) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
    )
    if text:
        safe = text.replace(":", "\\:")
        vf += (
            f",drawtext=text='{safe}':fontsize=48:fontcolor=white:"
            f"box=1:boxcolor=black@0.5:"
            f"x=(w-text_w)/2:y=h-text_h-40:fontfile='{FONT}'"
        )
    return run([
        str(FFMPEG), "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "h264_nvenc", "-cq", "20", "-preset", "p7",
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst),
    ], f"Normalize {src.stem} ({fps}fps)")


def make_title_card(text: str, dst: Path, fps: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"drawtext=text='{text}':fontsize=110:fontcolor=white:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-40:fontfile='{FONT}',"
        f"drawtext=text='Lightning strikes':fontsize=48:fontcolor=white@0.8:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+80:fontfile='{FONT}'"
    )
    return run([
        str(FFMPEG), "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:d={TITLE_CARD_SEC}:r={fps}",
        "-vf", vf,
        "-c:v", "h264_nvenc", "-cq", "20", "-preset", "p7",
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst),
    ], f"Title card '{text}'")


def concat(files: list, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    list_file = dst.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{p.absolute()}'\n" for p in files),
        encoding="utf-8",
    )
    ok = run([
        str(FFMPEG), "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(dst),
    ], f"Concatenate {len(files)} segments -> {dst.name}")
    list_file.unlink(missing_ok=True)
    return ok


def build_merged(kind: str, fps: int, add_timestamp: bool) -> Path:
    norm_dir = MERGED_DIR / f"{kind}_norm"
    if norm_dir.exists():
        shutil.rmtree(norm_dir)

    videos = sorted(
        d.name for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / "clips").exists()
    )

    groups = {}
    for v in videos:
        groups.setdefault(video_date(v), []).append(v)

    segments = []
    for date in sorted(groups):
        card = norm_dir / f"_card_{date.replace('-', '')}.mp4"
        if not make_title_card(date, card, fps):
            return None
        segments.append(card)
        for v in sorted(groups[date]):
            src_dir = (ENHANCED_DIR if kind == "enhanced" else OUTPUT_DIR) / v / "clips"
            clips = sorted(src_dir.glob("*.mp4"))
            ts_map = load_timestamps(v) if add_timestamp else {}
            for clip in clips:
                idx = int(clip.stem.split("_")[1])
                ts = ts_map.get(idx)
                dst = norm_dir / f"{v}_{clip.stem}.mp4"
                if not make_normalized_clip(clip, dst, fps, text=ts if add_timestamp else None):
                    return None
                segments.append(dst)

    out = MERGED_DIR / f"{kind}_merged_by_date.mp4"
    if not concat(segments, out):
        return None
    shutil.rmtree(norm_dir, ignore_errors=True)
    return out


def main():
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    print("Building enhanced merged video (60 fps)...")
    out = build_merged("enhanced", 60, add_timestamp=True)
    if out is None:
        sys.exit(1)

    print("Building original merged video (30 fps)...")
    out2 = build_merged("original", 30, add_timestamp=False)
    if out2 is None:
        sys.exit(1)

    print(f"\nDone!\n  {out}\n  {out2}")


if __name__ == "__main__":
    main()
