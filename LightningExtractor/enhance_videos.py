#!/usr/bin/env python3
"""
enhance_videos.py - AI enhancement pipeline for lightning clips.

Pipeline per clip:
  1. Denoise + extract frames via FFmpeg nlmeans_vulkan
  2. RIFE 2x frame interpolation (doubles fps for smooth slow-motion)
  3. Soft lightning blend (preserve strike + cloud glow, no hard edges)
  4. NVENC H.265 encode (no audio)
  5. Side-by-side comparison video (original vs enhanced)

After all clips: merge per-video comparisons into one comparison reel.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
TOOLS_DIR = BASE_DIR / "tools"
RIFE_DIR = TOOLS_DIR / "rife"
REALESRGAN_DIR = TOOLS_DIR / "realesrgan"
FFMPEG = BASE_DIR / "ffmpeg.EXE"

OUTPUT_DIR = BASE_DIR / "output"
ENHANCED_DIR = BASE_DIR / "enhanced_output"

RIFE_EXE = RIFE_DIR / "rife-ncnn-vulkan.exe"
REALESRGAN_EXE = REALESRGAN_DIR / "realesrgan-ncnn-vulkan.exe"
RIFE_MODEL = str(RIFE_DIR / "rife-v4.6")
REALESRGAN_MODEL = str(REALESRGAN_DIR / "models")

MAX_RIFE_SIZE = 1920


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def run_cmd(cmd: List[str], desc: str = "", timeout: int = None,
            check: bool = True) -> bool:
    if desc:
        eprint(f"  {desc}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            err = r.stderr.strip()[:300]
            eprint(f"  FAILED (code {r.returncode}): {err}")
            if check:
                return False
        return True
    except subprocess.TimeoutExpired:
        eprint(f"  TIMEOUT after {timeout}s")
        return False
    except FileNotFoundError as e:
        eprint(f"  COMMAND NOT FOUND: {e}")
        return False


def get_clip_fps(clip_path: Path) -> float:
    r = subprocess.run(
        [str(FFMPEG), "-i", str(clip_path)],
        capture_output=True, text=True
    )
    for line in r.stderr.split("\n"):
        if "fps" in line and "Stream" in line:
            try:
                return float(line.split("fps")[0].split()[-1])
            except (ValueError, IndexError):
                pass
    return 30.0


def denoise_and_extract(clip_path: Path, frames_dir: Path) -> bool:
    frames_dir.mkdir(parents=True, exist_ok=True)
    return run_cmd([
        str(FFMPEG), "-y",
        "-init_hw_device", "vulkan=vk:0",
        "-filter_hw_device", "vk",
        "-i", str(clip_path),
        "-vf", "hwupload=derive_device=vulkan,nlmeans_vulkan=s=1.0:p=7:r=3,hwdownload,format=yuv420p",
        "-q:v", "1", str(frames_dir / "frame_%08d.png")
    ], "Denoise + extract frames", timeout=180)


def run_rife(input_dir: Path, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_cmd([
        str(RIFE_EXE), "-m", RIFE_MODEL,
        "-i", str(input_dir), "-o", str(output_dir),
        "-g", "0", "-j", "2:2:2"
    ], "RIFE 2x interpolation", timeout=300)


def protect_lightning(orig_dir: Path, rife_dir: Path) -> None:
    orig = sorted(orig_dir.glob("*.png"))
    rife = sorted(rife_dir.glob("*.png"))
    if len(rife) < 2:
        return
    for i in range(len(orig) - 1):
        a = cv2.imread(str(orig[i]))
        b = cv2.imread(str(orig[i + 1]))
        interp_idx = 2 * i + 1
        if interp_idx >= len(rife):
            break
        interp = cv2.imread(str(rife[interp_idx]))
        if interp is None:
            continue
        diff = cv2.absdiff(a, b)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        alpha = np.clip(
            (gray_diff.astype(np.float32) - 25.0) / 80.0, 0.0, 1.0
        )[..., None]
        if alpha.max() <= 0.0:
            continue
        bright_a = a.mean(axis=2)
        bright_b = b.mean(axis=2)
        src = np.where(bright_a[..., None] > bright_b[..., None], a, b)
        result = interp.astype(np.float32) * (1.0 - alpha) + src.astype(np.float32) * alpha
        cv2.imwrite(str(rife[interp_idx]), np.clip(result, 0, 255).astype(np.uint8))


def run_realesrgan(input_dir: Path, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_cmd([
        str(REALESRGAN_EXE),
        "-i", str(input_dir), "-o", str(output_dir),
        "-m", REALESRGAN_MODEL,
        "-n", "realesrgan-x4plus",
        "-s", "2", "-g", "0", "-j", "2:2:2"
    ], "Real-ESRGAN 4x upscale", timeout=600)


def encode_final(frames_dir: Path, output_path: Path, fps: int = 60) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return run_cmd([
        str(FFMPEG), "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "%08d.png"),
        "-c:v", "hevc_nvenc",
        "-cq", "18", "-preset", "p7",
        "-vf", "format=yuv420p",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ], "Encode final (NVENC H.265)", timeout=180)


def create_comparison(orig_path: Path, enhanced_path: Path,
                      comp_path: Path) -> bool:
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(orig_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    if w >= h:
        scale = "960:540"
    else:
        scale = "540:960"
    filter_complex = (
        f"[0:v]scale={scale},setsar=1,drawtext=text='Original':fontsize=24:"
        f"x=(w-text_w)/2:y=h-30:fontcolor=white:box=1:boxcolor=black@0.5[orig];"
        f"[1:v]scale={scale},setsar=1,drawtext=text='AI Enhanced':fontsize=24:"
        f"x=(w-text_w)/2:y=h-30:fontcolor=white:box=1:boxcolor=black@0.5[enh];"
        f"[orig][enh]hstack=inputs=2,format=yuv420p[out]"
    )
    return run_cmd([
        str(FFMPEG), "-y",
        "-i", str(orig_path), "-i", str(enhanced_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "h264_nvenc", "-cq", "23", "-preset", "p7",
        str(comp_path)
    ], "Create comparison video", timeout=120)


def process_clip(clip_path: Path, enhanced_dir: Path,
                 temp_base: Path) -> Tuple[bool, Optional[Path]]:
    video_name = clip_path.parent.parent.name
    clip_name = clip_path.stem

    enhanced_clip = enhanced_dir / video_name / "clips" / clip_path.name
    comp_clip = enhanced_dir / video_name / "comparison" / clip_path.name

    if enhanced_clip.exists() and comp_clip.exists():
        eprint(f"  Already exists, skipping")
        return True, comp_clip

    temp_dir = temp_base / video_name / clip_name
    frames_dir = temp_dir / "frames"
    rife_dir = temp_dir / "rife"

    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        if not denoise_and_extract(clip_path, frames_dir):
            return False, None

        if not run_rife(frames_dir, rife_dir):
            return False, None

        protect_lightning(frames_dir, rife_dir)

        fps = get_clip_fps(clip_path)
        target_fps = round(fps * 2)
        if not encode_final(rife_dir, enhanced_clip, target_fps):
            return False, None

        create_comparison(clip_path, enhanced_clip, comp_clip)
        return True, comp_clip if comp_clip.exists() else None

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_video(video_name: str, temp_dir: Path,
                  max_workers: int = 1) -> None:
    clips_dir = OUTPUT_DIR / video_name / "clips"
    if not clips_dir.exists():
        return

    clip_paths = sorted(clips_dir.glob("*.mp4"))
    if not clip_paths:
        return

    n = len(clip_paths)
    print(f"\n{'='*60}")
    print(f"  {video_name} — {n} clip{'s' if n != 1 else ''}")
    print(f"{'='*60}")

    def _work(clip_path):
        return clip_path, process_clip(
            clip_path, ENHANCED_DIR, temp_dir
        )

    results = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_work, c): c for c in clip_paths}
            for f in tqdm(as_completed(futures), total=n, desc=video_name):
                cp, (ok, comp) = f.result()
                results.append((cp, ok, comp))
    else:
        for cp in tqdm(clip_paths, desc=video_name):
            ok, comp = process_clip(cp, ENHANCED_DIR, temp_dir)
            results.append((cp, ok, comp))

    ok_count = sum(1 for _, ok, _ in results if ok)
    print(f"  Done: {ok_count}/{n} clips enhanced")

    comp_dir = ENHANCED_DIR / video_name / "comparison"
    if comp_dir.exists():
        comps = sorted(comp_dir.glob("*.mp4"))
        if len(comps) > 1:
            concat_file = temp_dir / f"{video_name}_concat.txt"
            with open(concat_file, "w") as f:
                for cp in comps:
                    f.write(f"file '{cp.absolute()}'\n")
            merged = ENHANCED_DIR / video_name / f"{video_name}_comparison.mp4"
            run_cmd([
                str(FFMPEG), "-y", "-f", "concat",
                "-safe", "0", "-i", str(concat_file),
                "-c", "copy", str(merged)
            ], f"Merge {len(comps)} comparisons into one video", timeout=120)
            concat_file.unlink(missing_ok=True)

    src_report = OUTPUT_DIR / video_name / "report.json"
    if src_report.exists():
        dst_report = ENHANCED_DIR / video_name / "report.json"
        dst_report.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_report, dst_report)

    src_reel = OUTPUT_DIR / video_name / f"{video_name}_reel.mp4"
    if src_reel.exists():
        dst_reel = ENHANCED_DIR / video_name / f"{video_name}_reel.mp4"
        dst_reel.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_reel, dst_reel)

    src_contact = OUTPUT_DIR / video_name / f"{video_name}_contact_sheet.png"
    if src_contact.exists():
        dst_contact = ENHANCED_DIR / video_name / f"{video_name}_contact_sheet.png"
        dst_contact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_contact, dst_contact)


def main():
    parser = argparse.ArgumentParser(
        description="AI-enhance lightning clips (denoise → RIFE → upscale → color)"
    )
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Video names to process (default: all)")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Parallel clip processing (default: 1)")
    parser.add_argument("--temp-dir", type=str, default=None,
                        help="Temp directory (default: enhanced_output/temp)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip clips that already have enhanced output")
    args = parser.parse_args()

    if not OUTPUT_DIR.exists():
        eprint(f"ERROR: No output directory found at {OUTPUT_DIR}")
        eprint("Run main.py --confidence 1 --use-gpu first to detect lightning.")
        sys.exit(1)

    temp_base = Path(args.temp_dir) if args.temp_dir else ENHANCED_DIR / "temp"
    temp_base.mkdir(parents=True, exist_ok=True)

    videos = args.videos or sorted(
        d.name for d in OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / "clips").exists()
    )

    if not videos:
        eprint("No video folders found with clips.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to enhance")
    print(f"Output: {ENHANCED_DIR}")
    print()

    for v in videos:
        process_video(v, temp_base, args.max_workers)

    total_enhanced = 0
    total_original = 0
    for v in videos:
        clips = sorted((ENHANCED_DIR / v / "clips").glob("*.mp4"))
        total_enhanced += len(clips)
        total_original += len(list((OUTPUT_DIR / v / "clips").glob("*.mp4")))

    print(f"\n{'='*60}")
    print(f"All done! {total_enhanced}/{total_original} clips enhanced")
    print(f"Results in: {ENHANCED_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
