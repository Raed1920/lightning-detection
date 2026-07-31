import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from config import Config
from contact_sheet import create_contact_sheet
from video_scanner import scan_videos, validate_video
from lightning_detector import detect_lightning
from clip_exporter import export_all_clips
from frame_selector import extract_all_sharpest
from lightning_isolator import isolate_all
from highlight_reel import compile_highlight_reel
from logger import setup_logger
from utils import (find_ffmpeg, check_ffmpeg_gpu_support,
                   has_cuda_opencv, has_cuda_pytorch, format_timestamp)


def process_single_video(video_path: Path, config: Config, logger) -> dict:
    logger.info(f"Processing {video_path.name}...")

    video_dir = config.output_dir / video_path.stem
    video_dir.mkdir(parents=True, exist_ok=True)

    detection_result = detect_lightning(str(video_path), config)
    detections = detection_result.detections
    logger.info(f"Frames scanned: {detection_result.total_frames_scanned}")
    logger.info(f"Lightning events detected: {len(detections)}")

    timestamps = [d.timestamp_sec for d in detections]
    timestamps_path = video_dir / "timestamps.txt"
    with open(str(timestamps_path), "w") as f:
        for ts in timestamps:
            f.write(f"{format_timestamp(ts)}\n")

    if not detections:
        logger.warning("No lightning detected, skipping export")
        report = _build_report(video_path, detections, [], [], [])
        _save_report(report, video_dir)
        return report

    logger.info("Exporting clips with timestamps and slow-mo...")
    clip_paths = export_all_clips(video_path, video_dir, timestamps, config)

    logger.info("Selecting sharpest frames...")
    frames_info = extract_all_sharpest(clip_paths, video_dir)

    logger.info("Compiling per-video highlight reel...")
    reel_path = video_dir / f"{video_path.stem}_reel.mp4"
    compile_highlight_reel(clip_paths, output_path=reel_path, config=config, delete_clips=False)

    if config.generate_contact_sheet and frames_info:
        logger.info("Generating contact sheet...")
        frame_paths = [Path(f["sharpest_frame"]) for f in frames_info]
        sheet_path = video_dir / f"{video_path.stem}_contact_sheet.png"
        create_contact_sheet(frame_paths, sheet_path)

    report = _build_report(video_path, detections, clip_paths, frames_info)
    _save_report(report, video_dir)

    logger.info(f"Completed {video_path.name}")
    return report


def _build_report(video_path, detections, clip_paths, frames_info):
    return {
        "video": video_path.name,
        "video_path": str(video_path),
        "total_detections": len(detections),
        "detections": [
            {
                "index": i + 1,
                "frame": d.frame_number,
                "timestamp": format_timestamp(d.timestamp_sec),
                "timestamp_sec": d.timestamp_sec,
                "confidence": round(d.confidence, 4),
                "brightness_score": round(d.brightness_score, 4),
                "bright_pixel_score": round(d.bright_pixel_score, 4),
                "edge_score": round(d.edge_score, 4),
            }
            for i, d in enumerate(detections)
        ],
        "clips": [str(p) for p in clip_paths],
        "frames": [
            {
                "index": i + 1,
                "sharpness": round(f["sharpness"], 2),
                "path": f["sharpest_frame"],
            }
            for i, f in enumerate(frames_info)
        ],
    }


def _save_report(report: dict, output_dir: Path):
    report_path = output_dir / "report.json"
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)


def load_config_from_args() -> tuple:
    parser = argparse.ArgumentParser(
        description="LightningExtractor - Automatic Lightning Detection & Extraction"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Input directory or single video file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Confidence threshold (default=0.5, higher= fewer detections)")
    parser.add_argument("--slowmo", type=float, default=None,
                        help="Slow-motion factor. 0.5 = 2x slow, 0.25 = 4x slow")
    parser.add_argument("--step", type=int, default=None,
                        help="Frame step (process every Nth frame)")
    parser.add_argument("--parallel", action="store_true", default=None,
                        help="Process videos in parallel")
    parser.add_argument("--use-gpu", action="store_true", default=None,
                        help="Use GPU acceleration if available")
    parser.add_argument("--compile", action="store_true", default=None,
                        help="Compile highlight reel after processing")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip transparent PNG generation")
    parser.add_argument("--no-contact", action="store_true",
                        help="Skip contact sheet generation")
    parser.add_argument("--log-level", type=str, default=None,
                        help="Log level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    config = Config()
    single_file = None

    if args.input:
        p = Path(args.input)
        if p.is_file():
            config.input_dir = p.parent
            single_file = p
        else:
            config.input_dir = p
    if args.output:
        config.output_dir = Path(args.output)
    if args.confidence is not None:
        config.confidence_threshold = args.confidence
    if args.slowmo is not None:
        config.slowmo_factor = args.slowmo
    if args.step is not None:
        config.frame_step = args.step
    if args.parallel is not None:
        config.parallel = args.parallel
    if args.use_gpu is not None:
        config.use_gpu = args.use_gpu
    if args.compile is not None:
        config.compile_reel = args.compile
    if args.no_png:
        config.generate_png = False
    if args.no_contact:
        config.generate_contact_sheet = False
    if args.log_level:
        config.log_level = args.log_level

    return config, single_file


def _setup_gpu(config: Config, logger):
    if not config.use_gpu:
        return

    logger.info("Checking GPU availability...")
    cuda_cv = has_cuda_opencv()
    cuda_pt = has_cuda_pytorch()
    ffmpeg_path = find_ffmpeg()
    ffmpeg_gpu = check_ffmpeg_gpu_support(ffmpeg_path) if ffmpeg_path else False

    if cuda_cv or cuda_pt:
        logger.info(f"GPU detected: OpenCV CUDA={cuda_cv}, PyTorch CUDA={cuda_pt}")
    else:
        logger.warning("No CUDA GPU detected (via OpenCV or PyTorch), falling back to CPU")

    if ffmpeg_gpu:
        logger.info("FFmpeg GPU encoding available (NVENC/AMF/QSV)")
    else:
        logger.info("FFmpeg GPU encoding not available, using software encoding")


def main():
    config, single_file = load_config_from_args()
    logger = setup_logger(level=config.log_level)

    logger.info("LightningExtractor v1.0")

    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        logger.error("FFmpeg not found. Install FFmpeg (https://ffmpeg.org) and add to PATH.")
        sys.exit(1)
    logger.info(f"FFmpeg found: {ffmpeg_path}")

    _setup_gpu(config, logger)

    if single_file:
        videos = [single_file]
    else:
        videos = scan_videos(config.input_dir, config.video_extensions)

    if not videos:
        logger.error(f"No video files found in {config.input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(videos)} video(s) to process")

    reports = []
    if config.parallel and len(videos) > 1:
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(process_single_video, v, config, logger): v
                for v in videos
            }
            for future in as_completed(futures):
                v = futures[future]
                try:
                    report = future.result()
                    reports.append(report)
                except Exception as e:
                    logger.error(f"Failed to process {v.name}: {e}")
    else:
        for video in videos:
            try:
                is_valid, msg, info = validate_video(video)
                if not is_valid:
                    logger.warning(f"Skipping {video.name}: {msg}")
                    continue
                logger.info(f"Video info: {info.get('width')}x{info.get('height')}"
                            f" @ {info.get('fps'):.2f}fps, {info.get('frame_count')} frames")
                report = process_single_video(video, config, logger)
                reports.append(report)
            except Exception as e:
                logger.error(f"Error processing {video.name}: {e}")

    if config.compile_reel and reports:
        logger.info("Compiling highlight reel...")
        all_clips = []
        for r in reports:
            all_clips.extend([Path(c) for c in r.get("clips", [])])
        if all_clips:
            reel_path = config.output_dir / "highlight_reel.mp4"
            try:
                compile_highlight_reel(all_clips, output_path=reel_path, config=config)
                logger.info(f"Highlight reel saved: {reel_path}")
            except Exception as e:
                logger.error(f"Failed to compile highlight reel: {e}")

    total_events = sum(r.get("total_detections", 0) for r in reports)
    logger.info(f"All done! Processed {len(videos)} video(s), {total_events} lightning events total")


if __name__ == "__main__":
    main()
