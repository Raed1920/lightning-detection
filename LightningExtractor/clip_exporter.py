import subprocess
from pathlib import Path
from typing import Optional

from config import Config
from utils import find_ffmpeg, get_gpu_encoder, format_timestamp


def _video_codec(config: Config) -> tuple:
    if not config.use_gpu:
        return ("libx264", [])
    enc = get_gpu_encoder(find_ffmpeg())
    if enc == "h264_nvenc":
        return (enc, ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-b:v", "10M", "-maxrate", "20M"])
    elif enc == "h264_amf":
        return (enc, ["-quality", "speed", "-rc", "cbr", "-b:v", "10M"])
    elif enc == "h264_qsv":
        return (enc, ["-preset", "medium", "-global_quality", "18"])
    return ("libx264", ["-preset", "fast", "-crf", "18"])


def export_clip(video_path: Path,
                output_path: Path,
                timestamp_sec: float,
                config: Config,
                index: int,
                slowmo_factor: Optional[float] = None) -> Path:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("FFmpeg not found. Install FFmpeg and add it to PATH.")

    output_path.mkdir(parents=True, exist_ok=True)

    start_time = max(0.0, timestamp_sec - config.clip_before_sec)
    duration = config.clip_before_sec + config.clip_after_sec

    ext = config.output_format
    clip_file = output_path / f"strike_{index:03d}.{ext}"

    codec, codec_opts = _video_codec(config)

    cmd = [
        ffmpeg, "-y",
        "-ss", str(start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", codec,
        *codec_opts,
        "-pix_fmt", "yuv420p",
    ]

    factor = slowmo_factor or config.slowmo_factor
    if factor is not None and factor > 0 and factor != 1.0:
        filter_str = f"setpts={1.0/factor:.2f}*PTS"
        cmd.extend(["-vf", filter_str])

    cmd.append(str(clip_file))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for clip {index}: {result.stderr[:500]}")

    return clip_file


def export_clip_with_timestamp(video_path: Path,
                               output_path: Path,
                               timestamp_sec: float,
                               config: Config,
                               index: int,
                               slowmo_factor: Optional[float] = None) -> Path:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("FFmpeg not found. Install FFmpeg and add it to PATH.")

    output_path.mkdir(parents=True, exist_ok=True)

    start_time = max(0.0, timestamp_sec - config.clip_before_sec)
    duration = config.clip_before_sec + config.clip_after_sec
    ts_str = format_timestamp(timestamp_sec).replace(":", "\\:")

    ext = config.output_format
    clip_file = output_path / f"strike_{index:03d}.{ext}"

    codec, codec_opts = _video_codec(config)

    factor = slowmo_factor or config.slowmo_factor
    filter_parts = []
    if factor is not None and factor > 0 and factor != 1.0:
        filter_parts.append(f"setpts={1.0/factor:.2f}*PTS")
    filter_parts.append(
        f"drawtext=text='{ts_str}':"
        f"fontsize=48:fontcolor=white:"
        f"box=1:boxcolor=black@0.5:"
        f"x=(w-text_w)/2:y=h-text_h-40:"
        f"fontfile='C\\:/Windows/Fonts/arial.ttf'"
    )
    filter_str = ",".join(filter_parts)

    cmd = [
        ffmpeg, "-y",
        "-ss", str(start_time),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", codec,
        *codec_opts,
        "-pix_fmt", "yuv420p",
        "-vf", filter_str,
        str(clip_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg timestamp clip {index} failed: {result.stderr[:500]}")

    return clip_file


def export_all_clips(video_path: Path,
                     output_dir: Path,
                     timestamps: list,
                     config: Config) -> list:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for i, ts in enumerate(timestamps):
        clip_path = export_clip_with_timestamp(video_path, clips_dir, ts, config, i + 1, slowmo_factor=config.slowmo_factor)
        clip_paths.append(clip_path)

    return clip_paths
