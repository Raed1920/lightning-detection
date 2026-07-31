import subprocess
from pathlib import Path
from typing import List, Optional

from utils import find_ffmpeg, get_gpu_encoder


def compile_highlight_reel(clip_paths: List[Path],
                           timestamp_clips: Optional[List[Path]] = None,
                           output_path: Path = Path("output/highlight_reel.mp4"),
                           config=None,
                           delete_clips: bool = False) -> Path:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("FFmpeg not found")

    if not clip_paths:
        raise ValueError("No clips to compile")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    codec = "libx264"
    codec_opts = ["-preset", "fast", "-crf", "18"]
    if config and config.use_gpu:
        enc = get_gpu_encoder(ffmpeg)
        if enc == "h264_nvenc":
            codec = enc
            codec_opts = ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-b:v", "10M", "-maxrate", "20M"]
        elif enc in ("h264_amf", "h264_qsv"):
            codec = enc
            codec_opts = ["-quality", "speed"]

    paths = timestamp_clips if timestamp_clips else clip_paths

    inputs = []
    filter_labels = []
    for i, cp in enumerate(paths):
        inputs.extend(["-i", str(cp.resolve())])
        filter_labels.append(f"[{i}:v]")

    filter_str = "".join(filter_labels) + f"concat=n={len(paths)}:v=1[out]"

    cmd = [
        ffmpeg, "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", codec,
        *codec_opts,
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {result.stderr[:500]}")

    if delete_clips:
        paths_to_delete = clip_paths + (timestamp_clips or [])
        for cp in paths_to_delete:
            cp.unlink(missing_ok=True)

    return output_path
