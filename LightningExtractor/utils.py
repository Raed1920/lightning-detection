import subprocess
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    common_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for p in common_paths:
        if Path(p).exists():
            return p
    return None


def check_ffmpeg_gpu_support(ffmpeg_path: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        return any(x in result.stdout for x in ("h264_nvenc", "h264_amf", "h264_qsv"))
    except Exception:
        return False


def get_gpu_encoder(ffmpeg_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        stdout = result.stdout
        for enc in ("h264_nvenc", "h264_amf", "h264_qsv"):
            if enc in stdout:
                return enc
        return None
    except Exception:
        return None


def has_cuda_opencv() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def has_cuda_pytorch() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def create_roi_mask(frame_shape: tuple, mask_path: Optional[Path] = None) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if mask_path and mask_path.exists():
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            mask_img = cv2.resize(mask_img, (w, h))
            mask = mask_img
    return mask


def format_timestamp(total_seconds: float) -> str:
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
