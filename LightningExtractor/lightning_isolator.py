from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm


def isolate_lightning(frame_path: Path, output_path: Path, index: int) -> Path:
    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read frame: {frame_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    min_area = 50
    mask = np.zeros_like(gray)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    kernel_dilate = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask

    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"lightning_{index:03d}.png"
    cv2.imwrite(str(out_file), rgba)

    return out_file


def isolate_all(frames_info: List[dict], output_dir: Path) -> List[Path]:
    png_dir = output_dir / "lightning_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    png_paths = []
    for i, info in enumerate(tqdm(frames_info, desc="Isolating lightning")):
        frame_path = Path(info["sharpest_frame"])
        out_path = isolate_lightning(frame_path, png_dir, i + 1)
        png_paths.append(out_path)

    return png_paths
