from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def compute_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_sharpest_frame(clip_path: Path, output_path: Path, index: int) -> Tuple[float, Path]:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open clip: {clip_path}")

    best_frame = None
    best_sharpness = -1.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total), desc=f"Frame {index:03d}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        sharpness = compute_sharpness(frame)
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame.copy()

    cap.release()

    if best_frame is None:
        raise RuntimeError(f"No frames found in clip: {clip_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"strike_{index:03d}.png"
    cv2.imwrite(str(out_file), best_frame)

    return best_sharpness, out_file


def extract_all_sharpest(clip_paths: List[Path], output_dir: Path) -> List[dict]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, clip_path in enumerate(clip_paths):
        sharpness, out_file = extract_sharpest_frame(clip_path, frames_dir, i + 1)
        results.append({
            "clip": str(clip_path),
            "sharpest_frame": str(out_file),
            "sharpness": sharpness,
        })

    return results
