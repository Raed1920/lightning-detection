from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from config import Config
from utils import create_roi_mask


@dataclass
class Detection:
    frame_number: int
    timestamp_sec: float
    confidence: float
    brightness_score: float
    bright_pixel_score: float
    edge_score: float


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    total_frames_scanned: int = 0
    fps: float = 0.0


def pre_scan_baseline(cap: cv2.VideoCapture, config: Config) -> dict:
    frames_sampled = 0
    brightness_vals = []
    edge_vals = []
    bright_ratios = []

    for _ in range(config.pre_scan_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_bright = float(gray.mean())
        brightness_vals.append(mean_bright)

        edges = cv2.Canny(gray, 50, 150)
        edge_vals.append(float(edges.sum() / 255))

        bright_pixels = np.sum(gray > 200)
        bright_ratios.append(bright_pixels / gray.size)

        frames_sampled += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if frames_sampled == 0:
        return {
            "brightness_mean": 0.0,
            "brightness_std": 1.0,
            "edge_mean": 0.0,
            "edge_std": 1.0,
            "bright_ratio_mean": 0.0,
        }

    arr_b = np.array(brightness_vals)
    arr_e = np.array(edge_vals)
    arr_r = np.array(bright_ratios)

    return {
        "brightness_mean": float(arr_b.mean()),
        "brightness_std": max(float(arr_b.std()), 1.0),
        "edge_mean": float(arr_e.mean()),
        "edge_std": max(float(arr_e.std()), 1.0),
        "bright_ratio_mean": float(arr_r.mean()),
    }


def detect_lightning(video_path: str, config: Config) -> DetectionResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if config.pre_scan_frames > 0:
        baseline = pre_scan_baseline(cap, config)
    else:
        baseline = {
            "brightness_mean": 0.0,
            "brightness_std": 1.0,
            "edge_mean": 0.0,
            "edge_std": 1.0,
            "bright_ratio_mean": 0.0,
        }

    roi_mask = create_roi_mask((h, w), config.roi_mask_path)

    prev_gray = None
    prev_edge_count = 0
    raw_scores = []

    step = config.frame_step
    effective_frames = total_frames // step

    pbar = tqdm(total=effective_frames, desc="Detecting lightning", unit="frm")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=roi_mask)

        mean_bright = float(gray_masked[roi_mask > 0].mean())

        brightness_diff = mean_bright - baseline["brightness_mean"]
        brightness_score = max(0.0, brightness_diff / baseline["brightness_std"])

        bright_pixels_200 = np.sum(gray_masked > 200)
        bright_pixels_220 = np.sum(gray_masked > 220)
        bright_pixels_240 = np.sum(gray_masked > 240)
        total_pixels = np.sum(roi_mask > 0)

        bright_ratio = (
            bright_pixels_200 * 0.3 +
            bright_pixels_220 * 0.3 +
            bright_pixels_240 * 0.4
        ) / total_pixels if total_pixels > 0 else 0.0

        bright_pixel_thresh = baseline["bright_ratio_mean"] + 2 * baseline["brightness_std"] / 255
        bright_pixel_score = max(0.0, (bright_ratio - bright_pixel_thresh) * 100)

        edges = cv2.Canny(gray_masked, 50, 150)
        edge_count = int(edges.sum() / 255)

        if prev_edge_count > 0:
            edge_diff = edge_count - prev_edge_count
            edge_score = max(0.0, edge_diff / (baseline["edge_std"] + 1))
        else:
            edge_score = 0.0

        prev_edge_count = edge_count

        confidence = (
            config.brightness_weight * brightness_score +
            config.bright_pixel_weight * bright_pixel_score +
            config.edge_weight * edge_score
        )

        timestamp = frame_idx / fps

        raw_scores.append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "confidence": confidence,
            "brightness": brightness_score,
            "bright_pixel": bright_pixel_score,
            "edge": edge_score,
        })

        pbar.update(1)
        frame_idx += 1

    pbar.close()
    cap.release()

    detections = _merge_detections(raw_scores, config, fps)
    return DetectionResult(
        detections=detections,
        total_frames_scanned=frame_idx,
        fps=fps,
    )


def _merge_detections(raw_scores: List[dict], config: Config, fps: float) -> List[Detection]:
    if not raw_scores:
        return []

    candidates = [s for s in raw_scores if s["confidence"] >= config.confidence_threshold]
    if not candidates:
        return []

    candidates.sort(key=lambda x: x["frame"])
    merge_frames = int(config.merge_interval_sec * fps)

    groups = []
    current_group = [candidates[0]]

    for c in candidates[1:]:
        if c["frame"] - current_group[-1]["frame"] <= merge_frames:
            current_group.append(c)
        else:
            groups.append(current_group)
            current_group = [c]

    groups.append(current_group)

    detections = []
    for group in groups:
        best = max(group, key=lambda x: x["confidence"])
        detections.append(Detection(
            frame_number=best["frame"],
            timestamp_sec=best["timestamp"],
            confidence=best["confidence"],
            brightness_score=best["brightness"],
            bright_pixel_score=best["bright_pixel"],
            edge_score=best["edge"],
        ))

    return detections
