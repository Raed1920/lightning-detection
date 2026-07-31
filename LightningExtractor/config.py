from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    video_extensions: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    frame_step: int = 2
    pre_scan_frames: int = 150

    brightness_weight: float = 0.45
    bright_pixel_weight: float = 0.35
    edge_weight: float = 0.20

    confidence_threshold: float = 0.5
    brightness_threshold_factor: float = 3.0
    edge_threshold_factor: float = 3.0
    bright_pixel_ratio_threshold: float = 0.01

    merge_interval_sec: float = 0.5

    clip_before_sec: float = 1.0
    clip_after_sec: float = 1.0
    slowmo_factor: Optional[float] = None

    use_gpu: bool = False
    parallel: bool = False
    max_workers: int = 4
    compile_reel: bool = False

    generate_png: bool = True
    generate_contact_sheet: bool = True

    roi_mask_path: Optional[Path] = None

    log_level: str = "INFO"
    log_file: Optional[Path] = None

    output_format: str = "mp4"

    def __post_init__(self):
        if self.frame_step < 1:
            raise ValueError("frame_step must be >= 1")
        if not 0 < self.confidence_threshold <= 100:
            raise ValueError("confidence_threshold must be in (0, 100]")
