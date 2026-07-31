from pathlib import Path
from typing import List, Tuple

import cv2


def scan_videos(input_dir: Path, extensions: tuple) -> List[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    videos = []
    for ext in extensions:
        videos.extend(input_dir.rglob(f"*{ext}"))
    return sorted(videos)


def validate_video(video_path: Path) -> Tuple[bool, str, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False, "Could not open video file", {}

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    if info["frame_count"] <= 0:
        cap.release()
        return False, "Video has no frames", {}

    if info["width"] <= 0 or info["height"] <= 0:
        cap.release()
        return False, "Invalid video dimensions", {}

    cap.release()
    return True, "OK", info
