from pathlib import Path
from typing import List

import cv2
import numpy as np


def create_contact_sheet(frame_paths: List[Path], output_path: Path,
                         cols: int = 4, thumb_width: int = 320) -> Path:
    if not frame_paths:
        raise ValueError("No frames to create contact sheet")

    thumbs = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_width / w
        thumb_h = int(h * scale)
        thumb = cv2.resize(img, (thumb_width, thumb_h))
        thumbs.append(thumb)

    if not thumbs:
        raise RuntimeError("No valid frames for contact sheet")

    rows = (len(thumbs) + cols - 1) // cols
    max_h = max(t.shape[0] for t in thumbs)
    gap = 5

    sheet_w = cols * (thumb_width + gap) + gap
    sheet_h = rows * (max_h + gap) + gap
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 32

    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        x = gap + c * (thumb_width + gap)
        y = gap + r * (max_h + gap)
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return output_path
