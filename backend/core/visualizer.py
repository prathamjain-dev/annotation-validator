"""
Annotation Visualization Utilities
Draw bounding boxes, labels, and Adala reasoning on images.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional
import os

# Status color map (BGR for OpenCV)
STATUS_COLORS = {
    "approved": (0, 200, 0),      # green
    "rejected": (0, 0, 220),      # red
    "relabeled": (220, 130, 0),   # orange
    "flagged": (0, 165, 255),     # yellow-orange
    "pending": (180, 180, 180),   # gray
}


def draw_annotations(
    image_path: str,
    detections: List[Dict[str, Any]],
    show_reasoning: bool = False,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on an image.
    Returns the annotated image as numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    font_scale = max(0.4, min(w, h) / 1000)
    thickness = max(1, int(min(w, h) / 400))

    for det in detections:
        if det.get("adala_status") == "rejected":
            continue  # Skip rejected detections

        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        status = det.get("adala_status", "pending")
        color = STATUS_COLORS.get(status, STATUS_COLORS["pending"])

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Build label
        cls_name = det.get("class", "?")
        conf = det.get("confidence", 0)
        label = f"{cls_name} {conf:.2f}"
        if status != "pending":
            label += f" [{status}]"

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # Optional: show reasoning snippet
        if show_reasoning and det.get("adala_reasoning"):
            reason = det["adala_reasoning"][:60] + "..." if len(det.get("adala_reasoning", "")) > 60 else det.get("adala_reasoning", "")
            cv2.putText(img, reason, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, 1)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)

    return img


def image_to_bytes(img: np.ndarray, ext: str = ".jpg") -> bytes:
    """Convert numpy image to bytes."""
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError("Image encoding failed")
    return buf.tobytes()
