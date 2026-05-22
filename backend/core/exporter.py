"""
YOLO Export Module
Converts internal annotations to YOLO format and packages as zip.
"""

import os
import shutil
import zipfile
import yaml
from typing import List, Dict, Any


def bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> List[float]:
    """Convert [x1, y1, x2, y2] to YOLO [cx, cy, w, h] normalized."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [cx, cy, w, h]


def export_yolo(
    annotations: List[Dict[str, Any]],
    classes: List[str],
    export_dir: str,
    dataset_name: str,
) -> str:
    """
    Export annotations to YOLO format.

    Returns path to zip file.

    Structure:
        dataset/
        ├── images/
        ├── labels/
        └── dataset.yaml
    """
    import cv2

    dataset_dir = os.path.join(export_dir, dataset_name)
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    class_to_id = {c: i for i, c in enumerate(classes)}

    for ann in annotations:
        image_path = ann.get("image_path", "")
        image_name = ann.get("image_name", "")
        detections = ann.get("detections", [])

        # Copy image
        dst_img = os.path.join(images_dir, image_name)
        if os.path.exists(image_path):
            shutil.copy2(image_path, dst_img)

        # Read image dimensions
        img = cv2.imread(image_path) if os.path.exists(image_path) else None
        img_h, img_w = (img.shape[:2] if img is not None else (640, 640))

        # Write label file
        stem = os.path.splitext(image_name)[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        lines = []
        for det in detections:
            if det.get("adala_status") == "rejected":
                continue  # Skip rejected
            cls_name = det.get("class", "")
            cls_id = class_to_id.get(cls_name, 0)
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            yolo_bbox = bbox_to_yolo(bbox, img_w, img_h)
            lines.append(f"{cls_id} {' '.join(f'{v:.6f}' for v in yolo_bbox)}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    # Write dataset.yaml
    yaml_content = {
        "path": dataset_dir,
        "train": "images",
        "val": "images",
        "nc": len(classes),
        "names": {i: c for i, c in enumerate(classes)},
    }
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    # Zip everything
    zip_path = os.path.join(export_dir, f"{dataset_name}_yolo.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                fp = os.path.join(root, file)
                arcname = os.path.relpath(fp, export_dir)
                zf.write(fp, arcname)

    return zip_path
