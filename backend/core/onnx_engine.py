"""
ONNX Inference Engine - YOLOv8 hardened
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def resize(image, input_shape):
    [height, width, _] = image.shape
    length = max((height, width))
    resized = np.zeros((length, length, 3), np.uint8)
    resized[0:height, 0:width] = image
    resized = cv2.resize(resized, (input_shape[1], input_shape[0]))
    return resized


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] + (x[..., 2] - x[..., 0]) / 2
    y[..., 1] = x[..., 1] + (x[..., 3] - x[..., 1]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def preprocess_image(img_path: str, target_h: int, target_w: int):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_r = resize(img_rgb, (target_h, target_w))
    img_norm = img_r.astype(np.float32) / 255.0
    img_nchw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]
    return img_nchw, (orig_h, orig_w)


# ── NMS ────────────────────────────────────────────────────────────────────────

def nms(boxes, scores, iou_threshold=0.45):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


# ── Postprocessor ──────────────────────────────────────────────────────────────

def postprocess_yolo(
    outputs: List[np.ndarray],
    orig_size: tuple,
    target_size: tuple,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    classes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Handles YOLOv8 ONNX output: [1, 4+nc, 8400]
    Also handles YOLOv5:         [1, 8400, 5+nc]
    """
    raw = outputs[0]
    logger.debug(f"Raw output shape: {raw.shape}  dtype={raw.dtype}")

    # ── Step 1: remove batch dim ───────────────────────────────────────────────
    if raw.ndim == 3:
        raw = raw[0]   # [4+nc, 8400] for v8  OR  [8400, 5+nc] for v5

    if raw.ndim != 2:
        logger.error(f"Unexpected tensor ndim={raw.ndim}")
        return []

    # ── Step 2: orient to [num_anchors, cols] ─────────────────────────────────
    # YOLOv8: [4+nc, 8400] → rows=4+nc, cols=8400 → rows < cols → transpose
    # YOLOv5: [8400, 5+nc] → rows=8400, cols=5+nc → rows > cols → keep
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T   # now [8400, 4+nc]
        logger.debug(f"Transposed to {raw.shape}")
    else:
        logger.debug(f"No transpose needed, shape={raw.shape}")

    num_anchors, cols = raw.shape
    num_classes = len(classes) if classes else 0

    logger.debug(
        f"num_anchors={num_anchors}  cols={cols}  "
        f"num_classes={num_classes}  "
        f"box_range min={raw[:,:4].min():.2f} max={raw[:,:4].max():.2f}  "
        f"score_range min={raw[:,4:].min():.4f} max={raw[:,4:].max():.4f}"
    )

    # ── Step 3: detect v5 vs v8 by column count ───────────────────────────────
    if num_classes > 0 and cols == 4 + num_classes:
        fmt = "v8"
    elif num_classes > 0 and cols == 5 + num_classes:
        fmt = "v5"
    elif cols > 4:
        col4_raw_max = float(raw[:, 4].max())
        fmt = "v5" if col4_raw_max <= 1.0 else "v8"
        logger.warning(
            f"num_classes mismatch — using heuristic. "
            f"col4_raw_max={col4_raw_max:.4f} → {fmt}"
        )
    else:
        logger.error(f"Cannot determine format: cols={cols}")
        return []

    logger.debug(f"Format: {fmt}")

    # ── Step 4: extract scores ─────────────────────────────────────────────────
    if fmt == "v8":
        class_scores = raw[:, 4:]
        class_ids    = np.argmax(class_scores, axis=1)
        scores       = class_scores[np.arange(num_anchors), class_ids]

        # YOLOv8 ONNX may output raw logits — apply sigmoid if max > 1
        if scores.max() > 1.0:
            logger.debug("Applying sigmoid to v8 class scores (raw logits detected)")
            scores = 1.0 / (1.0 + np.exp(-scores))

    else:  # v5
        obj_conf     = raw[:, 4]
        class_scores = raw[:, 5:]
        class_ids    = np.argmax(class_scores, axis=1)
        class_confs  = class_scores[np.arange(num_anchors), class_ids]
        scores       = obj_conf * class_confs

    logger.debug(
        f"Scores — min={scores.min():.4f}  max={scores.max():.4f}  "
        f"mean={scores.mean():.4f}  above_{conf_threshold}={(scores > conf_threshold).sum()}"
    )

    # ── Step 5: threshold ──────────────────────────────────────────────────────
    mask      = scores > conf_threshold
    filtered  = raw[mask]
    scores    = scores[mask]
    class_ids = class_ids[mask]

    if len(scores) == 0:
        logger.info(f"No detections above conf={conf_threshold}  (max was {raw[:,4:].max():.4f})")
        return []

    # ── Step 6: decode boxes cx,cy,w,h → x1,y1,x2,y2 ─────────────────────────
    cx, cy, bw, bh = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # ── Step 7: scale boxes back to original image space ──────────────────────
    orig_h, orig_w = orig_size
    target_h, target_w = target_size
    scale_x = max(orig_h, orig_w) / target_w
    scale_y = max(orig_h, orig_w) / target_h
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # ── Step 8: NMS ───────────────────────────────────────────────────────────
    keep = nms(boxes, scores, iou_threshold)
    logger.info(f"Detections after NMS: {len(keep)}  (pre-NMS: {len(scores)})")

    detections = []
    for i in keep:
        cls_id   = int(class_ids[i])
        cls_name = (classes[cls_id] if classes and cls_id < len(classes)
                    else str(cls_id))
        detections.append({
            "class":           cls_name,
            "class_id":        cls_id,
            "confidence":      float(scores[i]),
            "bbox":            [float(boxes[i][0]), float(boxes[i][1]),
                                float(boxes[i][2]), float(boxes[i][3])],
            "adala_status":    "pending",
            "adala_reasoning": None,
        })

    return detections


# ── Detector class ─────────────────────────────────────────────────────────────

class ONNXDetector:
    POSTPROCESSORS: Dict[str, Callable] = {"yolo": postprocess_yolo}

    def __init__(self, model_path: str, classes: Optional[List[str]] = None):
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if p in ort.get_available_providers()]
        self.session      = ort.InferenceSession(model_path, providers=providers)
        self.classes      = classes
        self._input_meta  = self.session.get_inputs()
        self._output_meta = self.session.get_outputs()

        raw_shape = self._input_meta[0].shape
        def _r(v): return v if isinstance(v, int) and v > 0 else 640
        self._target_h = _r(raw_shape[2]) if len(raw_shape) > 2 else 640
        self._target_w = _r(raw_shape[3]) if len(raw_shape) > 3 else 640

        logger.info(
            f"Model loaded: {model_path}\n"
            f"  input : {raw_shape}  target={self._target_h}x{self._target_w}\n"
            f"  output: {[list(o.shape) for o in self._output_meta]}\n"
            f"  classes ({len(classes) if classes else 0}): {classes}\n"
            f"  providers: {providers}"
        )

    @property
    def input_name(self) -> str:
        return self._input_meta[0].name

    @property
    def input_shape(self) -> list:
        return [s if isinstance(s, int) else 640 for s in self._input_meta[0].shape]

    @property
    def output_shapes(self) -> list:
        return [list(o.shape) for o in self._output_meta]

    def run(self, img_path: str, conf_threshold=0.25,
            iou_threshold=0.45, postprocessor="yolo"):
        img_nchw, orig_size = preprocess_image(
            img_path, self._target_h, self._target_w)
        outputs = self.session.run(None, {self.input_name: img_nchw})
        fn = self.POSTPROCESSORS.get(postprocessor, postprocess_yolo)
        return fn(outputs, orig_size, (self._target_h, self._target_w),
                  conf_threshold=conf_threshold,
                  iou_threshold=iou_threshold,
                  classes=self.classes)

    def inspect(self) -> Dict[str, Any]:
        return {
            "input_name":    self.input_name,
            "input_shape":   self.input_shape,
            "output_shapes": self.output_shapes,
            "target_hw":     (self._target_h, self._target_w),
            "providers":     self.session.get_providers(),
            "classes":       self.classes,
            "num_classes":   len(self.classes) if self.classes else None,
        }

    @classmethod
    def register_postprocessor(cls, name: str, fn: Callable):
        cls.POSTPROCESSORS[name] = fn


def inspect_onnx_model(model_path: str) -> Dict[str, Any]:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return {
        "inputs":  [{"name": i.name, "shape": list(i.shape), "type": i.type}
                    for i in session.get_inputs()],
        "outputs": [{"name": o.name, "shape": list(o.shape), "type": o.type}
                    for o in session.get_outputs()],
    }