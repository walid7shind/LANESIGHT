import numpy as np
from typing import List, Dict
from ultralytics import YOLO


class YOLOInferencer:
    """
    YOLO wrapper for object detection.

    Intended usage:
      - mask out vehicles
      - downweight lane confidence near cars
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str | None = None,
        classes: List[int] | None = None,
        conf_thres: float = 0.3,
    ):
        """
        Args:
            weights: YOLO model path or name
            device: 'cuda', 'cpu', or None
            classes: class IDs to keep (e.g. cars)
            conf_thres: confidence threshold
        """
        self.model = YOLO(weights)
        self.device = device
        self.classes = classes
        self.conf = conf_thres

    def infer_frame(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Args:
            frame_bgr: (H,W,3) BGR image

        Returns:
            detections: list of dicts:
              {
                'xyxy': (x1,y1,x2,y2),
                'conf': float,
                'cls': int
              }
        """
        results = self.model(
            frame_bgr,
            device=self.device,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
        )

        dets = []

        if len(results) == 0:
            return dets

        boxes = results[0].boxes
        if boxes is None:
            return dets

        for b in boxes:
            dets.append(
                {
                    "xyxy": tuple(map(int, b.xyxy[0].tolist())),
                    "conf": float(b.conf[0]),
                    "cls": int(b.cls[0]),
                }
            )

        return dets
