"""
FruitDetector
Detect fruits in images using pre-trained Detectoron2 model.
"""

import torch
import cv2
import numpy as np
from typing import Dict, Any, List
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import logging
from pathlib import Path

class FruitDetector:
    """High-level fruit detection interface."""

    def __init__(self, model_path: str = 'models/fruit_detection/', device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.cfg = self._setup_cfg(model_path)
        self.predictor = DefaultPredictor(self.cfg)

    def _setup_cfg(self, model_path: str):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # apples
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = str(Path(model_path) / 'model_final.pth')
        cfg.MODEL.DEVICE = self.device
        return cfg

    def detect_fruits(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            outputs = self.predictor(image)
            instances = outputs['instances']
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()

            detections: List[Dict[str, Any]] = []
            for box, score in zip(boxes, scores):
                detections.append({
                    'bbox': box.tolist(),
                    'score': float(score)
                })

            return {
                'num_fruits': len(detections),
                'detections': detections
            }
        except Exception as e:
            logging.error(f"Fruit detection failed: {e}")
            return {
                'num_fruits': 0,
                'detections': []
            }