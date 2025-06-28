"""
PlantAnalyzer
Analyze plant health and growth stage using computer vision.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, Any
from pathlib import Path
import logging

class HealthClassifier(nn.Module):
    """Simple health classification model placeholder."""
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class PlantAnalyzer:
    """High-level plant health and growth analyzer."""
    def __init__(self, model_paths: Dict[str, Any], device: str = "auto"):
        self.device = self._get_device(device)
        self.health_model = HealthClassifier().to(self.device)
        self.model_path = Path(model_paths.get('health_classification_path', 'models/health_classification/'))
        self._load_models()

    def _get_device(self, device):
        if device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _load_models(self):
        if (self.model_path / 'health_model.pth').exists():
            try:
                state = torch.load(self.model_path / 'health_model.pth', map_location=self.device)
                self.health_model.load_state_dict(state)
                logging.info("Health model loaded")
            except Exception as e:
                logging.warning(f"Failed to load health model: {e}")

    def preprocess(self, image: np.ndarray, size=(224,224)) -> torch.Tensor:
        if len(image.shape)==3 and image.shape[2]==3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size).astype(np.float32)/255.0
        tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(self.device)
        return tensor

    def analyze_health(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            self.health_model.eval()
            inp = self.preprocess(image)
            with torch.no_grad():
                logits = self.health_model(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            health_score = float(probs.max())
            detected_class = int(probs.argmax())
            return {
                'health_score': health_score,
                'detected_class': detected_class,
                'probabilities': probs.tolist()
            }
        except Exception as e:
            logging.error(f"Health analysis failed: {e}")
            return {}
