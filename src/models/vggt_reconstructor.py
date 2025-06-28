"""
VGGT 3D Reconstructor
Implements VGGT (Vision-Guided Gaussian Splatting Transformer) for single-image 3D reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import open3d as o3d
from pathlib import Path
import logging

class VGGTEncoder(nn.Module):
    """VGGT Encoder for feature extraction from single images."""
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 512):
        super().__init__()
        
        # Convolutional backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(input_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(feature_dim, 8, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Feature maps [B, feature_dim, H//8, W//8]
        """
        features = self.backbone(x)
        
        # Reshape for attention
        B, C, H, W = features.shape
        features_flat = features.view(B, C, H*W).transpose(1, 2)  # [B, H*W, C]
        
        # Apply attention
        attended_features, _ = self.attention(features_flat, features_flat, features_flat)
        
        # Reshape back
        attended_features = attended_features.transpose(1, 2).view(B, C, H, W)
        
        return attended_features

class VGGTDecoder(nn.Module):
    """VGGT Decoder for 3D point cloud generation."""
    
    def __init__(self, feature_dim: int = 512, max_points: int = 4096):
        super().__init__()
        
        self.max_points = max_points
        
        # Point cloud prediction head
        self.point_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, max_points * 3),  # 3D coordinates
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, max_points),
            nn.Sigmoid()
        )
        
        # Color prediction
        self.color_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, max_points * 3),  # RGB colors
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through decoder.
        
        Args:
            features: Feature maps from encoder [B, feature_dim, H, W]
            
        Returns:
            Dictionary with points, colors, and confidence scores
        """
        # Predict 3D points
        points = self.point_head(features)
        points = points.view(-1, self.max_points, 3)
        
        # Predict confidence scores
        confidence = self.confidence_head(features)
        confidence = confidence.view(-1, self.max_points, 1)
        
        # Predict colors
        colors = self.color_head(features)
        colors = colors.view(-1, self.max_points, 3)
        
        return {
            'points': points,
            'colors': colors,
            'confidence': confidence
        }

class VGGTModel(nn.Module):
    """Complete VGGT Model for 3D reconstruction."""
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 512, max_points: int = 4096):
        super().__init__()
        
        self.encoder = VGGTEncoder(input_dim, feature_dim)
        self.decoder = VGGTDecoder(feature_dim, max_points)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Complete forward pass."""
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs

class VGGTReconstructor:
    """High-level interface for VGGT 3D reconstruction."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize VGGT Reconstructor.
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on
        """
        self.device = self._get_device(device)
        self.model = VGGTModel().to(self.device)
        self.model_path = Path(model_path)
        
        # Load model if exists
        if self.model_path.exists() and (self.model_path / "model.pth").exists():
            self.load_model()
        else:
            logging.warning(f"Model not found at {model_path}. Using random initialization.")
            
        self.model.eval()
        
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def load_model(self):
        """Load trained model weights."""
        try:
            checkpoint = torch.load(self.model_path / "model.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Preprocess input image for model inference.
        
        Args:
            image: Input image as numpy array
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def reconstruct_3d(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[Optional[np.ndarray], Optional[o3d.geometry.TriangleMesh]]:
        """
        Reconstruct 3D point cloud from single image.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for points
            
        Returns:
            Tuple of (point_cloud, mesh) or (None, None) if reconstruction fails
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
            
            # Extract outputs
            points = outputs['points'][0].cpu().numpy()  # [N, 3]
            colors = outputs['colors'][0].cpu().numpy()  # [N, 3]
            confidence = outputs['confidence'][0].cpu().numpy()  # [N, 1]
            
            # Filter by confidence
            valid_mask = confidence.flatten() > confidence_threshold
            if not np.any(valid_mask):
                logging.warning("No points above confidence threshold")
                return None, None
            
            points = points[valid_mask]
            colors = colors[valid_mask]
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Create mesh using Poisson reconstruction
            mesh = self.create_mesh_from_points(pcd)
            
            return points, mesh
            
        except Exception as e:
            logging.error(f"3D reconstruction failed: {e}")
            return None, None
    
    def create_mesh_from_points(self, point_cloud: o3d.geometry.PointCloud) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Create triangle mesh from point cloud using Poisson reconstruction.
        
        Args:
            point_cloud: Open3D point cloud
            
        Returns:
            Triangle mesh or None if reconstruction fails
        """
        try:
            # Estimate normals
            point_cloud.estimate_normals()
            
            # Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                point_cloud, depth=9
            )
            
            # Remove outlier triangles
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            return mesh
            
        except Exception as e:
            logging.error(f"Mesh creation failed: {e}")
            return None
    
    def export_ply(self, points: np.ndarray, colors: np.ndarray, output_path: str):
        """Export point cloud to PLY format."""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            o3d.io.write_point_cloud(output_path, pcd)
            logging.info(f"Point cloud exported to {output_path}")
            
        except Exception as e:
            logging.error(f"PLY export failed: {e}")
    
    def export_obj(self, mesh: o3d.geometry.TriangleMesh, output_path: str):
        """Export mesh to OBJ format."""
        try:
            o3d.io.write_triangle_mesh(output_path, mesh)
            logging.info(f"Mesh exported to {output_path}")
            
        except Exception as e:
            logging.error(f"OBJ export failed: {e}")
    
    def get_reconstruction_info(self, points: np.ndarray, mesh: o3d.geometry.TriangleMesh = None) -> Dict[str, Any]:
        """Get information about the reconstruction."""
        info = {
            'num_points': len(points) if points is not None else 0,
            'bounds': {
                'min': points.min(axis=0).tolist() if points is not None else None,
                'max': points.max(axis=0).tolist() if points is not None else None,
            },
            'has_mesh': mesh is not None,
        }
        
        if mesh is not None:
            info.update({
                'num_vertices': len(mesh.vertices),
                'num_triangles': len(mesh.triangles),
                'is_watertight': mesh.is_watertight(),
                'is_orientable': mesh.is_orientable(),
            })
            
        return info