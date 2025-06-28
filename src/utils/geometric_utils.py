# BioViT3R-Beta Geometric Utilities
# 3D Geometry Calculations and Transformations Module

import numpy as np
import cv2
from typing import Tuple, List, Optional, Union, Dict, Any
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.transform import Rotation
import trimesh
from dataclasses import dataclass
import open3d as o3d

@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters"""
    fx: float  # Focal length in x
    fy: float  # Focal length in y  
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int  # Image width
    height: int  # Image height
    k1: float = 0.0  # Radial distortion
    k2: float = 0.0  # Radial distortion
    p1: float = 0.0  # Tangential distortion
    p2: float = 0.0  # Tangential distortion

class GeometricUtils:
    """Comprehensive 3D geometry utilities for plant analysis"""
    
    @staticmethod
    def estimate_camera_params(image_size: Tuple[int, int], 
                              fov_degrees: float = 60.0) -> CameraParameters:
        """
        Estimate camera parameters from image size and field of view
        
        Args:
            image_size: (width, height) of image
            fov_degrees: Horizontal field of view in degrees
            
        Returns:
            Estimated camera parameters
        """
        width, height = image_size
        
        # Estimate focal length from FOV
        fx = width / (2 * np.tan(np.radians(fov_degrees) / 2))
        fy = fx  # Assume square pixels
        
        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0
        
        return CameraParameters(fx, fy, cx, cy, width, height)
    
    @staticmethod
    def pixel_to_world(pixel_coords: np.ndarray,
                       depth_map: np.ndarray,
                       camera_params: CameraParameters) -> np.ndarray:
        """
        Convert pixel coordinates to 3D world coordinates
        
        Args:
            pixel_coords: Pixel coordinates (N, 2) as [u, v]
            depth_map: Depth values for each pixel
            camera_params: Camera parameters
            
        Returns:
            3D world coordinates (N, 3) as [x, y, z]
        """
        if pixel_coords.ndim == 1:
            pixel_coords = pixel_coords.reshape(1, -1)
            
        u, v = pixel_coords[:, 0], pixel_coords[:, 1]
        
        # Get depth values
        if depth_map.ndim == 2:
            z = depth_map[v.astype(int), u.astype(int)]
        else:
            z = depth_map
            
        # Convert to world coordinates
        x = (u - camera_params.cx) * z / camera_params.fx
        y = (v - camera_params.cy) * z / camera_params.fy
        
        return np.column_stack([x, y, z])
    
    @staticmethod
    def world_to_pixel(world_coords: np.ndarray,
                       camera_params: CameraParameters) -> np.ndarray:
        """
        Project 3D world coordinates to pixel coordinates
        
        Args:
            world_coords: 3D coordinates (N, 3) as [x, y, z]
            camera_params: Camera parameters
            
        Returns:
            Pixel coordinates (N, 2) as [u, v]
        """
        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)
            
        x, y, z = world_coords[:, 0], world_coords[:, 1], world_coords[:, 2]
        
        # Avoid division by zero
        z = np.where(z == 0, 1e-6, z)
        
        u = camera_params.fx * x / z + camera_params.cx
        v = camera_params.fy * y / z + camera_params.cy
        
        return np.column_stack([u, v])
    
    @staticmethod
    def compute_point_cloud_normals(points: np.ndarray,
                                   k_neighbors: int = 20) -> np.ndarray:
        """
        Compute normal vectors for point cloud
        
        Args:
            points: Point cloud coordinates (N, 3)
            k_neighbors: Number of neighbors for normal estimation
            
        Returns:
            Normal vectors (N, 3)
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )
        
        return np.asarray(pcd.normals)
    
    @staticmethod
    def filter_statistical_outliers(points: np.ndarray,
                                   nb_neighbors: int = 20,
                                   std_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove statistical outliers from point cloud
        
        Args:
            points: Point cloud coordinates (N, 3)
            nb_neighbors: Number of neighbors to analyze
            std_ratio: Standard deviation ratio threshold
            
        Returns:
            Filtered points and indices of inliers
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        filtered_pcd, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        filtered_points = np.asarray(filtered_pcd.points)
        return filtered_points, np.array(inlier_indices)
    
    @staticmethod
    def compute_plant_volume(points: np.ndarray,
                            method: str = 'convex_hull') -> float:
        """
        Compute plant volume from point cloud
        
        Args:
            points: Point cloud coordinates (N, 3)
            method: Volume computation method ('convex_hull', 'alpha_shape')
            
        Returns:
            Estimated volume in cubic units
        """
        if method == 'convex_hull':
            try:
                hull = ConvexHull(points)
                return hull.volume
            except:
                return 0.0
                
        elif method == 'alpha_shape':
            # Create mesh using alpha shapes
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Compute normals if not available
            if not pcd.has_normals():
                pcd.estimate_normals()
                
            # Create alpha shape mesh
            alpha = 0.03
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha
            )
            
            if mesh.is_watertight():
                return mesh.get_volume()
            else:
                return 0.0
        
        return 0.0
    
    @staticmethod
    def compute_surface_area(points: np.ndarray,
                            mesh: Optional[Any] = None) -> float:
        """
        Compute plant surface area
        
        Args:
            points: Point cloud coordinates (N, 3)
            mesh: Optional precomputed mesh
            
        Returns:
            Surface area estimation
        """
        if mesh is not None and hasattr(mesh, 'get_surface_area'):
            return mesh.get_surface_area()
        
        # Estimate surface area from point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Create Poisson mesh
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        return mesh.get_surface_area()
    
    @staticmethod
    def compute_plant_height(points: np.ndarray,
                            percentile: float = 95.0) -> float:
        """
        Compute plant height using percentile method
        
        Args:
            points: Point cloud coordinates (N, 3)
            percentile: Percentile for height calculation
            
        Returns:
            Plant height
        """
        z_coords = points[:, 2]
        max_height = np.percentile(z_coords, percentile)
        min_height = np.percentile(z_coords, 5.0)  # Ground level
        
        return max_height - min_height
    
    @staticmethod
    def compute_canopy_coverage(points: np.ndarray,
                               grid_resolution: float = 0.01) -> Dict[str, float]:
        """
        Compute canopy coverage metrics
        
        Args:
            points: Point cloud coordinates (N, 3)
            grid_resolution: Grid cell size for coverage calculation
            
        Returns:
            Dictionary with coverage metrics
        """
        # Project points to XY plane
        xy_points = points[:, :2]
        
        # Create bounding box
        min_coords = np.min(xy_points, axis=0)
        max_coords = np.max(xy_points, axis=0)
        
        # Create grid
        x_bins = np.arange(min_coords[0], max_coords[0] + grid_resolution, grid_resolution)
        y_bins = np.arange(min_coords[1], max_coords[1] + grid_resolution, grid_resolution)
        
        # Count occupied cells
        hist, _, _ = np.histogram2d(xy_points[:, 0], xy_points[:, 1], 
                                   bins=[x_bins, y_bins])
        
        occupied_cells = np.sum(hist > 0)
        total_cells = len(x_bins) * len(y_bins)
        
        coverage_ratio = occupied_cells / total_cells if total_cells > 0 else 0
        canopy_area = occupied_cells * (grid_resolution ** 2)
        
        return {
            'coverage_ratio': coverage_ratio,
            'canopy_area': canopy_area,
            'occupied_cells': occupied_cells,
            'total_cells': total_cells
        }
    
    @staticmethod
    def align_point_clouds(source: np.ndarray,
                          target: np.ndarray,
                          method: str = 'icp') -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two point clouds using ICP or other methods
        
        Args:
            source: Source point cloud (N, 3)
            target: Target point cloud (M, 3)
            method: Alignment method ('icp', 'coherent_point_drift')
            
        Returns:
            Aligned source points and transformation matrix
        """
        if method == 'icp':
            # Use Open3D ICP
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target)
            
            # Perform ICP
            threshold = 0.02
            trans_init = np.eye(4)
            
            reg_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # Apply transformation
            transformation = reg_result.transformation
            source_pcd.transform(transformation)
            
            aligned_points = np.asarray(source_pcd.points)
            return aligned_points, transformation
        
        return source, np.eye(4)
    
    @staticmethod
    def compute_geometric_features(points: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive geometric features from point cloud
        
        Args:
            points: Point cloud coordinates (N, 3)
            
        Returns:
            Dictionary of geometric features
        """
        features = {}
        
        # Basic statistics
        features['num_points'] = len(points)
        features['centroid_x'] = np.mean(points[:, 0])
        features['centroid_y'] = np.mean(points[:, 1])
        features['centroid_z'] = np.mean(points[:, 2])
        
        # Bounding box features
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        bbox_size = max_coords - min_coords
        
        features['bbox_width'] = bbox_size[0]
        features['bbox_length'] = bbox_size[1]
        features['bbox_height'] = bbox_size[2]
        features['bbox_volume'] = np.prod(bbox_size)
        
        # Plant-specific metrics
        features['plant_height'] = GeometricUtils.compute_plant_height(points)
        features['estimated_volume'] = GeometricUtils.compute_plant_volume(points)
        
        # Canopy metrics
        canopy_metrics = GeometricUtils.compute_canopy_coverage(points)
        features.update(canopy_metrics)
        
        # Density features
        if len(points) > 0:
            features['point_density'] = len(points) / features['bbox_volume'] if features['bbox_volume'] > 0 else 0
        
        # Shape complexity
        if len(points) >= 4:
            try:
                hull = ConvexHull(points)
                features['convex_hull_volume'] = hull.volume
                features['convex_hull_area'] = hull.area
                features['shape_complexity'] = features['estimated_volume'] / features['convex_hull_volume'] if features['convex_hull_volume'] > 0 else 0
            except:
                features['convex_hull_volume'] = 0
                features['convex_hull_area'] = 0
                features['shape_complexity'] = 0
        
        return features
    
    @staticmethod
    def rotate_points(points: np.ndarray,
                     rotation_matrix: np.ndarray,
                     center: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Rotate point cloud around a center point
        
        Args:
            points: Point cloud coordinates (N, 3)
            rotation_matrix: 3x3 rotation matrix
            center: Center of rotation (default: centroid)
            
        Returns:
            Rotated points
        """
        if center is None:
            center = np.mean(points, axis=0)
        
        # Translate to origin
        centered_points = points - center
        
        # Apply rotation
        rotated_points = centered_points @ rotation_matrix.T
        
        # Translate back
        return rotated_points + center
    
    @staticmethod
    def compute_principal_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute principal axes of point cloud using PCA
        
        Args:
            points: Point cloud coordinates (N, 3)
            
        Returns:
            Eigenvalues and eigenvectors (principal axes)
        """
        # Center the points
        centered_points = points - np.mean(points, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        return eigenvalues, eigenvectors

# Utility functions for common geometric operations
def distance_point_to_plane(point: np.ndarray, 
                           plane_point: np.ndarray, 
                           plane_normal: np.ndarray) -> float:
    """Compute distance from point to plane"""
    return np.abs(np.dot(point - plane_point, plane_normal)) / np.linalg.norm(plane_normal)

def project_point_to_plane(point: np.ndarray,
                          plane_point: np.ndarray,
                          plane_normal: np.ndarray) -> np.ndarray:
    """Project point onto plane"""
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    distance = np.dot(point - plane_point, plane_normal)
    return point - distance * plane_normal

def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors in radians"""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    return np.arccos(cos_angle)