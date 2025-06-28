# BioViT3R-Beta File Utilities
# File I/O Operations for Various Formats Module

import os
import json
import yaml
import pickle
import csv
import h5py
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import zipfile
import tarfile
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import pandas as pd
from PIL import Image
import trimesh

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileUtils:
    """Comprehensive file I/O utilities for BioViT3R-Beta"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if necessary
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_extension(filepath: Union[str, Path]) -> str:
        """Get file extension in lowercase"""
        return Path(filepath).suffix.lower()
    
    @staticmethod
    def is_image_file(filepath: Union[str, Path]) -> bool:
        """Check if file is a supported image format"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        return FileUtils.get_file_extension(filepath) in image_extensions
    
    @staticmethod
    def is_video_file(filepath: Union[str, Path]) -> bool:
        """Check if file is a supported video format"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        return FileUtils.get_file_extension(filepath) in video_extensions
    
    @staticmethod
    def load_image(filepath: Union[str, Path], 
                   color_mode: str = 'RGB') -> np.ndarray:
        """
        Load image with various format support
        
        Args:
            filepath: Path to image file
            color_mode: Color mode ('RGB', 'BGR', 'GRAY')
            
        Returns:
            Image array
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Image not found: {filepath}")
        
        if color_mode == 'RGB':
            image = cv2.imread(str(filepath))
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_mode == 'BGR':
            return cv2.imread(str(filepath))
        elif color_mode == 'GRAY':
            return cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        
        # Fallback to PIL for other formats
        try:
            with Image.open(filepath) as pil_img:
                if color_mode == 'RGB':
                    return np.array(pil_img.convert('RGB'))
                elif color_mode == 'GRAY':
                    return np.array(pil_img.convert('L'))
        except Exception as e:
            logger.error(f"Failed to load image {filepath}: {e}")
            raise
    
    @staticmethod
    def save_image(image: np.ndarray, 
                   filepath: Union[str, Path],
                   quality: int = 95) -> bool:
        """
        Save image with format detection
        
        Args:
            image: Image array
            filepath: Output path
            quality: JPEG quality (1-100)
            
        Returns:
            Success status
        """
        try:
            filepath = Path(filepath)
            FileUtils.ensure_directory(filepath.parent)
            
            extension = FileUtils.get_file_extension(filepath)
            
            if extension in ['.jpg', '.jpeg']:
                # Convert RGB to BGR for OpenCV
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image
                cv2.imwrite(str(filepath), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif extension == '.png':
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image
                cv2.imwrite(str(filepath), image_bgr)
            else:
                # Use PIL for other formats
                if len(image.shape) == 3:
                    pil_img = Image.fromarray(image.astype(np.uint8))
                else:
                    pil_img = Image.fromarray(image.astype(np.uint8), mode='L')
                pil_img.save(filepath)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save image to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_point_cloud(filepath: Union[str, Path]) -> np.ndarray:
        """
        Load point cloud from various formats
        
        Args:
            filepath: Path to point cloud file
            
        Returns:
            Point cloud array (N, 3) or (N, 6) with colors
        """
        filepath = Path(filepath)
        extension = FileUtils.get_file_extension(filepath)
        
        try:
            if extension in ['.ply', '.pcd', '.xyz']:
                pcd = o3d.io.read_point_cloud(str(filepath))
                points = np.asarray(pcd.points)
                
                # Include colors if available
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)
                    return np.hstack([points, colors])
                return points
                
            elif extension == '.npy':
                return np.load(filepath)
                
            elif extension == '.npz':
                data = np.load(filepath)
                return data['points'] if 'points' in data else data[data.files[0]]
                
            elif extension == '.csv':
                df = pd.read_csv(filepath)
                return df.values
                
            else:
                raise ValueError(f"Unsupported point cloud format: {extension}")
                
        except Exception as e:
            logger.error(f"Failed to load point cloud from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_point_cloud(points: np.ndarray,
                        filepath: Union[str, Path],
                        colors: Optional[np.ndarray] = None,
                        normals: Optional[np.ndarray] = None) -> bool:
        """
        Save point cloud to various formats
        
        Args:
            points: Point coordinates (N, 3)
            filepath: Output path
            colors: Point colors (N, 3)
            normals: Point normals (N, 3)
            
        Returns:
            Success status
        """
        try:
            filepath = Path(filepath)
            FileUtils.ensure_directory(filepath.parent)
            extension = FileUtils.get_file_extension(filepath)
            
            if extension in ['.ply', '.pcd', '.xyz']:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                
                if normals is not None:
                    pcd.normals = o3d.utility.Vector3dVector(normals)
                
                o3d.io.write_point_cloud(str(filepath), pcd)
                
            elif extension == '.npy':
                if colors is not None:
                    data = np.hstack([points, colors])
                else:
                    data = points
                np.save(filepath, data)
                
            elif extension == '.npz':
                save_dict = {'points': points}
                if colors is not None:
                    save_dict['colors'] = colors
                if normals is not None:
                    save_dict['normals'] = normals
                np.savez(filepath, **save_dict)
                
            elif extension == '.csv':
                columns = ['x', 'y', 'z']
                data = points
                
                if colors is not None:
                    columns.extend(['r', 'g', 'b'])
                    data = np.hstack([data, colors])
                
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(filepath, index=False)
                
            return True
        except Exception as e:
            logger.error(f"Failed to save point cloud to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_mesh(filepath: Union[str, Path]) -> trimesh.Trimesh:
        """
        Load 3D mesh from file
        
        Args:
            filepath: Path to mesh file
            
        Returns:
            Trimesh object
        """
        try:
            return trimesh.load(str(filepath))
        except Exception as e:
            logger.error(f"Failed to load mesh from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_mesh(mesh: Union[trimesh.Trimesh, o3d.geometry.TriangleMesh],
                  filepath: Union[str, Path]) -> bool:
        """
        Save 3D mesh to file
        
        Args:
            mesh: Mesh object (trimesh or Open3D)
            filepath: Output path
            
        Returns:
            Success status
        """
        try:
            filepath = Path(filepath)
            FileUtils.ensure_directory(filepath.parent)
            
            if isinstance(mesh, trimesh.Trimesh):
                mesh.export(str(filepath))
            elif hasattr(mesh, 'vertices'):  # Open3D mesh
                o3d.io.write_triangle_mesh(str(filepath), mesh)
            else:
                raise ValueError("Unsupported mesh type")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save mesh to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_json(data: Dict[str, Any], 
                  filepath: Union[str, Path],
                  indent: int = 2) -> bool:
        """Save data to JSON file"""
        try:
            filepath = Path(filepath)
            FileUtils.ensure_directory(filepath.parent)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], 
                  filepath: Union[str, Path]) -> bool:
        """Save data to YAML file"""
        try:
            filepath = Path(filepath)
            FileUtils.ensure_directory(filepath.parent)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save YAML to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_pickle(filepath: Union[str, Path]) -> Any:
        """Load pickle file"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load pickle from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_pickle(data: Any, filepath: Union[str, Path]) -> bool:
        """Save data to pickle file"""
        try:
            filepath = Path(filepath)
            FileUtils.ensure_directory(filepath.parent)
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save pickle to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_hdf5(filepath: Union[str, Path], 
                  dataset_name: Optional[str] = None) -> Union[np.ndarray, Dict]:
        """
        Load data from HDF5 file
        
        Args:
            filepath: Path to HDF5 file
            dataset_name: Specific dataset name or None for all
            
        Returns:
            Dataset array or dictionary of datasets
        """
        try:
            with h5py.File(filepath, 'r') as f:
                if dataset_name:
                    return f[dataset_name][()]
                else:
                    return {key: f[key][()] for key in f.keys()}
        except Exception as e:
            logger.error(f"Failed to load HDF5 from {filepath}: {e}")
            raise
    
    @staticmethod
    def save_hdf5(data: Union[np.ndarray, Dict[str, np.ndarray]], 
                  filepath: Union[str, Path],
                  dataset_name: str = 'data') -> bool:
        """
        Save data to HDF5 file
        
        Args:
            data: Array or dictionary of arrays
            filepath: Output path
            dataset_name: Dataset name (if data is array)
            
        Returns:
            Success status
        """
        try:
            filepath = Path(filepath)
            FileUtils.ensure_directory(filepath.parent)
            
            with h5py.File(filepath, 'w') as f:
                if isinstance(data, dict):
                    for key, value in data.items():
                        f.create_dataset(key, data=value)
                else:
                    f.create_dataset(dataset_name, data=data)
            return True
        except Exception as e:
            logger.error(f"Failed to save HDF5 to {filepath}: {e}")
            return False
    
    @staticmethod
    def create_archive(source_path: Union[str, Path],
                      archive_path: Union[str, Path],
                      format: str = 'zip') -> bool:
        """
        Create archive from directory or files
        
        Args:
            source_path: Source directory or file
            archive_path: Output archive path
            format: Archive format ('zip', 'tar', 'tar.gz')
            
        Returns:
            Success status
        """
        try:
            source_path = Path(source_path)
            archive_path = Path(archive_path)
            FileUtils.ensure_directory(archive_path.parent)
            
            if format == 'zip':
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    if source_path.is_file():
                        zf.write(source_path, source_path.name)
                    else:
                        for file_path in source_path.rglob('*'):
                            if file_path.is_file():
                                relative_path = file_path.relative_to(source_path)
                                zf.write(file_path, relative_path)
            
            elif format in ['tar', 'tar.gz']:
                mode = 'w:gz' if format == 'tar.gz' else 'w'
                with tarfile.open(archive_path, mode) as tf:
                    tf.add(source_path, arcname=source_path.name)
            
            return True
        except Exception as e:
            logger.error(f"Failed to create archive {archive_path}: {e}")
            return False
    
    @staticmethod
    def extract_archive(archive_path: Union[str, Path],
                       extract_path: Union[str, Path]) -> bool:
        """
        Extract archive to directory
        
        Args:
            archive_path: Path to archive file
            extract_path: Extraction directory
            
        Returns:
            Success status
        """
        try:
            archive_path = Path(archive_path)
            extract_path = Path(extract_path)
            FileUtils.ensure_directory(extract_path)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_path)
            elif archive_path.suffix in ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:*') as tf:
                    tf.extractall(extract_path)
            
            return True
        except Exception as e:
            logger.error(f"Failed to extract archive {archive_path}: {e}")
            return False
    
    @staticmethod
    def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return {'exists': False}
        
        stat = filepath.stat()
        
        info = {
            'exists': True,
            'name': filepath.name,
            'stem': filepath.stem,
            'suffix': filepath.suffix,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'is_file': filepath.is_file(),
            'is_dir': filepath.is_dir(),
            'absolute_path': str(filepath.absolute())
        }
        
        # Add image-specific info
        if FileUtils.is_image_file(filepath):
            try:
                with Image.open(filepath) as img:
                    info['image_size'] = img.size
                    info['image_mode'] = img.mode
                    info['image_format'] = img.format
            except:
                pass
        
        return info
    
    @staticmethod
    def cleanup_temp_files(directory: Union[str, Path], 
                          pattern: str = "temp_*",
                          max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age
        
        Args:
            directory: Directory to clean
            pattern: File pattern to match
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files deleted
        """
        directory = Path(directory)
        deleted_count = 0
        
        if not directory.exists():
            return 0
        
        max_age_seconds = max_age_hours * 3600
        current_time = datetime.now().timestamp()
        
        try:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted temp file: {file_path}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return deleted_count

# Utility functions for common file operations
def batch_convert_images(input_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        target_format: str = 'jpg',
                        quality: int = 95) -> int:
    """Batch convert images to target format"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    FileUtils.ensure_directory(output_dir)
    
    converted_count = 0
    
    for img_path in input_dir.rglob('*'):
        if FileUtils.is_image_file(img_path):
            try:
                image = FileUtils.load_image(img_path)
                output_path = output_dir / f"{img_path.stem}.{target_format}"
                
                if FileUtils.save_image(image, output_path, quality):
                    converted_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to convert {img_path}: {e}")
    
    return converted_count

def find_files_by_size(directory: Union[str, Path],
                      min_size_mb: float = 0,
                      max_size_mb: float = float('inf')) -> List[Path]:
    """Find files within size range"""
    directory = Path(directory)
    matching_files = []
    
    min_bytes = min_size_mb * 1024 * 1024
    max_bytes = max_size_mb * 1024 * 1024
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            size = file_path.stat().st_size
            if min_bytes <= size <= max_bytes:
                matching_files.append(file_path)
    
    return matching_files