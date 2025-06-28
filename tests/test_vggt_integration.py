#!/usr/bin/env python3
"""
BioViT3R-Beta VGGT Integration Tests
Comprehensive tests for VGGT 3D reconstruction functionality and integration.
"""

import unittest
import numpy as np
import torch
import cv2
import tempfile
import shutil
from pathlib import Path
import sys
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.vggt_reconstructor import VGGTReconstructor
from src.utils.geometric_utils import validate_point_cloud, calculate_mesh_quality

class TestVGGTIntegration(unittest.TestCase):
    """Test VGGT 3D reconstruction integration and functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {cls.device}")

        # Create temporary directory for test outputs
        cls.temp_dir = Path(tempfile.mkdtemp())

        # Initialize VGGT reconstructor
        cls.vggt = VGGTReconstructor(device=cls.device)

        # Create test images
        cls.test_images = cls._create_test_images()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _create_test_images(cls):
        """Create synthetic test images for reconstruction."""
        test_images = {}

        # Simple gradient image
        gradient_img = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(224):
            gradient_img[i, :, 0] = int(255 * i / 224)  # Red gradient
        test_images["gradient"] = gradient_img

        # Checkerboard pattern
        checkerboard = np.zeros((224, 224, 3), dtype=np.uint8)
        square_size = 16
        for i in range(0, 224, square_size):
            for j in range(0, 224, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    checkerboard[i:i+square_size, j:j+square_size] = 255
        test_images["checkerboard"] = checkerboard

        # Circular pattern (simulating fruit)
        circular = np.zeros((224, 224, 3), dtype=np.uint8)
        center = (112, 112)
        radius = 80
        cv2.circle(circular, center, radius, (0, 255, 0), -1)  # Green circle
        cv2.circle(circular, center, radius//2, (255, 0, 0), -1)  # Red center
        test_images["circular"] = circular

        return test_images

    def test_vggt_initialization(self):
        """Test VGGT model initialization."""
        # Test model loads without errors
        self.assertIsNotNone(self.vggt)
        self.assertEqual(self.vggt.device, self.device)

        # Test model parameters
        self.assertTrue(hasattr(self.vggt, 'model'))
        self.assertTrue(hasattr(self.vggt, 'preprocessor'))

    def test_image_preprocessing(self):
        """Test image preprocessing pipeline."""
        for image_name, image in self.test_images.items():
            with self.subTest(image=image_name):
                # Test preprocessing
                processed = self.vggt.preprocess_image(image)

                # Check tensor properties
                self.assertIsInstance(processed, torch.Tensor)
                self.assertEqual(processed.device, self.device)
                self.assertEqual(processed.shape[0], 1)  # Batch dimension
                self.assertEqual(processed.shape[1], 3)  # RGB channels

                # Check value range (should be normalized)
                self.assertGreaterEqual(processed.min().item(), -3.0)
                self.assertLessEqual(processed.max().item(), 3.0)

    def test_single_image_reconstruction(self):
        """Test 3D reconstruction from single images."""
        for image_name, image in self.test_images.items():
            with self.subTest(image=image_name):
                # Perform reconstruction
                result = self.vggt.reconstruct_3d(image)

                # Check result structure
                self.assertIsInstance(result, dict)
                self.assertIn("point_cloud", result)
                self.assertIn("depth_map", result)

                # Validate point cloud
                point_cloud = result["point_cloud"]
                self.assertIsInstance(point_cloud, np.ndarray)
                self.assertEqual(point_cloud.shape[1], 3)  # 3D coordinates
                self.assertGreater(point_cloud.shape[0], 0)  # Non-empty

                # Validate depth map
                depth_map = result["depth_map"]
                self.assertIsInstance(depth_map, np.ndarray)
                self.assertEqual(len(depth_map.shape), 2)  # 2D depth map

    def test_mesh_generation(self):
        """Test mesh generation from point clouds."""
        image = self.test_images["circular"]  # Use circular pattern for best mesh

        # Reconstruct with mesh generation enabled
        result = self.vggt.reconstruct_3d(image, generate_mesh=True)

        # Check mesh is generated
        self.assertIn("mesh", result)
        mesh = result["mesh"]

        # Validate mesh structure
        self.assertIn("vertices", mesh)
        self.assertIn("faces", mesh)

        vertices = mesh["vertices"]
        faces = mesh["faces"]

        self.assertIsInstance(vertices, np.ndarray)
        self.assertIsInstance(faces, np.ndarray)
        self.assertEqual(vertices.shape[1], 3)  # 3D vertices
        self.assertEqual(faces.shape[1], 3)     # Triangular faces

    def test_reconstruction_consistency(self):
        """Test reconstruction consistency across multiple runs."""
        image = self.test_images["gradient"]

        # Perform multiple reconstructions
        results = []
        for i in range(3):
            result = self.vggt.reconstruct_3d(image)
            results.append(result)

        # Check consistency
        for i in range(1, len(results)):
            pc1 = results[0]["point_cloud"]
            pc2 = results[i]["point_cloud"]

            # Point clouds should have similar number of points
            point_diff = abs(len(pc1) - len(pc2)) / max(len(pc1), len(pc2))
            self.assertLess(point_diff, 0.1)  # Less than 10% difference

    def test_batch_reconstruction(self):
        """Test batch processing of multiple images."""
        images = list(self.test_images.values())

        # Batch reconstruction
        results = self.vggt.reconstruct_batch(images)

        # Check results
        self.assertEqual(len(results), len(images))

        for i, result in enumerate(results):
            self.assertIsInstance(result, dict)
            self.assertIn("point_cloud", result)
            self.assertIn("depth_map", result)

    def test_point_cloud_validation(self):
        """Test point cloud validation utilities."""
        image = self.test_images["circular"]
        result = self.vggt.reconstruct_3d(image)
        point_cloud = result["point_cloud"]

        # Test validation function
        validation_result = validate_point_cloud(point_cloud)

        self.assertIsInstance(validation_result, dict)
        self.assertIn("valid", validation_result)
        self.assertIn("num_points", validation_result)
        self.assertIn("bounds", validation_result)

        # Point cloud should be valid for test images
        self.assertTrue(validation_result["valid"])
        self.assertGreater(validation_result["num_points"], 0)

    def test_reconstruction_parameters(self):
        """Test different reconstruction parameter settings."""
        image = self.test_images["checkerboard"]

        # Test different quality settings
        quality_settings = ["low", "medium", "high"]

        for quality in quality_settings:
            with self.subTest(quality=quality):
                result = self.vggt.reconstruct_3d(image, quality=quality)

                self.assertIn("point_cloud", result)
                point_cloud = result["point_cloud"]

                # Higher quality should generally produce more points
                if quality == "high":
                    self.assertGreater(len(point_cloud), 1000)
                elif quality == "low":
                    self.assertLess(len(point_cloud), 5000)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid image shapes
        invalid_images = [
            np.zeros((100, 100)),           # Missing channel dimension
            np.zeros((100, 100, 1)),        # Single channel
            np.zeros((50, 50, 3)),          # Too small
            np.zeros((1000, 1000, 3)),      # Too large
        ]

        for i, invalid_img in enumerate(invalid_images):
            with self.subTest(invalid_image=i):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        result = self.vggt.reconstruct_3d(invalid_img)
                        # Should either handle gracefully or raise appropriate exception
                        if result is not None:
                            self.assertIsInstance(result, dict)
                    except (ValueError, RuntimeError, TypeError):
                        # Expected exceptions are acceptable
                        pass

    def test_output_formats(self):
        """Test different output format options."""
        image = self.test_images["circular"]

        # Test PLY format export
        result = self.vggt.reconstruct_3d(image, output_format="ply")
        self.assertIn("ply_data", result)

        # Test OBJ format export
        result = self.vggt.reconstruct_3d(image, output_format="obj")
        self.assertIn("obj_data", result)

    def test_memory_usage(self):
        """Test memory usage during reconstruction."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform multiple reconstructions
        for _ in range(5):
            for image in self.test_images.values():
                result = self.vggt.reconstruct_3d(image)
                # Explicitly delete to help garbage collection
                del result

        # Force garbage collection
        import gc
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for test)
        self.assertLess(memory_increase, 500, 
                       f"Memory usage increased by {memory_increase:.1f}MB")

    def test_performance_benchmarks(self):
        """Test reconstruction performance benchmarks."""
        import time

        image = self.test_images["circular"]

        # Warm-up run
        self.vggt.reconstruct_3d(image)

        # Timed runs
        times = []
        for _ in range(5):
            start_time = time.time()
            result = self.vggt.reconstruct_3d(image)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Performance should be reasonable
        if self.device.type == "cuda":
            self.assertLess(avg_time, 5.0, "GPU reconstruction should be under 5 seconds")
        else:
            self.assertLess(avg_time, 30.0, "CPU reconstruction should be under 30 seconds")

        # Timing should be consistent (low standard deviation)
        self.assertLess(std_time / avg_time, 0.5, "Timing should be relatively consistent")


class TestVGGTUtilities(unittest.TestCase):
    """Test VGGT utility functions."""

    def test_geometric_utilities(self):
        """Test geometric utility functions."""
        # Create sample point cloud
        point_cloud = np.random.rand(1000, 3) * 10  # Random 3D points

        # Test point cloud validation
        validation = validate_point_cloud(point_cloud)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["num_points"], 1000)

        # Test bounds calculation
        bounds = validation["bounds"]
        self.assertEqual(len(bounds), 3)  # x, y, z bounds
        for bound in bounds:
            self.assertEqual(len(bound), 2)  # min, max

    def test_mesh_quality_metrics(self):
        """Test mesh quality calculation."""
        # Create simple triangle mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0], 
            [0, 1, 0],
            [0, 0, 1]
        ])

        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])

        mesh = {"vertices": vertices, "faces": faces}
        quality = calculate_mesh_quality(mesh)

        self.assertIsInstance(quality, dict)
        self.assertIn("surface_area", quality)
        self.assertIn("volume", quality)
        self.assertIn("aspect_ratio", quality)


if __name__ == "__main__":
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
