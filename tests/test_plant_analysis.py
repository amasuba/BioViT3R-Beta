#!/usr/bin/env python3
"""
BioViT3R-Beta Plant Analysis Pipeline Tests
Comprehensive tests for plant health analysis, growth classification, and biomass estimation.
"""

import unittest
import numpy as np
import torch
import cv2
import tempfile
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.plant_analyzer import PlantAnalyzer
from src.models.fruit_detector import FruitDetector
from src.utils.metrics import ClassificationMetrics

class TestPlantAnalysisPipeline(unittest.TestCase):
    """Test complete plant analysis pipeline functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        cls.plant_analyzer = PlantAnalyzer(device=cls.device)
        cls.fruit_detector = FruitDetector(device=cls.device)

        # Create test images
        cls.test_images = cls._create_plant_test_images()

    @classmethod
    def _create_plant_test_images(cls):
        """Create synthetic plant images for testing."""
        test_images = {}

        # Healthy plant image (green dominant)
        healthy_plant = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create green leafy pattern
        for i in range(5):
            center = (50 + i*40, 128)
            cv2.ellipse(healthy_plant, center, (30, 60), 0, 0, 360, (0, 180, 0), -1)
        test_images["healthy_plant"] = healthy_plant

        # Diseased plant image (brown spots)
        diseased_plant = healthy_plant.copy()
        # Add brown disease spots
        for i in range(8):
            center = (np.random.randint(50, 200), np.random.randint(50, 200))
            cv2.circle(diseased_plant, center, np.random.randint(5, 15), (139, 69, 19), -1)
        test_images["diseased_plant"] = diseased_plant

        # Plant with fruits (green leaves + red fruits)
        fruiting_plant = healthy_plant.copy()
        # Add red circular fruits
        fruit_centers = [(80, 100), (120, 140), (160, 90), (200, 130)]
        for center in fruit_centers:
            cv2.circle(fruiting_plant, center, 15, (255, 0, 0), -1)
        test_images["fruiting_plant"] = fruiting_plant

        # Stressed plant (yellow/brown coloring)
        stressed_plant = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(5):
            center = (50 + i*40, 128)
            cv2.ellipse(stressed_plant, center, (30, 60), 0, 0, 360, (255, 255, 0), -1)
        test_images["stressed_plant"] = stressed_plant

        return test_images

    def test_plant_analyzer_initialization(self):
        """Test plant analyzer initialization."""
        self.assertIsNotNone(self.plant_analyzer)
        self.assertEqual(self.plant_analyzer.device, self.device)

        # Test model components
        self.assertTrue(hasattr(self.plant_analyzer, 'health_classifier'))
        self.assertTrue(hasattr(self.plant_analyzer, 'growth_classifier'))

    def test_health_classification(self):
        """Test plant health classification."""
        for image_name, image in self.test_images.items():
            with self.subTest(image=image_name):
                # Perform health analysis
                health_result = self.plant_analyzer.analyze_health(image)

                # Check result structure
                self.assertIsInstance(health_result, dict)
                self.assertIn("health_score", health_result)
                self.assertIn("disease_probability", health_result)
                self.assertIn("stress_indicators", health_result)
                self.assertIn("confidence", health_result)

                # Validate score ranges
                self.assertGreaterEqual(health_result["health_score"], 0.0)
                self.assertLessEqual(health_result["health_score"], 1.0)
                self.assertGreaterEqual(health_result["confidence"], 0.0)
                self.assertLessEqual(health_result["confidence"], 1.0)

                # Semantic validation based on image type
                if "healthy" in image_name:
                    self.assertGreater(health_result["health_score"], 0.5)
                elif "diseased" in image_name or "stressed" in image_name:
                    self.assertLess(health_result["health_score"], 0.7)

    def test_growth_stage_classification(self):
        """Test plant growth stage classification."""
        for image_name, image in self.test_images.items():
            with self.subTest(image=image_name):
                # Perform growth analysis
                growth_result = self.plant_analyzer.classify_growth_stage(image)

                # Check result structure
                self.assertIsInstance(growth_result, dict)
                self.assertIn("growth_stage", growth_result)
                self.assertIn("stage_confidence", growth_result)
                self.assertIn("estimated_days", growth_result)

                # Validate growth stage
                valid_stages = ["seedling", "vegetative", "flowering", "fruiting", "senescence"]
                self.assertIn(growth_result["growth_stage"], valid_stages)

                # Semantic validation
                if "fruiting" in image_name:
                    self.assertEqual(growth_result["growth_stage"], "fruiting")

    def test_comprehensive_plant_analysis(self):
        """Test comprehensive plant analysis combining all metrics."""
        image = self.test_images["fruiting_plant"]

        # Perform comprehensive analysis
        analysis_result = self.plant_analyzer.comprehensive_analysis(image)

        # Check complete result structure
        self.assertIsInstance(analysis_result, dict)

        # Health analysis components
        self.assertIn("health_analysis", analysis_result)
        health = analysis_result["health_analysis"]
        self.assertIn("health_score", health)
        self.assertIn("disease_probability", health)

        # Growth analysis components
        self.assertIn("growth_analysis", analysis_result)
        growth = analysis_result["growth_analysis"]
        self.assertIn("growth_stage", growth)
        self.assertIn("stage_confidence", growth)

        # Environmental analysis
        self.assertIn("environmental_analysis", analysis_result)
        env = analysis_result["environmental_analysis"]
        self.assertIn("light_conditions", env)
        self.assertIn("water_stress", env)

        # Recommendations
        self.assertIn("recommendations", analysis_result)
        recommendations = analysis_result["recommendations"]
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    def test_fruit_detection_integration(self):
        """Test integration with fruit detection model."""
        fruiting_image = self.test_images["fruiting_plant"]

        # Detect fruits in plant image
        detection_result = self.fruit_detector.detect_fruits(fruiting_image)

        # Check detection result structure
        self.assertIsInstance(detection_result, dict)
        self.assertIn("detections", detection_result)
        self.assertIn("fruit_count", detection_result)
        self.assertIn("confidence_scores", detection_result)

        # Should detect some fruits in fruiting plant image
        self.assertGreater(detection_result["fruit_count"], 0)
        self.assertGreater(len(detection_result["detections"]), 0)

        # Validate detection format
        for detection in detection_result["detections"]:
            self.assertIn("bbox", detection)
            self.assertIn("class", detection)
            self.assertIn("confidence", detection)

            bbox = detection["bbox"]
            self.assertEqual(len(bbox), 4)  # x, y, width, height

    def test_biomass_estimation(self):
        """Test biomass estimation functionality."""
        for image_name, image in self.test_images.items():
            with self.subTest(image=image_name):
                # Estimate biomass
                biomass_result = self.plant_analyzer.estimate_biomass(image)

                # Check result structure
                self.assertIsInstance(biomass_result, dict)
                self.assertIn("estimated_biomass", biomass_result)
                self.assertIn("leaf_area", biomass_result)
                self.assertIn("plant_volume", biomass_result)
                self.assertIn("confidence", biomass_result)

                # Validate values
                self.assertGreaterEqual(biomass_result["estimated_biomass"], 0.0)
                self.assertGreaterEqual(biomass_result["leaf_area"], 0.0)
                self.assertGreaterEqual(biomass_result["plant_volume"], 0.0)

    def test_disease_detection_specificity(self):
        """Test specific disease detection capabilities."""
        diseased_image = self.test_images["diseased_plant"]

        # Perform detailed disease analysis
        disease_result = self.plant_analyzer.detect_diseases(diseased_image)

        # Check disease-specific results
        self.assertIsInstance(disease_result, dict)
        self.assertIn("detected_diseases", disease_result)
        self.assertIn("disease_severity", disease_result)
        self.assertIn("affected_area_percentage", disease_result)

        detected_diseases = disease_result["detected_diseases"]
        self.assertIsInstance(detected_diseases, list)

        # Each detected disease should have proper structure
        for disease in detected_diseases:
            self.assertIn("disease_name", disease)
            self.assertIn("confidence", disease)
            self.assertIn("affected_regions", disease)

    def test_stress_indicator_analysis(self):
        """Test stress indicator detection."""
        stressed_image = self.test_images["stressed_plant"]

        # Analyze stress indicators
        stress_result = self.plant_analyzer.analyze_stress_indicators(stressed_image)

        # Check stress analysis structure
        self.assertIsInstance(stress_result, dict)
        self.assertIn("stress_level", stress_result)
        self.assertIn("stress_types", stress_result)
        self.assertIn("stress_indicators", stress_result)

        # Validate stress level
        self.assertIn(stress_result["stress_level"], ["low", "medium", "high"])

        # Stress types should be a list
        stress_types = stress_result["stress_types"]
        self.assertIsInstance(stress_types, list)

        # Common stress types
        valid_stress_types = ["water_stress", "nutrient_deficiency", "light_stress", "temperature_stress"]
        for stress_type in stress_types:
            self.assertIn(stress_type, valid_stress_types)

    def test_temporal_analysis(self):
        """Test temporal analysis capabilities."""
        # Simulate time series data
        time_series_images = [
            self.test_images["healthy_plant"],
            self.test_images["fruiting_plant"],
            self.test_images["stressed_plant"]
        ]

        timestamps = ["2024-01-01", "2024-02-01", "2024-03-01"]

        # Perform temporal analysis
        temporal_result = self.plant_analyzer.analyze_temporal_progression(
            time_series_images, timestamps
        )

        # Check temporal analysis structure
        self.assertIsInstance(temporal_result, dict)
        self.assertIn("growth_progression", temporal_result)
        self.assertIn("health_trend", temporal_result)
        self.assertIn("biomass_change", temporal_result)
        self.assertIn("growth_rate", temporal_result)

        # Validate progression data
        growth_progression = temporal_result["growth_progression"]
        self.assertIsInstance(growth_progression, list)
        self.assertEqual(len(growth_progression), len(time_series_images))

    def test_analysis_pipeline_performance(self):
        """Test analysis pipeline performance."""
        import time

        image = self.test_images["healthy_plant"]

        # Time comprehensive analysis
        start_time = time.time()
        result = self.plant_analyzer.comprehensive_analysis(image)
        end_time = time.time()

        analysis_time = end_time - start_time

        # Analysis should complete in reasonable time
        if self.device.type == "cuda":
            self.assertLess(analysis_time, 10.0, "GPU analysis should be under 10 seconds")
        else:
            self.assertLess(analysis_time, 30.0, "CPU analysis should be under 30 seconds")

    def test_batch_analysis(self):
        """Test batch processing of multiple plant images."""
        images = list(self.test_images.values())

        # Perform batch analysis
        batch_results = self.plant_analyzer.analyze_batch(images)

        # Check batch results
        self.assertEqual(len(batch_results), len(images))

        for i, result in enumerate(batch_results):
            self.assertIsInstance(result, dict)
            self.assertIn("health_analysis", result)
            self.assertIn("growth_analysis", result)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid image formats
        invalid_inputs = [
            np.zeros((100, 100)),           # Missing channel dimension
            np.zeros((50, 50, 3)),          # Too small
            "invalid_input",                # Wrong type
            None                            # None input
        ]

        for i, invalid_input in enumerate(invalid_inputs):
            with self.subTest(invalid_input=i):
                try:
                    result = self.plant_analyzer.analyze_health(invalid_input)
                    # Should handle gracefully
                    if result is not None:
                        self.assertIsInstance(result, dict)
                        self.assertIn("error", result)
                except (ValueError, TypeError, AttributeError):
                    # Expected exceptions are acceptable
                    pass

    def test_configuration_parameters(self):
        """Test different configuration parameters."""
        image = self.test_images["healthy_plant"]

        # Test different analysis modes
        modes = ["fast", "accurate", "comprehensive"]

        for mode in modes:
            with self.subTest(mode=mode):
                result = self.plant_analyzer.analyze_health(image, mode=mode)

                self.assertIsInstance(result, dict)
                self.assertIn("health_score", result)

                # Fast mode should have lower confidence but faster processing
                if mode == "fast":
                    self.assertIn("processing_time", result)
                elif mode == "comprehensive":
                    self.assertIn("detailed_analysis", result)


class TestPlantAnalysisMetrics(unittest.TestCase):
    """Test plant analysis metrics and evaluation."""

    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        # Create mock predictions and ground truth
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]  # 3 classes
        y_pred = [0, 1, 1, 0, 1, 2, 0, 2, 2]  # Some misclassifications

        metrics = ClassificationMetrics(num_classes=3)
        metrics.update(y_pred, y_true)
        result = metrics.compute()

        # Check metrics structure
        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1_score", result)
        self.assertIn("confusion_matrix", result)

        # Validate metric ranges
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_health_score_calculation(self):
        """Test health score calculation methodology."""
        # This would test the specific algorithm for calculating health scores
        plant_analyzer = PlantAnalyzer()

        # Mock feature extraction results
        mock_features = {
            "green_ratio": 0.8,
            "brown_ratio": 0.1,
            "leaf_area": 0.6,
            "disease_spots": 2
        }

        health_score = plant_analyzer._calculate_health_score(mock_features)

        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)
