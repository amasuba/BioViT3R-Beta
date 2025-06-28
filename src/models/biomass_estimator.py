# BioViT3R-Beta Biomass Estimation Module
# Volumetric Biomass Estimation from 3D Point Clouds

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
from dataclasses import dataclass
import json
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class AllometricParameters:
    """Container for allometric relationship parameters"""
    species: str
    a_coefficient: float  # Scaling coefficient
    b_exponent: float     # Allometric exponent
    r_squared: float      # Model fit quality
    sample_size: int      # Training sample size
    height_range: Tuple[float, float]  # Valid height range (cm)
    
@dataclass
class BiomassEstimate:
    """Container for biomass estimation results"""
    total_biomass: float           # Total estimated biomass (kg)
    above_ground_biomass: float    # Above-ground biomass (kg)
    leaf_biomass: float           # Leaf biomass estimate (kg)
    stem_biomass: float           # Stem biomass estimate (kg)
    fruit_biomass: float          # Fruit biomass estimate (kg)
    confidence_score: float       # Estimation confidence (0-1)
    method_used: str              # Estimation method
    volume_m3: float              # Plant volume (cubic meters)
    height_m: float               # Plant height (meters)

class BiomassEstimator:
    """Advanced biomass estimation from 3D plant reconstructions"""
    
    def __init__(self, 
                 species_config_path: Optional[str] = None,
                 default_species: str = "generic"):
        """
        Initialize biomass estimator
        
        Args:
            species_config_path: Path to species-specific parameters
            default_species: Default species to use
        """
        self.default_species = default_species
        self.allometric_params = self.load_allometric_parameters(species_config_path)
        self.density_factors = self.get_density_factors()
        
    def load_allometric_parameters(self, config_path: Optional[str]) -> Dict[str, AllometricParameters]:
        """Load species-specific allometric parameters"""
        
        # Default parameters from literature
        default_params = {
            "generic": AllometricParameters(
                species="generic",
                a_coefficient=0.0673,
                b_exponent=2.176,
                r_squared=0.85,
                sample_size=150,
                height_range=(10, 500)
            ),
            "apple": AllometricParameters(
                species="apple",
                a_coefficient=0.083,
                b_exponent=2.31,
                r_squared=0.91,
                sample_size=87,
                height_range=(50, 800)
            ),
            "citrus": AllometricParameters(
                species="citrus", 
                a_coefficient=0.074,
                b_exponent=2.24,
                r_squared=0.88,
                sample_size=65,
                height_range=(30, 600)
            ),
            "tomato": AllometricParameters(
                species="tomato",
                a_coefficient=0.125,
                b_exponent=1.89,
                r_squared=0.82,
                sample_size=120,
                height_range=(20, 300)
            ),
            "grape": AllometricParameters(
                species="grape",
                a_coefficient=0.095,
                b_exponent=2.05,
                r_squared=0.86,
                sample_size=98,
                height_range=(40, 400)
            )
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    custom_params = json.load(f)
                
                # Override defaults with custom parameters
                for species, params in custom_params.items():
                    default_params[species] = AllometricParameters(**params)
                    
            except Exception as e:
                logger.warning(f"Failed to load custom parameters: {e}")
        
        return default_params
    
    def get_density_factors(self) -> Dict[str, Dict[str, float]]:
        """Get tissue density factors for different plant parts"""
        return {
            "generic": {
                "leaf": 0.3,      # kg/m³ effective density
                "stem": 0.8,      # kg/m³ effective density  
                "fruit": 0.9,     # kg/m³ effective density
                "branch": 0.6     # kg/m³ effective density
            },
            "apple": {
                "leaf": 0.25,
                "stem": 0.85,
                "fruit": 0.95,
                "branch": 0.65
            },
            "citrus": {
                "leaf": 0.28,
                "stem": 0.82,
                "fruit": 0.92,
                "branch": 0.62
            },
            "tomato": {
                "leaf": 0.35,
                "stem": 0.45,
                "fruit": 0.88,
                "branch": 0.50
            }
        }
    
    def estimate_biomass(self, 
                        point_cloud: np.ndarray,
                        species: Optional[str] = None,
                        fruit_detections: Optional[List[Dict]] = None,
                        method: str = "volumetric") -> BiomassEstimate:
        """
        Estimate plant biomass from 3D point cloud
        
        Args:
            point_cloud: 3D point coordinates (N, 3)
            species: Plant species identifier
            fruit_detections: Optional fruit detection results
            method: Estimation method ('volumetric', 'allometric', 'hybrid')
            
        Returns:
            Biomass estimation results
        """
        if species is None:
            species = self.default_species
        
        # Get species parameters
        if species not in self.allometric_params:
            logger.warning(f"Species {species} not found, using generic parameters")
            species = "generic"
        
        params = self.allometric_params[species]
        density_factors = self.density_factors.get(species, self.density_factors["generic"])
        
        # Compute basic geometric properties
        volume = self.compute_plant_volume(point_cloud)
        height = self.compute_plant_height(point_cloud)
        canopy_metrics = self.compute_canopy_metrics(point_cloud)
        
        # Select estimation method
        if method == "volumetric":
            biomass_est = self.volumetric_estimation(
                volume, height, density_factors, canopy_metrics
            )
        elif method == "allometric":
            biomass_est = self.allometric_estimation(
                height, params, point_cloud
            )
        elif method == "hybrid":
            vol_est = self.volumetric_estimation(
                volume, height, density_factors, canopy_metrics
            )
            allo_est = self.allometric_estimation(
                height, params, point_cloud
            )
            biomass_est = self.combine_estimates(vol_est, allo_est)
        else:
            raise ValueError(f"Unknown estimation method: {method}")
        
        # Add fruit biomass if detections provided
        if fruit_detections:
            fruit_biomass = self.estimate_fruit_biomass(fruit_detections, species)
            biomass_est.fruit_biomass = fruit_biomass
            biomass_est.total_biomass += fruit_biomass
        
        # Set additional properties
        biomass_est.volume_m3 = volume
        biomass_est.height_m = height
        biomass_est.method_used = method
        
        return biomass_est
    
    def compute_plant_volume(self, points: np.ndarray) -> float:
        """
        Compute plant volume using convex hull
        
        Args:
            points: Point cloud coordinates (N, 3)
            
        Returns:
            Volume in cubic meters
        """
        try:
            if len(points) < 4:
                return 0.0
            
            hull = ConvexHull(points)
            
            # Convert from mm³ to m³ (assuming points are in mm)
            volume_m3 = hull.volume / (1000**3)
            
            return volume_m3
        except:
            # Fallback: bounding box volume
            bbox_size = np.ptp(points, axis=0)  # max - min for each dimension
            volume_m3 = np.prod(bbox_size) / (1000**3)
            return volume_m3 * 0.6  # Apply occupancy factor
    
    def compute_plant_height(self, points: np.ndarray) -> float:
        """
        Compute plant height
        
        Args:
            points: Point cloud coordinates (N, 3)
            
        Returns:
            Height in meters
        """
        z_coords = points[:, 2]
        height_mm = np.percentile(z_coords, 95) - np.percentile(z_coords, 5)
        return height_mm / 1000  # Convert to meters
    
    def compute_canopy_metrics(self, points: np.ndarray) -> Dict[str, float]:
        """Compute canopy structure metrics"""
        z_coords = points[:, 2]
        xy_coords = points[:, :2]
        
        # Vertical distribution
        z_range = np.ptp(z_coords)
        z_mean = np.mean(z_coords)
        z_std = np.std(z_coords)
        
        # Horizontal spread
        xy_range = np.ptp(xy_coords, axis=0)
        canopy_diameter = np.mean(xy_range) / 1000  # Convert to meters
        
        # Density metrics
        point_density = len(points) / (np.prod(np.ptp(points, axis=0)) / (1000**3))
        
        return {
            'canopy_diameter': canopy_diameter,
            'height_variability': z_std / z_range if z_range > 0 else 0,
            'point_density': point_density,
            'aspect_ratio': canopy_diameter / (z_range / 1000) if z_range > 0 else 0
        }
    
    def volumetric_estimation(self, 
                            volume: float,
                            height: float,
                            density_factors: Dict[str, float],
                            canopy_metrics: Dict[str, float]) -> BiomassEstimate:
        """
        Estimate biomass using volumetric approach
        
        Args:
            volume: Plant volume (m³)
            height: Plant height (m)
            density_factors: Tissue density factors
            canopy_metrics: Canopy structure metrics
            
        Returns:
            Biomass estimate
        """
        # Estimate tissue volumes based on plant structure
        canopy_fraction = min(0.8, canopy_metrics.get('aspect_ratio', 0.5))
        
        # Leaf volume (upper canopy)
        leaf_volume = volume * canopy_fraction * 0.6
        
        # Stem volume (structural support)
        stem_volume = volume * 0.2 * (height / 2.0)  # Scales with height
        
        # Branch volume
        branch_volume = volume * (1 - canopy_fraction) * 0.4
        
        # Compute biomass for each tissue type
        leaf_biomass = leaf_volume * density_factors["leaf"]
        stem_biomass = stem_volume * density_factors["stem"] 
        branch_biomass = branch_volume * density_factors["branch"]
        
        total_biomass = leaf_biomass + stem_biomass + branch_biomass
        above_ground_biomass = total_biomass  # Assuming no root estimation
        
        # Confidence based on point density and structure
        confidence = min(1.0, canopy_metrics.get('point_density', 0) / 1000 + 0.3)
        
        return BiomassEstimate(
            total_biomass=total_biomass,
            above_ground_biomass=above_ground_biomass,
            leaf_biomass=leaf_biomass,
            stem_biomass=stem_biomass,
            fruit_biomass=0.0,
            confidence_score=confidence,
            method_used="volumetric",
            volume_m3=volume,
            height_m=height
        )
    
    def allometric_estimation(self,
                            height: float,
                            params: AllometricParameters,
                            point_cloud: np.ndarray) -> BiomassEstimate:
        """
        Estimate biomass using allometric relationships
        
        Args:
            height: Plant height (m)
            params: Allometric parameters
            point_cloud: Point cloud for additional metrics
            
        Returns:
            Biomass estimate
        """
        # Convert height to cm for allometric equation
        height_cm = height * 100
        
        # Check if height is within valid range
        height_range_cm = params.height_range
        if height_cm < height_range_cm[0] or height_cm > height_range_cm[1]:
            logger.warning(f"Height {height_cm}cm outside valid range {height_range_cm}")
        
        # Apply allometric equation: Biomass = a * Height^b
        total_biomass = params.a_coefficient * (height_cm ** params.b_exponent)
        
        # Partition biomass into components (species-specific ratios)
        if params.species == "tomato":
            leaf_fraction = 0.4
            stem_fraction = 0.35
        elif params.species in ["apple", "citrus"]:
            leaf_fraction = 0.3
            stem_fraction = 0.5
        else:  # generic
            leaf_fraction = 0.35
            stem_fraction = 0.45
        
        leaf_biomass = total_biomass * leaf_fraction
        stem_biomass = total_biomass * stem_fraction
        
        # Confidence based on model R² and height validity
        confidence = params.r_squared
        if height_cm < height_range_cm[0] or height_cm > height_range_cm[1]:
            confidence *= 0.7  # Reduce confidence for extrapolation
        
        return BiomassEstimate(
            total_biomass=total_biomass,
            above_ground_biomass=total_biomass,
            leaf_biomass=leaf_biomass,
            stem_biomass=stem_biomass,
            fruit_biomass=0.0,
            confidence_score=confidence,
            method_used="allometric",
            volume_m3=0.0,  # Not computed in this method
            height_m=height
        )
    
    def combine_estimates(self, 
                         vol_est: BiomassEstimate,
                         allo_est: BiomassEstimate) -> BiomassEstimate:
        """
        Combine volumetric and allometric estimates
        
        Args:
            vol_est: Volumetric estimate
            allo_est: Allometric estimate
            
        Returns:
            Combined estimate
        """
        # Weight estimates by their confidence scores
        vol_weight = vol_est.confidence_score
        allo_weight = allo_est.confidence_score
        total_weight = vol_weight + allo_weight
        
        if total_weight == 0:
            return vol_est  # Fallback
        
        vol_w = vol_weight / total_weight
        allo_w = allo_weight / total_weight
        
        # Weighted combination
        combined_total = vol_w * vol_est.total_biomass + allo_w * allo_est.total_biomass
        combined_leaf = vol_w * vol_est.leaf_biomass + allo_w * allo_est.leaf_biomass
        combined_stem = vol_w * vol_est.stem_biomass + allo_w * allo_est.stem_biomass
        combined_confidence = (vol_est.confidence_score + allo_est.confidence_score) / 2
        
        return BiomassEstimate(
            total_biomass=combined_total,
            above_ground_biomass=combined_total,
            leaf_biomass=combined_leaf,
            stem_biomass=combined_stem,
            fruit_biomass=0.0,
            confidence_score=combined_confidence,
            method_used="hybrid",
            volume_m3=vol_est.volume_m3,
            height_m=max(vol_est.height_m, allo_est.height_m)
        )
    
    def estimate_fruit_biomass(self, 
                              detections: List[Dict],
                              species: str) -> float:
        """
        Estimate fruit biomass from detection results
        
        Args:
            detections: Fruit detection results
            species: Plant species
            
        Returns:
            Estimated fruit biomass (kg)
        """
        if not detections:
            return 0.0
        
        # Species-specific fruit mass estimates (kg per fruit)
        fruit_masses = {
            "apple": 0.18,      # Average apple ~180g
            "citrus": 0.25,     # Average orange ~250g
            "tomato": 0.12,     # Average tomato ~120g
            "grape": 0.005,     # Average grape ~5g
            "generic": 0.15     # Generic fruit mass
        }
        
        avg_fruit_mass = fruit_masses.get(species, fruit_masses["generic"])
        
        # Count high-confidence detections
        high_conf_detections = [d for d in detections if d.get('confidence', 0) > 0.7]
        fruit_count = len(high_conf_detections)
        
        # Estimate size variation from bounding boxes
        if high_conf_detections:
            bbox_areas = []
            for detection in high_conf_detections:
                bbox = detection.get('bbox', [0, 0, 100, 100])
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                bbox_areas.append(area)
            
            # Adjust mass based on relative size
            mean_area = np.mean(bbox_areas)
            size_factors = [area / mean_area for area in bbox_areas]
            
            total_mass = sum(avg_fruit_mass * factor for factor in size_factors)
        else:
            total_mass = fruit_count * avg_fruit_mass
        
        return total_mass
    
    def validate_estimate(self, estimate: BiomassEstimate, species: str) -> Dict[str, Any]:
        """
        Validate biomass estimate against known ranges
        
        Args:
            estimate: Biomass estimate to validate
            species: Plant species
            
        Returns:
            Validation results
        """
        # Typical biomass ranges by species (kg)
        biomass_ranges = {
            "apple": {"min": 5, "max": 150, "typical": 25},
            "citrus": {"min": 3, "max": 80, "typical": 18},
            "tomato": {"min": 0.5, "max": 8, "typical": 2.5},
            "grape": {"min": 1, "max": 30, "typical": 8},
            "generic": {"min": 0.5, "max": 100, "typical": 15}
        }
        
        range_info = biomass_ranges.get(species, biomass_ranges["generic"])
        
        validation = {
            "within_expected_range": range_info["min"] <= estimate.total_biomass <= range_info["max"],
            "relative_to_typical": estimate.total_biomass / range_info["typical"],
            "confidence_category": (
                "High" if estimate.confidence_score > 0.8 else
                "Medium" if estimate.confidence_score > 0.6 else
                "Low"
            ),
            "warnings": []
        }
        
        # Add warnings for unusual estimates
        if estimate.total_biomass < range_info["min"]:
            validation["warnings"].append("Estimated biomass below typical minimum")
        elif estimate.total_biomass > range_info["max"]:
            validation["warnings"].append("Estimated biomass above typical maximum")
        
        if estimate.confidence_score < 0.5:
            validation["warnings"].append("Low confidence estimate")
        
        return validation

# Utility functions
def load_custom_allometric_model(species: str, training_data: Dict[str, List[float]]) -> AllometricParameters:
    """
    Fit custom allometric model from training data
    
    Args:
        species: Species name
        training_data: Dict with 'heights' and 'biomass' lists
        
    Returns:
        Fitted allometric parameters
    """
    heights = np.array(training_data['heights'])
    biomass = np.array(training_data['biomass'])
    
    # Fit allometric equation: y = a * x^b
    def allometric_func(x, a, b):
        return a * (x ** b)
    
    try:
        popt, pcov = curve_fit(allometric_func, heights, biomass)
        a, b = popt
        
        # Calculate R²
        y_pred = allometric_func(heights, a, b)
        ss_res = np.sum((biomass - y_pred) ** 2)
        ss_tot = np.sum((biomass - np.mean(biomass)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return AllometricParameters(
            species=species,
            a_coefficient=a,
            b_exponent=b,
            r_squared=r_squared,
            sample_size=len(heights),
            height_range=(float(np.min(heights)), float(np.max(heights)))
        )
        
    except Exception as e:
        logger.error(f"Failed to fit allometric model: {e}")
        raise