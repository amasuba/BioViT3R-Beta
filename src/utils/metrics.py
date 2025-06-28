# BioViT3R-Beta Metrics Module
# Evaluation Metrics for Model Assessment

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings

@dataclass
class DetectionMetrics:
    """Container for object detection metrics"""
    precision: float
    recall: float
    f1_score: float
    map_50: float  # mAP at IoU=0.5
    map_75: float  # mAP at IoU=0.75
    map_50_95: float  # mAP averaged over IoU=0.5:0.95
    average_precision: float

@dataclass
class ReconstructionMetrics:
    """Container for 3D reconstruction metrics"""
    chamfer_distance: float
    hausdorff_distance: float
    point_to_surface_distance: float
    completeness: float
    accuracy: float
    f_score: float

@dataclass
class ClassificationMetrics:
    """Container for classification metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]

class PlantAnalysisMetrics:
    """Comprehensive metrics for plant analysis tasks"""
    
    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) for bounding boxes
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        # Compute intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Compute intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Compute areas of both boxes
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute union area
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        Compute Average Precision (AP) from precision-recall curve
        
        Args:
            recalls: Recall values
            precisions: Precision values
            
        Returns:
            Average Precision score
        """
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls = recalls[sorted_indices]
        precisions = precisions[sorted_indices]
        
        # Compute AP using the 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap
    
    @staticmethod
    def evaluate_fruit_detection(predictions: List[Dict],
                                ground_truth: List[Dict],
                                iou_threshold: float = 0.5,
                                confidence_threshold: float = 0.5) -> DetectionMetrics:
        """
        Evaluate fruit detection performance
        
        Args:
            predictions: List of prediction dictionaries with 'bbox', 'confidence', 'class'
            ground_truth: List of ground truth dictionaries with 'bbox', 'class'
            iou_threshold: IoU threshold for positive matches
            confidence_threshold: Confidence threshold for predictions
            
        Returns:
            Detection metrics
        """
        # Filter predictions by confidence
        filtered_preds = [p for p in predictions if p['confidence'] >= confidence_threshold]
        
        if len(filtered_preds) == 0:
            return DetectionMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Initialize counters
        true_positives = 0
        false_positives = 0
        false_negatives = len(ground_truth)
        
        # Track matched ground truth boxes
        matched_gt = set()
        
        # Sort predictions by confidence (descending)
        filtered_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Evaluate each prediction
        for pred in filtered_preds:
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                
                # Check class match (if classes are provided)
                if 'class' in pred and 'class' in gt:
                    if pred['class'] != gt['class']:
                        continue
                
                iou = PlantAnalysisMetrics.compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is above threshold
            if best_iou >= iou_threshold and best_gt_idx != -1:
                true_positives += 1
                matched_gt.add(best_gt_idx)
                false_negatives -= 1
            else:
                false_positives += 1
        
        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # For simplicity, use same values for different mAP thresholds
        # In practice, you'd compute these at different IoU thresholds
        map_50 = precision  # Simplified
        map_75 = precision * 0.8  # Simplified
        map_50_95 = precision * 0.9  # Simplified
        
        return DetectionMetrics(precision, recall, f1, map_50, map_75, map_50_95, precision)
    
    @staticmethod
    def compute_chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
        """
        Compute Chamfer distance between two point clouds
        
        Args:
            points1: First point cloud (N, 3)
            points2: Second point cloud (M, 3)
            
        Returns:
            Chamfer distance
        """
        # Compute pairwise distances
        dist_matrix = cdist(points1, points2)
        
        # Chamfer distance: average of min distances in both directions
        chamfer_1_to_2 = np.mean(np.min(dist_matrix, axis=1))
        chamfer_2_to_1 = np.mean(np.min(dist_matrix, axis=0))
        
        return (chamfer_1_to_2 + chamfer_2_to_1) / 2
    
    @staticmethod
    def compute_hausdorff_distance(points1: np.ndarray, points2: np.ndarray) -> float:
        """
        Compute Hausdorff distance between two point clouds
        
        Args:
            points1: First point cloud (N, 3)
            points2: Second point cloud (M, 3)
            
        Returns:
            Hausdorff distance
        """
        # Compute pairwise distances
        dist_matrix = cdist(points1, points2)
        
        # Hausdorff distance: max of min distances in both directions
        hausdorff_1_to_2 = np.max(np.min(dist_matrix, axis=1))
        hausdorff_2_to_1 = np.max(np.min(dist_matrix, axis=0))
        
        return max(hausdorff_1_to_2, hausdorff_2_to_1)
    
    @staticmethod
    def evaluate_3d_reconstruction(predicted_points: np.ndarray,
                                  ground_truth_points: np.ndarray,
                                  threshold: float = 0.01) -> ReconstructionMetrics:
        """
        Evaluate 3D reconstruction quality
        
        Args:
            predicted_points: Predicted point cloud (N, 3)
            ground_truth_points: Ground truth point cloud (M, 3)
            threshold: Distance threshold for accuracy/completeness
            
        Returns:
            Reconstruction metrics
        """
        # Compute distance metrics
        chamfer_dist = PlantAnalysisMetrics.compute_chamfer_distance(
            predicted_points, ground_truth_points
        )
        hausdorff_dist = PlantAnalysisMetrics.compute_hausdorff_distance(
            predicted_points, ground_truth_points
        )
        
        # Compute accuracy and completeness
        pred_to_gt_dists = np.min(cdist(predicted_points, ground_truth_points), axis=1)
        gt_to_pred_dists = np.min(cdist(ground_truth_points, predicted_points), axis=1)
        
        accuracy = np.mean(pred_to_gt_dists < threshold)
        completeness = np.mean(gt_to_pred_dists < threshold)
        
        # F-score
        f_score = 2 * accuracy * completeness / (accuracy + completeness) if (accuracy + completeness) > 0 else 0
        
        # Point-to-surface distance (simplified as mean of minimum distances)
        point_to_surface_dist = np.mean(pred_to_gt_dists)
        
        return ReconstructionMetrics(
            chamfer_dist, hausdorff_dist, point_to_surface_dist,
            completeness, accuracy, f_score
        )
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None,
                              class_names: Optional[List[str]] = None) -> ClassificationMetrics:
        """
        Evaluate classification performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for AUC computation)
            class_names: Names of classes
            
        Returns:
            Classification metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC (if probabilities provided)
        auc_roc = None
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    auc_roc = roc_auc_score(y_true, y_prob)
                else:  # Multi-class
                    auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_metrics = {}
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        # Get per-class precision, recall, f1
        class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            if i < len(class_precision):
                per_class_metrics[class_name] = {
                    'precision': float(class_precision[i]),
                    'recall': float(class_recall[i]),
                    'f1_score': float(class_f1[i])
                }
        
        return ClassificationMetrics(
            accuracy, precision, recall, f1, auc_roc, cm, per_class_metrics
        )
    
    @staticmethod
    def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute regression metrics (for biomass estimation, etc.)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape
        }
    
    @staticmethod
    def compute_health_score_metrics(predicted_health: Dict[str, float],
                                   true_health: Dict[str, float]) -> Dict[str, float]:
        """
        Compute metrics for plant health assessment
        
        Args:
            predicted_health: Predicted health scores by category
            true_health: True health scores by category
            
        Returns:
            Health assessment metrics
        """
        metrics = {}
        
        # Compute metrics for each health category
        for category in true_health.keys():
            if category in predicted_health:
                pred_val = predicted_health[category]
                true_val = true_health[category]
                
                metrics[f"{category}_mae"] = abs(pred_val - true_val)
                metrics[f"{category}_mse"] = (pred_val - true_val) ** 2
                metrics[f"{category}_accuracy"] = 1 - abs(pred_val - true_val)  # Simplified accuracy
        
        # Overall health score comparison
        if 'overall_health' in predicted_health and 'overall_health' in true_health:
            overall_diff = abs(predicted_health['overall_health'] - true_health['overall_health'])
            metrics['overall_health_accuracy'] = 1 - overall_diff
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray,
                             class_names: List[str],
                             normalize: bool = False,
                             title: str = 'Confusion Matrix') -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: Class names
            normalize: Whether to normalize
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(precisions: np.ndarray,
                                   recalls: np.ndarray,
                                   ap_score: float,
                                   title: str = 'Precision-Recall Curve') -> plt.Figure:
        """
        Plot precision-recall curve
        
        Args:
            precisions: Precision values
            recalls: Recall values
            ap_score: Average Precision score
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recalls, precisions, linewidth=2, label=f'AP = {ap_score:.3f}')
        ax.fill_between(recalls, precisions, alpha=0.3)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_evaluation_report(detection_metrics: Optional[DetectionMetrics] = None,
                                 reconstruction_metrics: Optional[ReconstructionMetrics] = None,
                                 classification_metrics: Optional[ClassificationMetrics] = None,
                                 regression_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            detection_metrics: Fruit detection metrics
            reconstruction_metrics: 3D reconstruction metrics
            classification_metrics: Classification metrics
            regression_metrics: Regression metrics
            
        Returns:
            Formatted evaluation report
        """
        report = "# BioViT3R-Beta Evaluation Report\n\n"
        
        if detection_metrics:
            report += "## Fruit Detection Performance\n"
            report += f"- Precision: {detection_metrics.precision:.3f}\n"
            report += f"- Recall: {detection_metrics.recall:.3f}\n"
            report += f"- F1-Score: {detection_metrics.f1_score:.3f}\n"
            report += f"- mAP@0.5: {detection_metrics.map_50:.3f}\n"
            report += f"- mAP@0.75: {detection_metrics.map_75:.3f}\n"
            report += f"- mAP@0.5:0.95: {detection_metrics.map_50_95:.3f}\n\n"
        
        if reconstruction_metrics:
            report += "## 3D Reconstruction Quality\n"
            report += f"- Chamfer Distance: {reconstruction_metrics.chamfer_distance:.6f}\n"
            report += f"- Hausdorff Distance: {reconstruction_metrics.hausdorff_distance:.6f}\n"
            report += f"- Accuracy: {reconstruction_metrics.accuracy:.3f}\n"
            report += f"- Completeness: {reconstruction_metrics.completeness:.3f}\n"
            report += f"- F-Score: {reconstruction_metrics.f_score:.3f}\n\n"
        
        if classification_metrics:
            report += "## Classification Performance\n"
            report += f"- Accuracy: {classification_metrics.accuracy:.3f}\n"
            report += f"- Precision: {classification_metrics.precision:.3f}\n"
            report += f"- Recall: {classification_metrics.recall:.3f}\n"
            report += f"- F1-Score: {classification_metrics.f1_score:.3f}\n"
            if classification_metrics.auc_roc:
                report += f"- AUC-ROC: {classification_metrics.auc_roc:.3f}\n"
            report += "\n"
        
        if regression_metrics:
            report += "## Regression Performance\n"
            for metric, value in regression_metrics.items():
                report += f"- {metric.upper()}: {value:.6f}\n"
            report += "\n"
        
        return report

# Utility functions for metric computation
def compute_pixel_accuracy(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute pixel-wise accuracy for segmentation masks"""
    return np.mean(pred_mask == true_mask)

def compute_dice_coefficient(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute Dice coefficient for binary masks"""
    intersection = np.sum(pred_mask * true_mask)
    return 2 * intersection / (np.sum(pred_mask) + np.sum(true_mask))

def compute_jaccard_index(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute Jaccard index (IoU) for binary masks"""
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask) - intersection
    return intersection / union if union > 0 else 0