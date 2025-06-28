# BioViT3R-Beta Visualization Utilities
# 3D Point Cloud and Analysis Visualization Module

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
import open3d as o3d
from typing import List, Dict, Tuple, Optional, Any
import seaborn as sns
from dataclasses import dataclass
import io
import base64

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    point_size: float = 2.0
    mesh_opacity: float = 0.8
    colormap: str = 'viridis'
    figure_size: Tuple[int, int] = (1200, 800)
    background_color: str = 'white'
    grid_visible: bool = True

class PlantVisualization:
    """Advanced plant analysis visualization toolkit"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.setup_style()
    
    def setup_style(self):
        """Setup matplotlib and seaborn styling"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def visualize_3d_reconstruction(self, 
                                   points: np.ndarray, 
                                   colors: Optional[np.ndarray] = None,
                                   mesh: Optional[Any] = None,
                                   title: str = "3D Plant Reconstruction") -> go.Figure:
        """
        Create interactive 3D visualization of reconstructed plant
        
        Args:
            points: Point cloud coordinates (N, 3)
            colors: Point colors (N, 3) or None
            mesh: Optional mesh object
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add point cloud
        if colors is None:
            colors = points[:, 2]  # Use Z-coordinate for coloring
            
        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1], 
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=self.config.point_size,
                color=colors,
                colorscale=self.config.colormap,
                opacity=0.8,
                colorbar=dict(title="Height (mm)")
            ),
            name="Point Cloud",
            text=[f"Point {i}" for i in range(len(points))],
            hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
        )
        fig.add_trace(scatter)
        
        # Add mesh if provided
        if mesh is not None:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            mesh_trace = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                opacity=self.config.mesh_opacity,
                color='lightblue',
                name="3D Mesh"
            )
            fig.add_trace(mesh_trace)
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)", 
                zaxis_title="Z (mm)",
                bgcolor=self.config.background_color,
                xaxis=dict(showgrid=self.config.grid_visible),
                yaxis=dict(showgrid=self.config.grid_visible),
                zaxis=dict(showgrid=self.config.grid_visible)
            ),
            width=self.config.figure_size[0],
            height=self.config.figure_size[1]
        )
        
        return fig
    
    def visualize_fruit_detection(self, 
                                  image: np.ndarray,
                                  detections: List[Dict],
                                  confidence_threshold: float = 0.5) -> plt.Figure:
        """
        Visualize fruit detection results on image
        
        Args:
            image: Input image (H, W, 3)
            detections: List of detection dictionaries
            confidence_threshold: Minimum confidence to display
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display image
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Draw detection boxes
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        for i, detection in enumerate(detections):
            if detection['confidence'] < confidence_threshold:
                continue
                
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            color = colors[i % len(colors)]
            
            # Draw bounding box
            rect = Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                linewidth=3, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{detection.get('class', 'fruit')}: {detection['confidence']:.2f}"
            ax.text(
                bbox[0], bbox[1] - 10, 
                label, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, 
                color='white', 
                weight='bold'
            )
        
        ax.set_title(f"Fruit Detection Results ({len([d for d in detections if d['confidence'] >= confidence_threshold])} fruits detected)")
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_health_assessment(self, 
                                   health_scores: Dict[str, float],
                                   disease_probabilities: Optional[Dict[str, float]] = None) -> go.Figure:
        """
        Create health assessment dashboard visualization
        
        Args:
            health_scores: Dictionary of health metrics
            disease_probabilities: Disease probability scores
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Overall Health", "Disease Risk", "Growth Metrics", "Health Timeline"),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Overall health gauge
        overall_health = health_scores.get('overall_health', 0.8)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_health * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Health Score (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Disease probabilities
        if disease_probabilities:
            diseases = list(disease_probabilities.keys())
            probs = list(disease_probabilities.values())
            
            fig.add_trace(
                go.Bar(
                    x=diseases,
                    y=probs,
                    marker_color=['red' if p > 0.5 else 'orange' if p > 0.3 else 'green' for p in probs],
                    name="Disease Risk"
                ),
                row=1, col=2
            )
        
        # Growth metrics
        growth_metrics = {k: v for k, v in health_scores.items() if k != 'overall_health'}
        if growth_metrics:
            fig.add_trace(
                go.Bar(
                    x=list(growth_metrics.keys()),
                    y=list(growth_metrics.values()),
                    marker_color='lightblue',
                    name="Growth Metrics"
                ),
                row=2, col=1
            )
        
        # Mock timeline data
        timeline_days = list(range(1, 15))
        timeline_health = [overall_health + np.random.normal(0, 0.05) for _ in timeline_days]
        
        fig.add_trace(
            go.Scatter(
                x=timeline_days,
                y=timeline_health,
                mode='lines+markers',
                name="Health Trend",
                line=dict(color='blue', width=3)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Plant Health Assessment Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def visualize_growth_stages(self, 
                               stage_predictions: Dict[str, float],
                               confidence_scores: Dict[str, float]) -> go.Figure:
        """
        Visualize growth stage classification results
        
        Args:
            stage_predictions: Predicted growth stages with probabilities
            confidence_scores: Confidence scores for each prediction
            
        Returns:
            Plotly figure
        """
        stages = list(stage_predictions.keys())
        probabilities = list(stage_predictions.values())
        confidences = list(confidence_scores.values())
        
        # Create combined bar chart
        fig = go.Figure()
        
        # Probability bars
        fig.add_trace(
            go.Bar(
                x=stages,
                y=probabilities,
                name="Probability",
                marker_color='lightblue',
                yaxis='y',
                offsetgroup=1
            )
        )
        
        # Confidence bars
        fig.add_trace(
            go.Bar(
                x=stages,
                y=confidences,
                name="Confidence",
                marker_color='orange',
                yaxis='y2',
                offsetgroup=2
            )
        )
        
        # Update layout for dual y-axis
        fig.update_layout(
            title="Growth Stage Classification Results",
            xaxis_title="Growth Stages",
            yaxis=dict(
                title="Probability",
                side="left"
            ),
            yaxis2=dict(
                title="Confidence Score",
                side="right",
                overlaying="y"
            ),
            barmode='group',
            height=600
        )
        
        return fig
    
    def create_analysis_report(self, 
                              analysis_results: Dict[str, Any],
                              save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive HTML analysis report
        
        Args:
            analysis_results: Complete analysis results dictionary
            save_path: Optional path to save HTML report
            
        Returns:
            HTML content as string
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BioViT3R-Beta Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .highlight {{ background-color: #ffeb3b; padding: 2px 4px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BioViT3R-Beta Plant Analysis Report</h1>
                <p>Generated on: {plt.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>🌱 Overall Health Assessment</h2>
                <div class="metric">
                    <strong>Overall Health Score:</strong> 
                    <span class="highlight">{analysis_results.get('health_score', 'N/A')}</span>
                </div>
                <div class="metric">
                    <strong>Growth Stage:</strong> 
                    <span class="highlight">{analysis_results.get('growth_stage', 'N/A')}</span>
                </div>
                <div class="metric">
                    <strong>Fruit Count:</strong> 
                    <span class="highlight">{analysis_results.get('fruit_count', 'N/A')}</span>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Detailed Metrics</h2>
                <p>Biomass Estimate: {analysis_results.get('biomass_estimate', 'N/A')} kg</p>
                <p>Disease Risk: {analysis_results.get('disease_risk', 'N/A')}</p>
                <p>Recommended Actions: {analysis_results.get('recommendations', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>🔍 Technical Details</h2>
                <p>Processing Time: {analysis_results.get('processing_time', 'N/A')} seconds</p>
                <p>Image Resolution: {analysis_results.get('image_resolution', 'N/A')}</p>
                <p>3D Points Generated: {analysis_results.get('point_count', 'N/A')}</p>
            </div>
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html_content)
                
        return html_content

    def plot_comparative_analysis(self, 
                                 datasets: Dict[str, Dict],
                                 metric: str = 'health_score') -> go.Figure:
        """
        Create comparative analysis across multiple plants or time periods
        
        Args:
            datasets: Dictionary of dataset names to metrics
            metric: Specific metric to compare
            
        Returns:
            Plotly figure
        """
        names = list(datasets.keys())
        values = [data.get(metric, 0) for data in datasets.values()]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=values,
                marker_color=px.colors.qualitative.Set3,
                text=[f"{v:.2f}" for v in values],
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title=f"Comparative Analysis: {metric.replace('_', ' ').title()}",
            xaxis_title="Dataset/Plant ID",
            yaxis_title=metric.replace('_', ' ').title(),
            height=500
        )
        
        return fig

# Utility functions for quick visualization
def quick_3d_plot(points: np.ndarray, title: str = "3D Visualization") -> go.Figure:
    """Quick 3D point cloud visualization"""
    viz = PlantVisualization()
    return viz.visualize_3d_reconstruction(points, title=title)

def quick_detection_plot(image: np.ndarray, detections: List[Dict]) -> plt.Figure:
    """Quick fruit detection visualization"""
    viz = PlantVisualization()
    return viz.visualize_fruit_detection(image, detections)

def save_figure_base64(fig) -> str:
    """Convert matplotlib or plotly figure to base64 string"""
    if hasattr(fig, 'to_html'):  # Plotly figure
        return fig.to_html(include_plotlyjs='cdn')
    else:  # Matplotlib figure
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return f"data:image/png;base64,{img_base64}"