# BioViT3R-Beta Main Application
# Gradio-based Interface for Plant Analysis with AI Assistant

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any, Optional, List, Tuple
import yaml
import json
from datetime import datetime
import os
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import BioViT3R modules
from src.models.vggt_reconstructor import VGGTReconstructor
from src.models.plant_analyzer import PlantAnalyzer
from src.models.fruit_detector import FruitDetector
from src.data.preprocessing import ImagePreprocessor
from src.utils.visualization import PlantVisualization
from src.utils.file_utils import FileUtils
from src.ai_assistant.chat_interface import ChatInterface
from src.ai_assistant.context_manager import ContextManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BioViT3RApp:
    """Main BioViT3R-Beta application with Gradio interface"""
    
    def __init__(self, config_path: str = "configs/app_config.yaml"):
        """Initialize the application with configuration"""
        self.config = self.load_config(config_path)
        self.setup_models()
        self.setup_components()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load application configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'models': {
                'vggt': {'model_path': 'models/vggt/', 'device': 'cuda'},
                'fruit_detector': {'model_path': 'models/fruit_detection/', 'confidence_threshold': 0.5},
                'health_classifier': {'model_path': 'models/health_classification/'}
            },
            'ui': {
                'theme': 'soft',
                'title': 'BioViT3R-Beta: AI-Powered Plant Analysis',
                'description': 'Advanced plant analysis with 3D reconstruction and AI assistance'
            },
            'processing': {
                'max_image_size': 1024,
                'temp_dir': 'temp/',
                'output_dir': 'outputs/'
            }
        }
    
    def setup_models(self):
        """Initialize all analysis models"""
        logger.info("Initializing models...")
        
        try:
            # Initialize VGGT reconstructor
            self.vggt_reconstructor = VGGTReconstructor(
                model_path=self.config['models']['vggt']['model_path'],
                device=self.config['models']['vggt']['device']
            )
            
            # Initialize plant analyzer
            self.plant_analyzer = PlantAnalyzer(
                health_model_path=self.config['models']['health_classifier']['model_path']
            )
            
            # Initialize fruit detector
            self.fruit_detector = FruitDetector(
                model_path=self.config['models']['fruit_detector']['model_path'],
                confidence_threshold=self.config['models']['fruit_detector']['confidence_threshold']
            )
            
            # Initialize preprocessor
            self.preprocessor = ImagePreprocessor()
            
            # Initialize visualization
            self.visualizer = PlantVisualization()
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def setup_components(self):
        """Setup application components"""
        # Initialize context manager for AI assistant
        self.context_manager = ContextManager()
        
        # Initialize chat interface
        self.chat_interface = ChatInterface(self.context_manager)
        
        # Setup directories
        os.makedirs(self.config['processing']['temp_dir'], exist_ok=True)
        os.makedirs(self.config['processing']['output_dir'], exist_ok=True)
    
    def process_plant_image(self, 
                           image: np.ndarray,
                           analysis_options: List[str],
                           progress=gr.Progress()) -> Tuple[Dict[str, Any], str, str, str]:
        """
        Main plant analysis pipeline
        
        Args:
            image: Input image array
            analysis_options: Selected analysis options
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (results_dict, 3d_plot_html, detection_image_path, summary_text)
        """
        if image is None:
            return {}, "", "", "No image provided"
        
        progress(0.1, desc="Preprocessing image...")
        
        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(
            image, target_size=self.config['processing']['max_image_size']
        )
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'image_shape': image.shape,
            'analysis_options': analysis_options
        }
        
        # Initialize outputs
        plot_html = ""
        detection_image_path = ""
        summary_parts = []
        
        try:
            # 3D Reconstruction
            if "3d_reconstruction" in analysis_options:
                progress(0.2, desc="Performing 3D reconstruction...")
                
                point_cloud = self.vggt_reconstructor.reconstruct(processed_image)
                mesh = self.vggt_reconstructor.generate_mesh(point_cloud)
                
                results['3d_reconstruction'] = {
                    'point_count': len(point_cloud),
                    'has_mesh': mesh is not None
                }
                
                # Create 3D visualization
                fig_3d = self.visualizer.visualize_3d_reconstruction(
                    point_cloud, mesh=mesh, title="3D Plant Reconstruction"
                )
                plot_html = fig_3d.to_html(include_plotlyjs='cdn')
                
                summary_parts.append(f"✓ 3D Reconstruction: {len(point_cloud)} points generated")
            
            # Fruit Detection
            if "fruit_detection" in analysis_options:
                progress(0.4, desc="Detecting fruits...")
                
                detections = self.fruit_detector.detect(processed_image)
                results['fruit_detection'] = {
                    'fruit_count': len(detections),
                    'detections': detections
                }
                
                # Create detection visualization
                fig_detection = self.visualizer.visualize_fruit_detection(
                    processed_image, detections
                )
                
                # Save detection image
                detection_image_path = os.path.join(
                    self.config['processing']['temp_dir'], 
                    f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                fig_detection.savefig(detection_image_path, dpi=150, bbox_inches='tight')
                
                summary_parts.append(f"✓ Fruit Detection: {len(detections)} fruits detected")
            
            # Health Analysis
            if "health_analysis" in analysis_options:
                progress(0.6, desc="Analyzing plant health...")
                
                health_results = self.plant_analyzer.analyze_health(processed_image)
                results['health_analysis'] = health_results
                
                overall_health = health_results.get('overall_health', 0)
                health_status = "Excellent" if overall_health > 0.8 else "Good" if overall_health > 0.6 else "Fair" if overall_health > 0.4 else "Poor"
                
                summary_parts.append(f"✓ Health Analysis: {health_status} ({overall_health:.1%})")
            
            # Growth Stage Classification
            if "growth_analysis" in analysis_options:
                progress(0.8, desc="Classifying growth stage...")
                
                growth_results = self.plant_analyzer.classify_growth_stage(processed_image)
                results['growth_analysis'] = growth_results
                
                predicted_stage = max(growth_results.items(), key=lambda x: x[1])[0]
                confidence = growth_results[predicted_stage]
                
                summary_parts.append(f"✓ Growth Stage: {predicted_stage} ({confidence:.1%} confidence)")
            
            # Biomass Estimation
            if "biomass_estimation" in analysis_options and "3d_reconstruction" in analysis_options:
                progress(0.9, desc="Estimating biomass...")
                
                if 'point_cloud' in locals():
                    biomass_estimate = self.plant_analyzer.estimate_biomass(point_cloud)
                    results['biomass_estimation'] = biomass_estimate
                    
                    summary_parts.append(f"✓ Biomass Estimate: {biomass_estimate:.2f} kg")
            
            progress(1.0, desc="Analysis complete!")
            
            # Store results in context for AI assistant
            self.context_manager.add_analysis_result(results)
            
            # Generate summary
            summary_text = "## Analysis Results\\n\\n" + "\\n".join(summary_parts)
            
            if not summary_parts:
                summary_text = "No analysis options selected. Please choose at least one analysis type."
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            summary_text = f"Error during analysis: {str(e)}"
            results['error'] = str(e)
        
        return results, plot_html, detection_image_path, summary_text
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        with gr.Blocks(
            title=self.config['ui']['title'],
            theme=self.config['ui']['theme']
        ) as interface:
            
            # Header
            gr.Markdown(f"# {self.config['ui']['title']}")
            gr.Markdown(self.config['ui']['description'])
            
            with gr.Row():
                # Left column - Input and Controls
                with gr.Column(scale=1):
                    # Image input
                    image_input = gr.Image(
                        label="Upload Plant Image",
                        type="numpy",
                        height=400
                    )
                    
                    # Analysis options
                    analysis_options = gr.CheckboxGroup(
                        label="Analysis Options",
                        choices=[
                            ("3D Reconstruction", "3d_reconstruction"),
                            ("Fruit Detection", "fruit_detection"), 
                            ("Health Analysis", "health_analysis"),
                            ("Growth Stage", "growth_analysis"),
                            ("Biomass Estimation", "biomass_estimation")
                        ],
                        value=["fruit_detection", "health_analysis"]
                    )
                    
                    # Analyze button
                    analyze_btn = gr.Button("🔍 Analyze Plant", variant="primary", size="lg")
                    
                    # Example images
                    gr.Examples(
                        examples=[
                            ["assets/demo_images/apple_tree.jpg"],
                            ["assets/demo_images/tomato_plant.jpg"],
                            ["assets/demo_images/citrus_tree.jpg"]
                        ],
                        inputs=image_input,
                        label="Example Images"
                    )
                
                # Right column - Results
                with gr.Column(scale=2):
                    # Results tabs
                    with gr.Tabs():
                        # Summary tab
                        with gr.Tab("📊 Summary"):
                            summary_output = gr.Markdown(label="Analysis Summary")
                        
                        # 3D Visualization tab
                        with gr.Tab("🌐 3D Reconstruction"):
                            plot_output = gr.HTML(label="3D Plant Model")
                        
                        # Detection Results tab
                        with gr.Tab("🍎 Fruit Detection"):
                            detection_output = gr.Image(label="Detection Results")
                        
                        # Raw Data tab
                        with gr.Tab("📋 Raw Data"):
                            json_output = gr.JSON(label="Analysis Results")
            
            # AI Assistant Section
            gr.Markdown("## 🤖 AI Agricultural Assistant")
            
            with gr.Row():
                with gr.Column():
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Agricultural AI Assistant",
                        height=300,
                        show_label=True
                    )
                    
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask about your plant analysis results...",
                            scale=4,
                            show_label=False
                        )
                        chat_btn = gr.Button("Send", scale=1, variant="secondary")
                    
                    # Quick suggestion buttons
                    quick_suggestions = gr.Row()
                    with quick_suggestions:
                        gr.Button("💡 Interpretation", size="sm")
                        gr.Button("🌱 Care Tips", size="sm") 
                        gr.Button("🔧 Troubleshooting", size="sm")
                        gr.Button("📈 Growth Advice", size="sm")
            
            # Event handlers
            analyze_btn.click(
                fn=self.process_plant_image,
                inputs=[image_input, analysis_options],
                outputs=[json_output, plot_output, detection_output, summary_output],
                show_progress=True
            )
            
            # Chat functionality
            def chat_response(message, history):
                if not message.strip():
                    return history, ""
                
                response = self.chat_interface.get_response(message)
                history.append([message, response])
                return history, ""
            
            chat_btn.click(
                fn=chat_response,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )
            
            chat_input.submit(
                fn=chat_response,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )
        
        return interface
    
    def launch(self, 
               share: bool = False,
               server_name: str = "127.0.0.1",
               server_port: int = 7860,
               **kwargs):
        """Launch the Gradio application"""
        
        interface = self.create_interface()
        
        logger.info(f"Launching BioViT3R-Beta on {server_name}:{server_port}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            **kwargs
        )

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BioViT3R-Beta Plant Analysis")
    parser.add_argument("--config", default="configs/app_config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--port", type=int, default=7860, 
                       help="Server port")
    parser.add_argument("--host", default="127.0.0.1", 
                       help="Server host")
    parser.add_argument("--share", action="store_true", 
                       help="Create public Gradio share link")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize and launch application
        app = BioViT3RApp(config_path=args.config)
        app.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port
        )
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()