# BioViT3R-Beta Command Line Interface
# CLI for batch processing and analysis workflows

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from datetime import datetime
import yaml
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.vggt_reconstructor import VGGTReconstructor
from src.models.plant_analyzer import PlantAnalyzer  
from src.models.fruit_detector import FruitDetector
from src.models.biomass_estimator import BiomassEstimator
from src.data.preprocessing import ImagePreprocessor
from src.utils.file_utils import FileUtils
from src.utils.video_processor import VideoProcessor
from src.utils.metrics import PlantAnalysisMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BioViT3RCLI:
    """Command-line interface for BioViT3R-Beta analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize CLI with configuration"""
        self.config_path = config_path or "configs/app_config.yaml"
        self.config = self.load_config()
        self.setup_models()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            return FileUtils.load_yaml(self.config_path)
        except Exception as e:
            logger.warning(f"Could not load config {self.config_path}: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default CLI configuration"""
        return {
            'models': {
                'vggt': {'model_path': 'models/vggt/', 'device': 'cpu'},
                'fruit_detector': {'model_path': 'models/fruit_detection/', 'confidence_threshold': 0.5},
                'health_classifier': {'model_path': 'models/health_classification/'}
            },
            'processing': {
                'max_image_size': 1024,
                'batch_size': 8,
                'num_workers': 4
            },
            'output': {
                'save_3d_models': True,
                'save_visualizations': True,
                'export_format': 'json'
            }
        }
    
    def setup_models(self):
        """Initialize analysis models"""
        logger.info("Initializing models...")
        
        try:
            self.preprocessor = ImagePreprocessor()
            
            # Initialize VGGT reconstructor
            vggt_config = self.config['models']['vggt']
            self.vggt_reconstructor = VGGTReconstructor(
                model_path=vggt_config['model_path'],
                device=vggt_config['device']
            )
            
            # Initialize plant analyzer
            health_config = self.config['models']['health_classifier']
            self.plant_analyzer = PlantAnalyzer(
                health_model_path=health_config['model_path']
            )
            
            # Initialize fruit detector
            fruit_config = self.config['models']['fruit_detector']
            self.fruit_detector = FruitDetector(
                model_path=fruit_config['model_path'],
                confidence_threshold=fruit_config['confidence_threshold']
            )
            
            # Initialize biomass estimator
            self.biomass_estimator = BiomassEstimator()
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def analyze_single_image(self,
                           image_path: str,
                           output_dir: str,
                           analysis_types: List[str],
                           export_formats: List[str] = None) -> Dict[str, Any]:
        """
        Analyze a single plant image
        
        Args:
            image_path: Path to input image
            output_dir: Output directory
            analysis_types: Types of analysis to perform
            export_formats: Export format options
            
        Returns:
            Analysis results dictionary
        """
        if export_formats is None:
            export_formats = ['json']
            
        logger.info(f"Analyzing image: {image_path}")
        
        # Load and preprocess image
        image = FileUtils.load_image(image_path)
        processed_image = self.preprocessor.preprocess_image(
            image, target_size=self.config['processing']['max_image_size']
        )
        
        results = {
            'input_file': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'image_shape': image.shape,
            'analysis_types': analysis_types
        }
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        try:
            # 3D Reconstruction
            if '3d' in analysis_types or 'reconstruction' in analysis_types:
                logger.info("Performing 3D reconstruction...")
                
                point_cloud = self.vggt_reconstructor.reconstruct(processed_image)
                mesh = self.vggt_reconstructor.generate_mesh(point_cloud)
                
                results['3d_reconstruction'] = {
                    'point_count': len(point_cloud),
                    'has_mesh': mesh is not None
                }
                
                # Save 3D data
                if self.config['output']['save_3d_models']:
                    # Save point cloud
                    pc_path = output_path / f"{base_name}_pointcloud.ply"
                    FileUtils.save_point_cloud(point_cloud, pc_path)
                    results['3d_reconstruction']['pointcloud_file'] = str(pc_path)
                    
                    # Save mesh if available
                    if mesh is not None:
                        mesh_path = output_path / f"{base_name}_mesh.ply"
                        FileUtils.save_mesh(mesh, mesh_path)
                        results['3d_reconstruction']['mesh_file'] = str(mesh_path)
            
            # Fruit Detection
            if 'fruit' in analysis_types or 'detection' in analysis_types:
                logger.info("Detecting fruits...")
                
                detections = self.fruit_detector.detect(processed_image)
                results['fruit_detection'] = {
                    'fruit_count': len(detections),
                    'detections': detections
                }
                
                # Save detection visualization
                if self.config['output']['save_visualizations']:
                    from src.utils.visualization import PlantVisualization
                    viz = PlantVisualization()
                    fig = viz.visualize_fruit_detection(processed_image, detections)
                    
                    vis_path = output_path / f"{base_name}_fruit_detection.png"
                    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
                    results['fruit_detection']['visualization_file'] = str(vis_path)
            
            # Health Analysis
            if 'health' in analysis_types:
                logger.info("Analyzing plant health...")
                
                health_results = self.plant_analyzer.analyze_health(processed_image)
                results['health_analysis'] = health_results
            
            # Growth Stage Analysis
            if 'growth' in analysis_types:
                logger.info("Classifying growth stage...")
                
                growth_results = self.plant_analyzer.classify_growth_stage(processed_image)
                results['growth_analysis'] = growth_results
            
            # Biomass Estimation
            if 'biomass' in analysis_types:
                logger.info("Estimating biomass...")
                
                if '3d_reconstruction' in results:
                    # Use point cloud if available
                    biomass_estimate = self.biomass_estimator.estimate_biomass(
                        point_cloud,
                        fruit_detections=results.get('fruit_detection', {}).get('detections', [])
                    )
                    results['biomass_estimation'] = biomass_estimate.__dict__
                else:
                    logger.warning("Biomass estimation requires 3D reconstruction")
            
            # Export results
            self.export_results(results, output_path, base_name, export_formats)
            
            logger.info(f"Analysis complete for {image_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            results['error'] = str(e)
            return results
    
    def batch_analyze_images(self,
                           input_dir: str,
                           output_dir: str,
                           analysis_types: List[str],
                           pattern: str = "*.jpg",
                           max_images: Optional[int] = None) -> Dict[str, Any]:
        """
        Batch analyze multiple images
        
        Args:
            input_dir: Input directory with images
            output_dir: Output directory
            analysis_types: Types of analysis to perform
            pattern: File pattern to match
            max_images: Maximum number of images to process
            
        Returns:
            Batch processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find images
        image_files = list(input_path.glob(pattern))
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        batch_results = {
            'input_directory': str(input_dir),
            'output_directory': str(output_dir),
            'total_images': len(image_files),
            'analysis_types': analysis_types,
            'start_time': datetime.now().isoformat(),
            'results': []
        }
        
        # Process each image
        successful = 0
        failed = 0
        
        for i, image_file in enumerate(image_files):
            logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # Create subdirectory for this image
                image_output_dir = output_path / image_file.stem
                
                result = self.analyze_single_image(
                    str(image_file),
                    str(image_output_dir),
                    analysis_types
                )
                
                if 'error' not in result:
                    successful += 1
                else:
                    failed += 1
                
                batch_results['results'].append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                batch_results['results'].append({
                    'input_file': str(image_file),
                    'error': str(e)
                })
                failed += 1
        
        # Finalize batch results
        batch_results.update({
            'end_time': datetime.now().isoformat(),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(image_files) if image_files else 0
        })
        
        # Save batch summary
        summary_path = output_path / "batch_summary.json"
        FileUtils.save_json(batch_results, summary_path)
        
        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
        return batch_results
    
    def analyze_video(self,
                     video_path: str,
                     output_dir: str,
                     analysis_types: List[str],
                     frame_interval: int = 30) -> Dict[str, Any]:
        """
        Analyze plant growth from video
        
        Args:
            video_path: Input video path
            output_dir: Output directory
            analysis_types: Types of analysis to perform
            frame_interval: Analyze every N frames
            
        Returns:
            Video analysis results
        """
        logger.info(f"Analyzing video: {video_path}")
        
        processor = VideoProcessor()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define analysis function for frames
        def analyze_frame(frame: np.ndarray) -> Dict[str, Any]:
            try:
                frame_results = {}
                
                # Preprocess frame
                processed_frame = self.preprocessor.preprocess_image(
                    frame, target_size=self.config['processing']['max_image_size']
                )
                
                # Perform selected analyses
                if 'fruit' in analysis_types:
                    detections = self.fruit_detector.detect(processed_frame)
                    frame_results['fruit_count'] = len(detections)
                    frame_results['fruit_detections'] = detections
                
                if 'health' in analysis_types:
                    health_results = self.plant_analyzer.analyze_health(processed_frame)
                    frame_results['health_score'] = health_results.get('overall_health', 0)
                
                if 'growth' in analysis_types:
                    growth_results = self.plant_analyzer.classify_growth_stage(processed_frame)
                    predicted_stage = max(growth_results.items(), key=lambda x: x[1])[0]
                    frame_results['growth_stage'] = predicted_stage
                
                return frame_results
                
            except Exception as e:
                return {'error': str(e)}
        
        # Process video frames
        frame_results = processor.process_video_frames(
            video_path,
            analyze_frame,
            frame_interval=frame_interval
        )
        
        # Analyze growth over time
        growth_analysis = processor.analyze_growth_over_time(frame_results)
        
        # Save results
        video_results = {
            'input_video': str(video_path),
            'total_frames_analyzed': len(frame_results),
            'frame_interval': frame_interval,
            'analysis_types': analysis_types,
            'growth_analysis': growth_analysis,
            'frame_results': [
                {
                    'frame_number': fr.frame_number,
                    'timestamp': fr.timestamp,
                    'analysis': fr.analysis_results
                }
                for fr in frame_results
            ]
        }
        
        # Export video analysis
        results_path = output_path / "video_analysis.json"
        FileUtils.save_json(video_results, results_path)
        
        # Save detailed summary
        processor.save_analysis_summary(frame_results, output_path / "detailed_summary.json")
        
        logger.info(f"Video analysis complete: {len(frame_results)} frames processed")
        return video_results
    
    def export_results(self,
                      results: Dict[str, Any],
                      output_dir: Path,
                      base_name: str,
                      formats: List[str]):
        """Export results in specified formats"""
        
        for format_type in formats:
            if format_type == 'json':
                json_path = output_dir / f"{base_name}_results.json"
                FileUtils.save_json(results, json_path)
                
            elif format_type == 'yaml':
                yaml_path = output_dir / f"{base_name}_results.yaml"
                FileUtils.save_yaml(results, yaml_path)
                
            elif format_type == 'csv':
                # Flatten results for CSV export
                flat_results = self.flatten_results(results)
                csv_path = output_dir / f"{base_name}_results.csv"
                
                import pandas as pd
                df = pd.DataFrame([flat_results])
                df.to_csv(csv_path, index=False)
    
    def flatten_results(self, results: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export"""
        flat = {}
        
        for key, value in results.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self.flatten_results(value, new_key))
            elif isinstance(value, list):
                flat[new_key] = len(value)  # Store count for lists
            else:
                flat[new_key] = value
                
        return flat

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="BioViT3R-Beta Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python main.py analyze-image input.jpg --output ./results --analysis fruit health

  # Batch process directory
  python main.py batch-analyze ./images --output ./batch_results --analysis all

  # Analyze video
  python main.py analyze-video growth.mp4 --output ./video_results --interval 60
        """
    )
    
    parser.add_argument('--config', '-c', default='configs/app_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single image analysis
    img_parser = subparsers.add_parser('analyze-image', help='Analyze single image')
    img_parser.add_argument('image_path', help='Path to input image')
    img_parser.add_argument('--output', '-o', required=True, help='Output directory')
    img_parser.add_argument('--analysis', '-a', nargs='+', 
                           choices=['3d', 'fruit', 'health', 'growth', 'biomass', 'all'],
                           default=['fruit', 'health'], help='Analysis types')
    img_parser.add_argument('--format', '-f', nargs='+',
                           choices=['json', 'yaml', 'csv'],
                           default=['json'], help='Export formats')
    
    # Batch analysis
    batch_parser = subparsers.add_parser('batch-analyze', help='Batch analyze images')
    batch_parser.add_argument('input_dir', help='Input directory with images')
    batch_parser.add_argument('--output', '-o', required=True, help='Output directory')
    batch_parser.add_argument('--analysis', '-a', nargs='+',
                             choices=['3d', 'fruit', 'health', 'growth', 'biomass', 'all'],
                             default=['fruit', 'health'], help='Analysis types')
    batch_parser.add_argument('--pattern', '-p', default='*.jpg',
                             help='File pattern to match')
    batch_parser.add_argument('--max-images', '-m', type=int,
                             help='Maximum number of images to process')
    
    # Video analysis
    video_parser = subparsers.add_parser('analyze-video', help='Analyze video')
    video_parser.add_argument('video_path', help='Path to input video')
    video_parser.add_argument('--output', '-o', required=True, help='Output directory')
    video_parser.add_argument('--analysis', '-a', nargs='+',
                             choices=['fruit', 'health', 'growth', 'all'],
                             default=['fruit', 'health'], help='Analysis types')
    video_parser.add_argument('--interval', '-i', type=int, default=30,
                             help='Frame interval for analysis')
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize CLI
        cli = BioViT3RCLI(args.config)
        
        # Handle 'all' analysis type
        if hasattr(args, 'analysis') and 'all' in args.analysis:
            if args.command == 'analyze-video':
                args.analysis = ['fruit', 'health', 'growth']
            else:
                args.analysis = ['3d', 'fruit', 'health', 'growth', 'biomass']
        
        # Execute command
        if args.command == 'analyze-image':
            results = cli.analyze_single_image(
                args.image_path,
                args.output,
                args.analysis,
                args.format
            )
            if 'error' in results:
                logger.error(f"Analysis failed: {results['error']}")
                sys.exit(1)
                
        elif args.command == 'batch-analyze':
            results = cli.batch_analyze_images(
                args.input_dir,
                args.output,
                args.analysis,
                args.pattern,
                args.max_images
            )
            if results['failed'] > 0:
                logger.warning(f"{results['failed']} images failed to process")
                
        elif args.command == 'analyze-video':
            results = cli.analyze_video(
                args.video_path,
                args.output,
                args.analysis,
                args.interval
            )
        
        logger.info("Command completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()