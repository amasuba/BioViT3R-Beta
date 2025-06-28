# BioViT3R-Beta Video Processing Module
# Multi-frame Video Analysis for Plant Growth Monitoring

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any, Generator
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoInfo:
    """Container for video metadata"""
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
    codec: str
    format: str

@dataclass
class FrameInfo:
    """Container for frame metadata"""
    frame_number: int
    timestamp: float
    frame_array: np.ndarray
    analysis_results: Optional[Dict[str, Any]] = None

class VideoProcessor:
    """Advanced video processing for plant growth analysis"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 temp_dir: Optional[str] = None):
        """
        Initialize video processor
        
        Args:
            max_workers: Maximum number of worker threads
            temp_dir: Temporary directory for processing
        """
        self.max_workers = max_workers
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_video_info(self, video_path: Union[str, Path]) -> VideoInfo:
        """
        Extract video metadata
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information
        """
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            duration = total_frames / fps if fps > 0 else 0
            
            return VideoInfo(
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
                width=width,
                height=height,
                codec=codec,
                format=Path(video_path).suffix
            )
        finally:
            cap.release()
    
    def extract_frames(self, 
                      video_path: Union[str, Path],
                      output_dir: Union[str, Path],
                      frame_interval: Optional[int] = None,
                      time_interval: Optional[float] = None,
                      max_frames: Optional[int] = None,
                      start_time: float = 0,
                      end_time: Optional[float] = None,
                      frame_format: str = 'jpg') -> List[Path]:
        """
        Extract frames from video with various sampling strategies
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            frame_interval: Extract every N frames
            time_interval: Extract frames at time intervals (seconds)
            max_frames: Maximum number of frames to extract
            start_time: Start time in seconds
            end_time: End time in seconds
            frame_format: Output image format
            
        Returns:
            List of extracted frame paths
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        video_info = self.get_video_info(video_path)
        
        # Determine sampling strategy
        if time_interval is not None:
            frame_interval = int(time_interval * video_info.fps)
        elif frame_interval is None:
            frame_interval = max(1, video_info.total_frames // 100)  # Default: 100 frames
        
        # Set start and end frames
        start_frame = int(start_time * video_info.fps)
        end_frame = int(end_time * video_info.fps) if end_time else video_info.total_frames
        
        extracted_paths = []
        frame_count = 0
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_num in range(start_frame, end_frame, frame_interval):
                if max_frames and frame_count >= max_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Generate frame filename
                timestamp = frame_num / video_info.fps
                filename = f"frame_{frame_num:06d}_{timestamp:.2f}s.{frame_format}"
                frame_path = output_dir / filename
                
                # Save frame
                if cv2.imwrite(str(frame_path), frame):
                    extracted_paths.append(frame_path)
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        self.logger.info(f"Extracted {frame_count} frames...")
        
        finally:
            cap.release()
        
        self.logger.info(f"Extracted {len(extracted_paths)} frames to {output_dir}")
        return extracted_paths
    
    def process_video_frames(self,
                           video_path: Union[str, Path],
                           analysis_function,
                           batch_size: int = 10,
                           frame_interval: int = 30,
                           max_frames: Optional[int] = None) -> List[FrameInfo]:
        """
        Process video frames with analysis function
        
        Args:
            video_path: Path to video file
            analysis_function: Function to analyze each frame
            batch_size: Number of frames to process in parallel
            frame_interval: Process every N frames
            max_frames: Maximum frames to process
            
        Returns:
            List of processed frame information
        """
        cap = cv2.VideoCapture(str(video_path))
        video_info = self.get_video_info(video_path)
        
        frame_results = []
        frame_batch = []
        
        try:
            frame_num = 0
            processed_count = 0
            
            while True:
                if max_frames and processed_count >= max_frames:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only at specified intervals
                if frame_num % frame_interval == 0:
                    timestamp = frame_num / video_info.fps
                    frame_info = FrameInfo(
                        frame_number=frame_num,
                        timestamp=timestamp,
                        frame_array=frame.copy()
                    )
                    frame_batch.append(frame_info)
                    
                    # Process batch when full
                    if len(frame_batch) >= batch_size:
                        batch_results = self._process_frame_batch(frame_batch, analysis_function)
                        frame_results.extend(batch_results)
                        frame_batch = []
                        processed_count += len(batch_results)
                        
                        self.logger.info(f"Processed {processed_count} frames...")
                
                frame_num += 1
            
            # Process remaining frames
            if frame_batch:
                batch_results = self._process_frame_batch(frame_batch, analysis_function)
                frame_results.extend(batch_results)
        
        finally:
            cap.release()
        
        return frame_results
    
    def _process_frame_batch(self, 
                           frame_batch: List[FrameInfo],
                           analysis_function) -> List[FrameInfo]:
        """Process batch of frames in parallel"""
        def analyze_frame(frame_info):
            try:
                results = analysis_function(frame_info.frame_array)
                frame_info.analysis_results = results
                return frame_info
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_info.frame_number}: {e}")
                frame_info.analysis_results = {"error": str(e)}
                return frame_info
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(analyze_frame, frame) for frame in frame_batch]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        # Sort by frame number to maintain order
        results.sort(key=lambda x: x.frame_number)
        return results
    
    def create_time_lapse(self,
                         frame_paths: List[Union[str, Path]],
                         output_path: Union[str, Path],
                         fps: float = 30,
                         include_timestamp: bool = True,
                         timestamp_format: str = "%H:%M:%S") -> bool:
        """
        Create time-lapse video from frame sequence
        
        Args:
            frame_paths: List of frame image paths
            output_path: Output video path
            fps: Output video frame rate
            include_timestamp: Whether to overlay timestamp
            timestamp_format: Timestamp format string
            
        Returns:
            Success status
        """
        if not frame_paths:
            return False
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            return False
        
        height, width = first_frame.shape[:2]
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        try:
            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                
                # Add timestamp overlay if requested
                if include_timestamp:
                    # Extract timestamp from filename if available
                    try:
                        filename = Path(frame_path).stem
                        if '_' in filename:
                            timestamp_str = filename.split('_')[-1].replace('s', '')
                            timestamp_seconds = float(timestamp_str)
                            
                            # Convert to time format
                            time_obj = datetime.fromtimestamp(timestamp_seconds)
                            time_text = time_obj.strftime(timestamp_format)
                            
                            # Add text overlay
                            cv2.putText(frame, time_text, (20, 40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    except:
                        pass
                
                out.write(frame)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(frame_paths)} frames for time-lapse")
        
        finally:
            out.release()
        
        self.logger.info(f"Time-lapse video saved to {output_path}")
        return True
    
    def analyze_growth_over_time(self,
                               analysis_results: List[FrameInfo],
                               metrics: List[str] = None) -> Dict[str, Any]:
        """
        Analyze plant growth trends over time
        
        Args:
            analysis_results: List of frame analysis results
            metrics: Specific metrics to track
            
        Returns:
            Growth analysis summary
        """
        if not analysis_results:
            return {}
        
        if metrics is None:
            metrics = ['plant_height', 'canopy_area', 'fruit_count', 'health_score']
        
        # Extract time series data
        timestamps = []
        metric_values = {metric: [] for metric in metrics}
        
        for frame_info in analysis_results:
            if frame_info.analysis_results and 'error' not in frame_info.analysis_results:
                timestamps.append(frame_info.timestamp)
                
                for metric in metrics:
                    value = frame_info.analysis_results.get(metric, 0)
                    metric_values[metric].append(value)
        
        if not timestamps:
            return {}
        
        # Compute growth statistics
        growth_analysis = {
            'time_range': {
                'start': min(timestamps),
                'end': max(timestamps),
                'duration_hours': (max(timestamps) - min(timestamps)) / 3600
            },
            'metrics': {}
        }
        
        for metric, values in metric_values.items():
            if values:
                growth_analysis['metrics'][metric] = {
                    'initial_value': values[0],
                    'final_value': values[-1],
                    'min_value': min(values),
                    'max_value': max(values),
                    'mean_value': np.mean(values),
                    'growth_rate': (values[-1] - values[0]) / len(values) if len(values) > 1 else 0,
                    'trend': 'increasing' if values[-1] > values[0] else 'decreasing' if values[-1] < values[0] else 'stable'
                }
        
        return growth_analysis
    
    def create_analysis_video(self,
                            frame_results: List[FrameInfo],
                            output_path: Union[str, Path],
                            overlay_function = None,
                            fps: float = 30) -> bool:
        """
        Create video with analysis overlays
        
        Args:
            frame_results: Processed frame results
            output_path: Output video path
            overlay_function: Function to add analysis overlays
            fps: Output frame rate
            
        Returns:
            Success status
        """
        if not frame_results:
            return False
        
        # Get frame dimensions from first frame
        first_frame = frame_results[0].frame_array
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        try:
            for frame_info in frame_results:
                frame = frame_info.frame_array.copy()
                
                # Apply analysis overlays
                if overlay_function and frame_info.analysis_results:
                    frame = overlay_function(frame, frame_info.analysis_results)
                
                out.write(frame)
        
        finally:
            out.release()
        
        return True
    
    def save_analysis_summary(self,
                            analysis_results: List[FrameInfo],
                            output_path: Union[str, Path]) -> bool:
        """
        Save comprehensive analysis summary to JSON
        
        Args:
            analysis_results: Frame analysis results
            output_path: Output JSON path
            
        Returns:
            Success status
        """
        try:
            # Prepare summary data
            summary = {
                'metadata': {
                    'total_frames': len(analysis_results),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'time_range': {
                        'start': min(r.timestamp for r in analysis_results) if analysis_results else 0,
                        'end': max(r.timestamp for r in analysis_results) if analysis_results else 0
                    }
                },
                'frames': []
            }
            
            # Add frame data
            for frame_info in analysis_results:
                frame_data = {
                    'frame_number': frame_info.frame_number,
                    'timestamp': frame_info.timestamp,
                    'analysis_results': frame_info.analysis_results
                }
                summary['frames'].append(frame_data)
            
            # Add growth analysis
            growth_analysis = self.analyze_growth_over_time(analysis_results)
            summary['growth_analysis'] = growth_analysis
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save analysis summary: {e}")
            return False

class FrameExtractor:
    """Specialized frame extraction utilities"""
    
    @staticmethod
    def extract_keyframes(video_path: Union[str, Path],
                         output_dir: Union[str, Path],
                         threshold: float = 0.3) -> List[Path]:
        """
        Extract keyframes based on scene changes
        
        Args:
            video_path: Input video path
            output_dir: Output directory
            threshold: Scene change threshold
            
        Returns:
            List of keyframe paths
        """
        cap = cv2.VideoCapture(str(video_path))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        keyframe_paths = []
        prev_frame = None
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Compute frame difference
                    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
                    
                    # Calculate change percentage
                    change_ratio = np.sum(diff > 30) / diff.size
                    
                    # Save keyframe if significant change detected
                    if change_ratio > threshold:
                        keyframe_path = output_dir / f"keyframe_{frame_num:06d}.jpg"
                        cv2.imwrite(str(keyframe_path), frame)
                        keyframe_paths.append(keyframe_path)
                
                prev_frame = frame
                frame_num += 1
        
        finally:
            cap.release()
        
        return keyframe_paths
    
    @staticmethod
    def extract_uniform_samples(video_path: Union[str, Path],
                              output_dir: Union[str, Path],
                              num_samples: int = 50) -> List[Path]:
        """
        Extract uniformly distributed frame samples
        
        Args:
            video_path: Input video path
            output_dir: Output directory
            num_samples: Number of samples to extract
            
        Returns:
            List of sample frame paths
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if num_samples >= total_frames:
            interval = 1
        else:
            interval = total_frames // num_samples
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sample_paths = []
        
        try:
            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    sample_path = output_dir / f"sample_{i:06d}.jpg"
                    cv2.imwrite(str(sample_path), frame)
                    sample_paths.append(sample_path)
                    
                    if len(sample_paths) >= num_samples:
                        break
        
        finally:
            cap.release()
        
        return sample_paths

# Utility functions for video analysis
def simple_overlay_function(frame: np.ndarray, analysis_results: Dict[str, Any]) -> np.ndarray:
    """Simple overlay function for analysis visualization"""
    if 'fruit_count' in analysis_results:
        cv2.putText(frame, f"Fruits: {analysis_results['fruit_count']}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if 'health_score' in analysis_results:
        cv2.putText(frame, f"Health: {analysis_results['health_score']:.2f}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def batch_process_videos(video_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        analysis_function,
                        max_workers: int = 2) -> Dict[str, Any]:
    """Process multiple videos in parallel"""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
    
    def process_single_video(video_path):
        processor = VideoProcessor(max_workers=2)
        results = processor.process_video_frames(video_path, analysis_function)
        
        # Save results
        output_file = output_dir / f"{video_path.stem}_analysis.json"
        processor.save_analysis_summary(results, output_file)
        
        return {
            'video_path': str(video_path),
            'results_path': str(output_file),
            'frame_count': len(results)
        }
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_video, video) for video in video_files]
        
        results = {}
        for future in as_completed(futures):
            result = future.result()
            results[result['video_path']] = result
    
    return results