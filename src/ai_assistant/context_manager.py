# src/ai_assistant/context_manager.py

from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from copy import deepcopy

logger = logging.getLogger(__name__)

@dataclass
class AnalysisContext:
    """Structure for storing analysis context"""
    timestamp: datetime
    image_path: str
    plant_health: Optional[Dict] = None
    growth_stage: Optional[Dict] = None
    fruit_detection: Optional[Dict] = None
    biomass_estimation: Optional[Dict] = None
    reconstruction_3d: Optional[Dict] = None
    environmental_data: Optional[Dict] = None
    metadata: Optional[Dict] = None

class ContextManager:
    """Manages analysis context for AI assistant interactions"""
    
    def __init__(self, max_contexts: int = 10, context_lifetime_hours: int = 24):
        self.contexts = []
        self.current_context = None
        self.max_contexts = max_contexts
        self.context_lifetime = timedelta(hours=context_lifetime_hours)
        
    def add_analysis_context(self, 
                           image_path: str,
                           plant_health: Optional[Dict] = None,
                           growth_stage: Optional[Dict] = None,
                           fruit_detection: Optional[Dict] = None,
                           biomass_estimation: Optional[Dict] = None,
                           reconstruction_3d: Optional[Dict] = None,
                           environmental_data: Optional[Dict] = None,
                           metadata: Optional[Dict] = None) -> str:
        """Add new analysis context"""
        
        context = AnalysisContext(
            timestamp=datetime.now(),
            image_path=image_path,
            plant_health=plant_health,
            growth_stage=growth_stage,
            fruit_detection=fruit_detection,
            biomass_estimation=biomass_estimation,
            reconstruction_3d=reconstruction_3d,
            environmental_data=environmental_data,
            metadata=metadata or {}
        )
        
        # Clean old contexts
        self._clean_old_contexts()
        
        # Add new context
        self.contexts.append(context)
        self.current_context = context
        
        # Limit context history
        if len(self.contexts) > self.max_contexts:
            self.contexts = self.contexts[-self.max_contexts:]
        
        logger.info(f"Added analysis context for image: {image_path}")
        return f"context_{len(self.contexts)}"
    
    def get_current_context(self) -> Optional[Dict]:
        """Get current analysis context as dictionary"""
        if not self.current_context:
            return None
        
        context_dict = asdict(self.current_context)
        # Convert datetime to string for JSON serialization
        context_dict["timestamp"] = self.current_context.timestamp.isoformat()
        
        return context_dict
    
    def get_context_by_index(self, index: int) -> Optional[Dict]:
        """Get context by index"""
        if 0 <= index < len(self.contexts):
            context = self.contexts[index]
            context_dict = asdict(context)
            context_dict["timestamp"] = context.timestamp.isoformat()
            return context_dict
        return None
    
    def get_all_contexts(self) -> List[Dict]:
        """Get all stored contexts"""
        result = []
        for context in self.contexts:
            context_dict = asdict(context)
            context_dict["timestamp"] = context.timestamp.isoformat()
            result.append(context_dict)
        return result
    
    def update_current_context(self, updates: Dict):
        """Update current context with new data"""
        if not self.current_context:
            logger.warning("No current context to update")
            return
        
        for key, value in updates.items():
            if hasattr(self.current_context, key):
                setattr(self.current_context, key, value)
                logger.info(f"Updated context field: {key}")
    
    def clear_current_context(self):
        """Clear current context"""
        self.current_context = None
        logger.info("Cleared current context")
    
    def clear_all_contexts(self):
        """Clear all stored contexts"""
        self.contexts = []
        self.current_context = None
        logger.info("Cleared all contexts")
    
    def _clean_old_contexts(self):
        """Remove contexts older than lifetime limit"""
        cutoff_time = datetime.now() - self.context_lifetime
        initial_count = len(self.contexts)
        
        self.contexts = [ctx for ctx in self.contexts if ctx.timestamp > cutoff_time]
        
        cleaned_count = initial_count - len(self.contexts)
        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} old contexts")
    
    def get_context_summary(self) -> Dict:
        """Get summary of all contexts"""
        if not self.contexts:
            return {
                "total_contexts": 0,
                "current_context": None,
                "oldest_context": None,
                "newest_context": None
            }
        
        return {
            "total_contexts": len(self.contexts),
            "current_context": self.current_context.image_path if self.current_context else None,
            "oldest_context": self.contexts[0].timestamp.isoformat(),
            "newest_context": self.contexts[-1].timestamp.isoformat(),
            "context_lifetime_hours": self.context_lifetime.total_seconds() / 3600
        }
    
    def find_contexts_by_criteria(self, criteria: Dict) -> List[Dict]:
        """Find contexts matching specific criteria"""
        matching_contexts = []
        
        for context in self.contexts:
            matches = True
            
            # Check image path criteria
            if "image_path" in criteria:
                if criteria["image_path"].lower() not in context.image_path.lower():
                    matches = False
            
            # Check health status criteria
            if "health_status" in criteria and context.plant_health:
                if context.plant_health.get("status") != criteria["health_status"]:
                    matches = False
            
            # Check growth stage criteria
            if "growth_stage" in criteria and context.growth_stage:
                if context.growth_stage.get("stage") != criteria["growth_stage"]:
                    matches = False
            
            # Check fruit count criteria
            if "min_fruits" in criteria and context.fruit_detection:
                fruit_count = len(context.fruit_detection.get("detections", []))
                if fruit_count < criteria["min_fruits"]:
                    matches = False
            
            # Check timestamp criteria
            if "after_date" in criteria:
                after_date = datetime.fromisoformat(criteria["after_date"])
                if context.timestamp < after_date:
                    matches = False
            
            if matches:
                context_dict = asdict(context)
                context_dict["timestamp"] = context.timestamp.isoformat()
                matching_contexts.append(context_dict)
        
        return matching_contexts
    
    def export_contexts(self, filepath: str):
        """Export all contexts to JSON file"""
        try:
            contexts_data = self.get_all_contexts()
            with open(filepath, 'w') as f:
                json.dump(contexts_data, f, indent=2)
            logger.info(f"Exported {len(contexts_data)} contexts to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export contexts: {e}")
    
    def import_contexts(self, filepath: str):
        """Import contexts from JSON file"""
        try:
            with open(filepath, 'r') as f:
                contexts_data = json.load(f)
            
            imported_contexts = []
            for ctx_data in contexts_data:
                # Convert timestamp string back to datetime
                ctx_data["timestamp"] = datetime.fromisoformat(ctx_data["timestamp"])
                context = AnalysisContext(**ctx_data)
                imported_contexts.append(context)
            
            self.contexts.extend(imported_contexts)
            
            # Limit total contexts
            if len(self.contexts) > self.max_contexts:
                self.contexts = self.contexts[-self.max_contexts:]
            
            logger.info(f"Imported {len(imported_contexts)} contexts from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to import contexts: {e}")
    
    def get_context_statistics(self) -> Dict:
        """Get statistics about stored contexts"""
        if not self.contexts:
            return {"message": "No contexts available"}
        
        stats = {
            "total_contexts": len(self.contexts),
            "health_status_distribution": {},
            "growth_stage_distribution": {},
            "average_fruit_count": 0,
            "context_age_range": {
                "oldest": self.contexts[0].timestamp.isoformat(),
                "newest": self.contexts[-1].timestamp.isoformat()
            }
        }
        
        # Analyze health status distribution
        health_counts = {}
        growth_counts = {}
        fruit_counts = []
        
        for context in self.contexts:
            if context.plant_health and "status" in context.plant_health:
                status = context.plant_health["status"]
                health_counts[status] = health_counts.get(status, 0) + 1
            
            if context.growth_stage and "stage" in context.growth_stage:
                stage = context.growth_stage["stage"]
                growth_counts[stage] = growth_counts.get(stage, 0) + 1
            
            if context.fruit_detection and "detections" in context.fruit_detection:
                fruit_count = len(context.fruit_detection["detections"])
                fruit_counts.append(fruit_count)
        
        stats["health_status_distribution"] = health_counts
        stats["growth_stage_distribution"] = growth_counts
        
        if fruit_counts:
            stats["average_fruit_count"] = sum(fruit_counts) / len(fruit_counts)
            stats["total_fruits_detected"] = sum(fruit_counts)
        
        return stats