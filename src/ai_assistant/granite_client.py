# src/ai_assistant/granite_client.py

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import aiohttp
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.credentials import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

logger = logging.getLogger(__name__)

@dataclass
class GraniteConfig:
    """Configuration for IBM Granite AI integration"""
    api_key: str
    project_id: str
    url: str = "https://us-south.ml.cloud.ibm.com"
    model_id: str = "ibm/granite-13b-chat-v2"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

class GraniteClient:
    """IBM Granite AI client for agricultural expertise and plant analysis assistance"""
    
    def __init__(self, config: Optional[GraniteConfig] = None):
        self.config = config or self._load_config_from_env()
        self.model = None
        self._initialize_client()
        
    def _load_config_from_env(self) -> GraniteConfig:
        """Load configuration from environment variables"""
        return GraniteConfig(
            api_key=os.getenv("IBM_WATSON_APIKEY", ""),
            project_id=os.getenv("IBM_PROJECT_ID", ""),
            url=os.getenv("IBM_WATSON_URL", "https://us-south.ml.cloud.ibm.com"),
            model_id=os.getenv("IBM_MODEL_ID", "ibm/granite-13b-chat-v2"),
            max_new_tokens=int(os.getenv("IBM_MAX_TOKENS", "512")),
            temperature=float(os.getenv("IBM_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("IBM_TOP_P", "0.9")),
            repetition_penalty=float(os.getenv("IBM_REP_PENALTY", "1.1"))
        )
    
    def _initialize_client(self):
        """Initialize IBM Watsonx AI client"""
        try:
            credentials = Credentials(
                url=self.config.url,
                api_key=self.config.api_key
            )
            
            # Set generation parameters
            parameters = {
                GenParams.MAX_NEW_TOKENS: self.config.max_new_tokens,
                GenParams.TEMPERATURE: self.config.temperature,
                GenParams.TOP_P: self.config.top_p,
                GenParams.REPETITION_PENALTY: self.config.repetition_penalty
            }
            
            self.model = Model(
                model_id=self.config.model_id,
                params=parameters,
                credentials=credentials,
                project_id=self.config.project_id
            )
            
            logger.info("IBM Granite client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Granite client: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if Granite client is properly initialized"""
        return self.model is not None
    
    async def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response from Granite AI model"""
        if not self.is_available():
            return "AI assistant is currently unavailable. Please check your IBM Watsonx configuration."
        
        try:
            # Enhance prompt with agricultural context
            enhanced_prompt = self._enhance_prompt_with_context(prompt, context)
            
            # Generate response
            response = self.model.generate_text(prompt=enhanced_prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating Granite response: {e}")
            return f"Error: Unable to generate response. {str(e)}"
    
    def _enhance_prompt_with_context(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Enhance user prompt with agricultural expertise context"""
        
        system_context = """You are an expert agricultural AI assistant specializing in:
- Plant health assessment and disease diagnosis
- Fruit detection and yield estimation
- Growth stage classification and phenological analysis  
- 3D plant reconstruction and morphological analysis
- Precision agriculture and sustainable farming practices

Provide accurate, scientifically-grounded responses focused on practical agricultural applications.
"""
        
        if context:
            analysis_context = self._format_analysis_context(context)
            enhanced_prompt = f"{system_context}\n\nAnalysis Context:\n{analysis_context}\n\nUser Question: {prompt}\n\nResponse:"
        else:
            enhanced_prompt = f"{system_context}\n\nUser Question: {prompt}\n\nResponse:"
        
        return enhanced_prompt
    
    def _format_analysis_context(self, context: Dict) -> str:
        """Format analysis results for context"""
        formatted_parts = []
        
        if "plant_health" in context:
            health = context["plant_health"]
            formatted_parts.append(f"Plant Health: {health.get('status', 'Unknown')} (confidence: {health.get('confidence', 0):.2f})")
        
        if "growth_stage" in context:
            growth = context["growth_stage"]
            formatted_parts.append(f"Growth Stage: {growth.get('stage', 'Unknown')} (confidence: {growth.get('confidence', 0):.2f})")
        
        if "fruit_detection" in context:
            fruits = context["fruit_detection"]
            count = len(fruits.get("detections", []))
            formatted_parts.append(f"Detected Fruits: {count} fruits found")
        
        if "biomass_estimation" in context:
            biomass = context["biomass_estimation"]
            formatted_parts.append(f"Estimated Biomass: {biomass.get('total_biomass', 0):.2f} kg")
        
        if "3d_reconstruction" in context:
            reconstruction = context["3d_reconstruction"]
            points = reconstruction.get("point_count", 0)
            formatted_parts.append(f"3D Reconstruction: {points} points generated")
        
        return "\n".join(formatted_parts) if formatted_parts else "No analysis context available"
    
    async def chat_with_analysis(self, user_message: str, analysis_results: Dict) -> str:
        """Chat interface with plant analysis context"""
        return await self.generate_response(user_message, analysis_results)
    
    def get_agricultural_suggestions(self, plant_type: str, issue: str) -> str:
        """Get agricultural suggestions for specific plant issues"""
        prompt = f"""
        As an agricultural expert, provide actionable recommendations for:
        Plant Type: {plant_type}
        Observed Issue: {issue}
        
        Include:
        1. Immediate actions to take
        2. Preventive measures
        3. Long-term management strategies
        4. When to consult specialists
        """
        
        try:
            return self.model.generate_text(prompt=prompt)
        except Exception as e:
            logger.error(f"Error generating agricultural suggestions: {e}")
            return "Unable to generate suggestions at this time."
    
    def explain_analysis_results(self, results: Dict) -> str:
        """Explain technical analysis results in user-friendly terms"""
        prompt = f"""
        Explain these plant analysis results in simple, practical terms for farmers and agricultural professionals:
        
        {json.dumps(results, indent=2)}
        
        Focus on:
        1. What these results mean for plant health
        2. Recommended actions based on findings
        3. Economic implications if relevant
        4. Timeline for expected outcomes
        """
        
        try:
            return self.model.generate_text(prompt=prompt)
        except Exception as e:
            logger.error(f"Error explaining analysis results: {e}")
            return "Unable to explain results at this time."