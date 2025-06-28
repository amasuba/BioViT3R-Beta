"""
BioViT3R-Beta AI Assistant Package

This package contains AI-powered agricultural assistance including:
- IBM Granite AI integration
- Context-aware chat interface
- Analysis context management
- Agricultural expertise delivery
"""

from .granite_client import GraniteClient, GraniteAIError
from .chat_interface import ChatInterface, ChatMessage
from .context_manager import ContextManager, AnalysisContext

__version__ = "1.0.0-beta"
__author__ = "BioViT3R Team"

__all__ = [
    "GraniteClient",
    "GraniteAIError",
    "ChatInterface", 
    "ChatMessage",
    "ContextManager",
    "AnalysisContext",
]

# AI assistant configurations
ASSISTANT_CONFIGS = {
    "default": {
        "max_context_length": 10,
        "max_response_length": 512,
        "temperature": 0.7,
        "agricultural_focus": True,
    },
    "detailed": {
        "max_context_length": 20,
        "max_response_length": 1024,
        "temperature": 0.5,
        "agricultural_focus": True,
    },
    "concise": {
        "max_context_length": 5,
        "max_response_length": 256,
        "temperature": 0.8,
        "agricultural_focus": True,
    },
}

class AIAssistantFactory:
    """Factory class for creating AI assistant components."""
    
    @staticmethod
    def create_chat_interface(config_name: str = "default", **kwargs):
        """Create a chat interface with specified configuration."""
        config = ASSISTANT_CONFIGS.get(config_name, ASSISTANT_CONFIGS["default"])
        config.update(kwargs)
        return ChatInterface(**config)
    
    @staticmethod
    def create_granite_client(**kwargs):
        """Create a Granite AI client."""
        return GraniteClient(**kwargs)
    
    @staticmethod
    def create_context_manager(max_contexts: int = 10, **kwargs):
        """Create a context manager."""
        return ContextManager(max_contexts=max_contexts, **kwargs)

# Convenience functions
def create_assistant(config_name: str = "default", **kwargs):
    """
    Create a complete AI assistant setup.
    
    Args:
        config_name (str): Configuration preset name
        **kwargs: Additional configuration parameters
        
    Returns:
        tuple: (chat_interface, granite_client, context_manager)
    """
    factory = AIAssistantFactory()
    
    granite_client = factory.create_granite_client(**kwargs)
    context_manager = factory.create_context_manager(**kwargs)
    chat_interface = factory.create_chat_interface(
        config_name=config_name,
        granite_client=granite_client,
        context_manager=context_manager,
        **kwargs
    )
    
    return chat_interface, granite_client, context_manager

def get_assistant_config(config_name: str = "default"):
    """Get AI assistant configuration by name."""
    return ASSISTANT_CONFIGS.get(config_name, ASSISTANT_CONFIGS["default"])