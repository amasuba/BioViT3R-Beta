# src/ai_assistant/__init__.py

"""
BioViT3R AI Assistant Module

This module provides IBM Granite AI integration for agricultural expertise,
contextual analysis assistance, and intelligent chat interfaces.
"""

from .granite_client import GraniteClient, GraniteConfig
from .chat_interface import ChatInterface
from .context_manager import ContextManager, AnalysisContext

__version__ = "1.0.0"
__author__ = "BioViT3R Team"

__all__ = [
    "GraniteClient",
    "GraniteConfig", 
    "ChatInterface",
    "ContextManager",
    "AnalysisContext"
]