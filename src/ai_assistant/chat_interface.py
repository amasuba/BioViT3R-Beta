# src/ai_assistant/chat_interface.py

import asyncio
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import gradio as gr
from .granite_client import GraniteClient
from .context_manager import ContextManager

logger = logging.getLogger(__name__)

class ChatInterface:
    """Gradio chat interface with agricultural AI assistance"""
    
    def __init__(self, granite_client: GraniteClient, context_manager: ContextManager):
        self.granite_client = granite_client
        self.context_manager = context_manager
        self.chat_history = []
        self.max_history_length = 50
        
    def create_chat_interface(self) -> gr.Interface:
        """Create Gradio chat interface"""
        
        with gr.Blocks(title="Agricultural AI Assistant") as interface:
            gr.Markdown("## 🌱 BioViT3R Agricultural AI Assistant")
            gr.Markdown("Ask questions about plant analysis, agricultural practices, or get explanations of your analysis results.")
            
            # Chat display
            chatbot = gr.Chatbot(
                value=[],
                height=400,
                show_label=False,
                avatar_images=["👤", "🤖"]
            )
            
            # Input components
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask about plant health, farming practices, or analysis results...",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
            
            # Context toggle
            with gr.Row():
                use_context = gr.Checkbox(
                    value=True,
                    label="Include analysis context in responses"
                )
                
            # Quick suggestions
            gr.Markdown("### Quick Questions:")
            with gr.Row():
                suggestion_btns = [
                    gr.Button("Explain my analysis results", size="sm"),
                    gr.Button("What does this health score mean?", size="sm"),
                    gr.Button("How to improve plant health?", size="sm"),
                    gr.Button("Best practices for this growth stage", size="sm")
                ]
            
            # Event handlers
            send_btn.click(
                self._process_message,
                inputs=[msg_input, chatbot, use_context],
                outputs=[msg_input, chatbot]
            )
            
            msg_input.submit(
                self._process_message,
                inputs=[msg_input, chatbot, use_context],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                self._clear_chat,
                outputs=[chatbot]
            )
            
            # Quick suggestion handlers
            for btn in suggestion_btns:
                btn.click(
                    self._handle_suggestion,
                    inputs=[btn, chatbot, use_context],
                    outputs=[chatbot]
                )
        
        return interface
    
    async def _process_message(self, user_input: str, chat_history: List, use_context: bool) -> Tuple[str, List]:
        """Process user message and generate AI response"""
        if not user_input.strip():
            return "", chat_history
        
        # Add user message to history
        chat_history.append([user_input, None])
        
        try:
            # Get analysis context if requested
            context = None
            if use_context:
                context = self.context_manager.get_current_context()
            
            # Generate AI response
            if self.granite_client.is_available():
                response = await self.granite_client.generate_response(user_input, context)
            else:
                response = self._get_fallback_response(user_input, context)
            
            # Add AI response to history
            chat_history[-1][1] = response
            
            # Manage chat history length
            if len(chat_history) > self.max_history_length:
                chat_history = chat_history[-self.max_history_length:]
            
            self.chat_history = chat_history
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            chat_history[-1][1] = "Sorry, I encountered an error processing your message. Please try again."
        
        return "", chat_history
    
    def _handle_suggestion(self, button_text: str, chat_history: List, use_context: bool) -> List:
        """Handle quick suggestion button clicks"""
        # Extract button text as user input
        user_input = button_text
        
        # Process as regular message
        result = asyncio.run(self._process_message(user_input, chat_history, use_context))
        return result[1]  # Return updated chat history
    
    def _clear_chat(self) -> List:
        """Clear chat history"""
        self.chat_history = []
        return []
    
    def _get_fallback_response(self, user_input: str, context: Optional[Dict] = None) -> str:
        """Provide fallback responses when AI is unavailable"""
        
        # Common agricultural questions and responses
        fallback_responses = {
            "health": "Plant health assessment typically considers factors like leaf color, growth rate, pest presence, and environmental stress indicators. For specific recommendations, consult local agricultural extension services.",
            
            "growth": "Plant growth stages vary by species but generally include germination, vegetative growth, flowering, and fruiting phases. Each stage has specific care requirements.",
            
            "disease": "Common plant diseases include fungal infections, bacterial diseases, and viral infections. Look for symptoms like spots, wilting, or unusual coloration. Consider laboratory testing for accurate diagnosis.",
            
            "nutrition": "Plants require primary nutrients (NPK), secondary nutrients (Ca, Mg, S), and micronutrients. Soil testing can help determine nutrient needs.",
            
            "water": "Proper watering depends on plant type, soil conditions, and climate. Most plants prefer consistent moisture without waterlogging."
        }
        
        # Simple keyword matching for fallback responses
        user_lower = user_input.lower()
        for keyword, response in fallback_responses.items():
            if keyword in user_lower:
                return f"AI Assistant is currently unavailable. Here's general guidance: {response}"
        
        # Include context information if available
        if context:
            context_info = self._format_context_summary(context)
            return f"AI Assistant is currently unavailable. Based on your analysis results: {context_info}"
        
        return "AI Assistant is currently unavailable. Please check your IBM Watsonx configuration and try again."
    
    def _format_context_summary(self, context: Dict) -> str:
        """Format analysis context for fallback responses"""
        summary_parts = []
        
        if "plant_health" in context:
            health = context["plant_health"]
            status = health.get("status", "Unknown")
            confidence = health.get("confidence", 0)
            summary_parts.append(f"Plant appears {status} (confidence: {confidence:.1%})")
        
        if "growth_stage" in context:
            growth = context["growth_stage"]
            stage = growth.get("stage", "Unknown")
            summary_parts.append(f"Growth stage: {stage}")
        
        if "fruit_detection" in context:
            fruits = context["fruit_detection"]
            count = len(fruits.get("detections", []))
            if count > 0:
                summary_parts.append(f"{count} fruits detected")
        
        if summary_parts:
            return ". ".join(summary_parts) + ". For detailed recommendations, please enable AI assistance."
        else:
            return "Analysis completed. Enable AI assistance for detailed insights."
    
    def add_system_message(self, message: str):
        """Add system message to chat history"""
        timestamp = datetime.now().strftime("%H:%M")
        system_msg = f"🔔 System ({timestamp}): {message}"
        self.chat_history.append([None, system_msg])
    
    def get_chat_history(self) -> List:
        """Get current chat history"""
        return self.chat_history
    
    def load_chat_history(self, history: List):
        """Load chat history from saved state"""
        self.chat_history = history[-self.max_history_length:] if history else []