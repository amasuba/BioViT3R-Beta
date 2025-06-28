#!/usr/bin/env python3
"""
BioViT3R-Beta AI Assistant Tests
Tests for IBM Granite AI assistant integration and chatbot functionality.
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ai_assistant.granite_client import GraniteClient
from src.ai_assistant.chat_interface import ChatInterface
from src.ai_assistant.context_manager import ContextManager

class TestGraniteClient(unittest.TestCase):
    """Test IBM Granite AI client functionality."""

    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'IBM_WATSON_APIKEY': 'test_api_key',
            'IBM_PROJECT_ID': 'test_project_id',
            'IBM_WATSON_URL': 'https://test.watson.ml.cloud.ibm.com'
        })
        self.env_patcher.start()

        # Initialize client
        self.client = GraniteClient()

    def tearDown(self):
        """Clean up test environment."""
        self.env_patcher.stop()

    def test_client_initialization(self):
        """Test Granite client initialization."""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.api_key, 'test_api_key')
        self.assertEqual(self.client.project_id, 'test_project_id')
        self.assertEqual(self.client.base_url, 'https://test.watson.ml.cloud.ibm.com')

    @patch('requests.post')
    def test_basic_chat_functionality(self, mock_post):
        """Test basic chat functionality."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{
                'generated_text': 'This is a test response from Granite AI about plant health.'
            }]
        }
        mock_post.return_value = mock_response

        # Test chat
        response = self.client.chat("What is wrong with my plant?")

        # Verify response
        self.assertIsInstance(response, str)
        self.assertIn("plant health", response)

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn('test.watson.ml.cloud.ibm.com', call_args[0][0])

    @patch('requests.post')
    def test_context_aware_responses(self, mock_post):
        """Test context-aware responses."""
        # Mock API responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{
                'generated_text': 'Based on the analysis showing 85% health score, your plant appears healthy.'
            }]
        }
        mock_post.return_value = mock_response

        # Create context
        analysis_context = {
            "health_score": 0.85,
            "growth_stage": "flowering",
            "detected_issues": []
        }

        # Test context-aware chat
        response = self.client.chat_with_context(
            "How is my plant doing?",
            analysis_context
        )

        # Verify context integration
        self.assertIsInstance(response, str)
        self.assertIn("85%", response)

        # Check that context was included in API call
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        self.assertIn("health_score", str(request_data))

    @patch('requests.post')
    def test_agricultural_expertise(self, mock_post):
        """Test agricultural expertise in responses."""
        # Mock expert response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [{
                'generated_text': 'For tomato plants showing yellow leaves, check for nitrogen deficiency. Apply balanced fertilizer and ensure proper drainage.'
            }]
        }
        mock_post.return_value = mock_response

        # Test agricultural question
        response = self.client.chat("My tomato plant has yellow leaves. What should I do?")

        # Verify agricultural expertise
        self.assertIsInstance(response, str)
        self.assertIn("nitrogen", response.lower())
        self.assertIn("fertilizer", response.lower())

    @patch('requests.post')
    def test_error_handling(self, mock_post):
        """Test error handling for API failures."""
        # Mock API error
        mock_post.side_effect = Exception("Network error")

        # Test error handling
        response = self.client.chat("Test question")

        # Should return fallback response
        self.assertIsInstance(response, str)
        self.assertIn("I'm having trouble", response)

    @patch('requests.post')
    def test_rate_limiting(self, mock_post):
        """Test rate limiting handling."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            'error': 'Rate limit exceeded'
        }
        mock_post.return_value = mock_response

        # Test rate limiting
        response = self.client.chat("Test question")

        # Should handle rate limiting gracefully
        self.assertIsInstance(response, str)
        self.assertIn("busy", response.lower())


class TestChatInterface(unittest.TestCase):
    """Test chat interface functionality."""

    def setUp(self):
        """Set up test environment."""
        # Mock Granite client
        self.mock_client = Mock()

        with patch('src.ai_assistant.chat_interface.GraniteClient') as mock_granite:
            mock_granite.return_value = self.mock_client
            self.chat_interface = ChatInterface()

    def test_interface_initialization(self):
        """Test chat interface initialization."""
        self.assertIsNotNone(self.chat_interface)
        self.assertTrue(hasattr(self.chat_interface, 'granite_client'))
        self.assertTrue(hasattr(self.chat_interface, 'context_manager'))

    def test_simple_conversation(self):
        """Test simple conversation flow."""
        # Mock client response
        self.mock_client.chat.return_value = "Hello! I'm here to help with your plants."

        # Test conversation
        response = self.chat_interface.respond("Hello")

        # Verify response
        self.assertIsInstance(response, str)
        self.assertIn("help", response)

        # Verify client was called
        self.mock_client.chat.assert_called_once_with("Hello")

    def test_context_management(self):
        """Test context management in conversations."""
        # Mock client response
        self.mock_client.chat_with_context.return_value = "Based on your analysis, your plant looks healthy!"

        # Add analysis context
        analysis_result = {
            "health_score": 0.9,
            "growth_stage": "flowering"
        }

        self.chat_interface.add_analysis_context(analysis_result)

        # Test context-aware response
        response = self.chat_interface.respond("How does my plant look?")

        # Verify context was used
        self.mock_client.chat_with_context.assert_called_once()
        call_args = self.mock_client.chat_with_context.call_args
        self.assertEqual(call_args[0][0], "How does my plant look?")
        self.assertIn("health_score", str(call_args[0][1]))

    def test_conversation_history(self):
        """Test conversation history tracking."""
        # Mock responses
        self.mock_client.chat.side_effect = [
            "Hello! How can I help?",
            "I can analyze your plant's health.",
            "You're welcome!"
        ]

        # Have conversation
        responses = []
        questions = ["Hi", "What can you do?", "Thanks"]

        for question in questions:
            response = self.chat_interface.respond(question)
            responses.append(response)

        # Check history
        history = self.chat_interface.get_conversation_history()
        self.assertEqual(len(history), 3)

        for i, exchange in enumerate(history):
            self.assertEqual(exchange["user"], questions[i])
            self.assertEqual(exchange["assistant"], responses[i])

    def test_quick_suggestions(self):
        """Test quick suggestion functionality."""
        # Test getting suggestions
        suggestions = self.chat_interface.get_quick_suggestions()

        # Should return list of suggestions
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

        # Each suggestion should be a string
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, str)

    def test_analysis_integration(self):
        """Test integration with analysis results."""
        # Mock analysis result
        analysis_result = {
            "health_analysis": {
                "health_score": 0.75,
                "disease_probability": 0.25,
                "stress_indicators": ["water_stress"]
            },
            "growth_analysis": {
                "growth_stage": "vegetative"
            },
            "recommendations": ["Increase watering", "Check soil drainage"]
        }

        # Mock client response
        self.mock_client.chat_with_context.return_value = "I see your plant has water stress. Here's what you should do..."

        # Add analysis and ask question
        self.chat_interface.add_analysis_context(analysis_result)
        response = self.chat_interface.respond("What should I do about the issues?")

        # Verify integration
        self.assertIsInstance(response, str)
        self.mock_client.chat_with_context.assert_called_once()

    def test_fallback_responses(self):
        """Test fallback responses when AI is unavailable."""
        # Mock client failure
        self.mock_client.chat.side_effect = Exception("AI service unavailable")

        # Test fallback
        response = self.chat_interface.respond("Help me with my plant")

        # Should return fallback response
        self.assertIsInstance(response, str)
        self.assertIn("trouble", response.lower())

    def test_agricultural_knowledge_base(self):
        """Test built-in agricultural knowledge base."""
        # Test offline agricultural responses
        agricultural_questions = [
            "What causes yellow leaves?",
            "How often should I water tomatoes?",
            "What is nitrogen deficiency?"
        ]

        for question in agricultural_questions:
            with self.subTest(question=question):
                # Mock client failure to test fallback knowledge
                self.mock_client.chat.side_effect = Exception("Network error")

                response = self.chat_interface.respond(question)

                # Should provide some agricultural guidance
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 20)  # Substantial response


class TestContextManager(unittest.TestCase):
    """Test context manager functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.context_manager = ContextManager(storage_dir=self.temp_dir)

    def test_context_storage(self):
        """Test context storage and retrieval."""
        # Add analysis context
        analysis_result = {
            "timestamp": "2024-01-01T12:00:00",
            "health_score": 0.8,
            "growth_stage": "flowering"
        }

        context_id = self.context_manager.add_analysis_context(analysis_result)

        # Retrieve context
        retrieved = self.context_manager.get_context(context_id)

        # Verify storage and retrieval
        self.assertIsNotNone(context_id)
        self.assertEqual(retrieved["health_score"], 0.8)
        self.assertEqual(retrieved["growth_stage"], "flowering")

    def test_context_history(self):
        """Test context history management."""
        # Add multiple contexts
        contexts = []
        for i in range(5):
            context = {
                "timestamp": f"2024-01-0{i+1}T12:00:00",
                "health_score": 0.5 + (i * 0.1),
                "session": f"session_{i}"
            }
            context_id = self.context_manager.add_analysis_context(context)
            contexts.append(context_id)

        # Get recent contexts
        recent = self.context_manager.get_recent_contexts(3)

        # Should return most recent 3
        self.assertEqual(len(recent), 3)

        # Should be in reverse chronological order
        for i in range(len(recent)-1):
            self.assertGreater(recent[i]["timestamp"], recent[i+1]["timestamp"])

    def test_context_search(self):
        """Test context search functionality."""
        # Add searchable contexts
        contexts = [
            {"plant_type": "tomato", "health_score": 0.8},
            {"plant_type": "pepper", "health_score": 0.6},
            {"plant_type": "tomato", "health_score": 0.9}
        ]

        for context in contexts:
            self.context_manager.add_analysis_context(context)

        # Search for tomato contexts
        tomato_contexts = self.context_manager.search_contexts(plant_type="tomato")

        # Should find 2 tomato contexts
        self.assertEqual(len(tomato_contexts), 2)
        for context in tomato_contexts:
            self.assertEqual(context["plant_type"], "tomato")

    def test_context_export(self):
        """Test context export functionality."""
        # Add some contexts
        for i in range(3):
            context = {
                "timestamp": f"2024-01-0{i+1}T12:00:00",
                "health_score": 0.5 + (i * 0.2)
            }
            self.context_manager.add_analysis_context(context)

        # Export contexts
        export_file = self.temp_dir / "exported_contexts.json"
        self.context_manager.export_contexts(export_file)

        # Verify export
        self.assertTrue(export_file.exists())

        with open(export_file, 'r') as f:
            exported_data = json.load(f)

        self.assertIn("contexts", exported_data)
        self.assertEqual(len(exported_data["contexts"]), 3)

    def test_context_cleanup(self):
        """Test context cleanup functionality."""
        # Add old contexts
        old_contexts = []
        for i in range(15):  # More than default max
            context = {
                "timestamp": f"2024-01-{i+1:02d}T12:00:00",
                "health_score": 0.5
            }
            context_id = self.context_manager.add_analysis_context(context)
            old_contexts.append(context_id)

        # Trigger cleanup
        self.context_manager.cleanup_old_contexts(max_contexts=10)

        # Should keep only 10 most recent
        remaining = self.context_manager.get_all_contexts()
        self.assertLessEqual(len(remaining), 10)


class TestAIAssistantIntegration(unittest.TestCase):
    """Test complete AI assistant integration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    @patch('src.ai_assistant.granite_client.GraniteClient')
    def test_full_integration_workflow(self, mock_granite):
        """Test complete integration workflow."""
        # Mock Granite client
        mock_client = Mock()
        mock_granite.return_value = mock_client
        mock_client.chat_with_context.return_value = "Your plant analysis shows excellent health!"

        # Initialize components
        chat_interface = ChatInterface()

        # Simulate analysis result
        analysis_result = {
            "health_analysis": {"health_score": 0.95},
            "growth_analysis": {"growth_stage": "flowering"},
            "recommendations": ["Continue current care"]
        }

        # Add context and chat
        chat_interface.add_analysis_context(analysis_result)
        response = chat_interface.respond("How is my plant doing?")

        # Verify full workflow
        self.assertIsInstance(response, str)
        self.assertIn("health", response)
        mock_client.chat_with_context.assert_called_once()

    def test_error_recovery(self):
        """Test error recovery in full system."""
        with patch('src.ai_assistant.granite_client.GraniteClient') as mock_granite:
            # Mock client that fails
            mock_client = Mock()
            mock_granite.return_value = mock_client
            mock_client.chat.side_effect = Exception("Service error")

            # Initialize and test
            chat_interface = ChatInterface()
            response = chat_interface.respond("Test question")

            # Should recover gracefully
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)
