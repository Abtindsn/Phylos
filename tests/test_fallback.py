import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import graph_builder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies globally before importing graph_builder
# We need to assign these to variables to keep them alive if needed, 
# but primarily we need them in sys.modules
mock_genai = MagicMock()
mock_langgraph = MagicMock()
mock_langgraph_graph = MagicMock()
mock_dotenv = MagicMock()
mock_bs4 = MagicMock()
mock_requests = MagicMock()
mock_pydantic = MagicMock()

sys.modules['google.generativeai'] = mock_genai
sys.modules['langgraph'] = mock_langgraph
sys.modules['langgraph.graph'] = mock_langgraph_graph
sys.modules['dotenv'] = mock_dotenv
sys.modules['bs4'] = mock_bs4
sys.modules['requests'] = mock_requests
sys.modules['pydantic'] = mock_pydantic

import graph_builder

class TestFallbackLogic(unittest.TestCase):
    def setUp(self):
        # Ensure USE_GEMINI is True for testing
        graph_builder.USE_GEMINI = True
        graph_builder.logger = MagicMock()

    def test_fallback_success_after_failures(self):
        """Test that _call_with_fallback retries on 429/404 and eventually succeeds."""
        
        mock_func = MagicMock()
        # First call: 429 Quota Exceeded
        # Second call: 404 Not Found
        # Third call: Success
        mock_func.side_effect = [
            Exception("429 Quota Exceeded"),
            Exception("404 Not Found"),
            "Success"
        ]

        # Call the function
        result = graph_builder._call_with_fallback("TestOperation", mock_func, prompt="test")

        # Assertions
        self.assertEqual(result, "Success")
        self.assertEqual(mock_func.call_count, 3)
        
        # Verify it tried different models
        # The first call might use the default or first fallback, subsequent calls iterate
        # We just want to ensure it kept trying
        
    def test_fallback_failure_all_models(self):
        """Test that it raises exception if all models fail."""
        mock_func = MagicMock()
        mock_func.side_effect = Exception("429 Quota Exceeded") # Always fail

        with self.assertRaises(Exception) as cm:
            graph_builder._call_with_fallback("TestOperation", mock_func, prompt="test")
        
        self.assertIn("429", str(cm.exception))
        # Should have tried all models in FALLBACK_MODELS
        self.assertEqual(mock_func.call_count, len(graph_builder.FALLBACK_MODELS))

    @patch('graph_builder.genai')
    def test_embedder_fallback(self, mock_genai):
        """Test that embedder uses fallback logic."""
        # This is harder to test directly without mocking the inner function of embedder
        # but we can test _call_with_fallback with genai.embed_content
        
        mock_embed = MagicMock()
        mock_embed.side_effect = [
            Exception("429 Quota Exceeded"),
            {"embedding": [0.1, 0.2]}
        ]
        
        result = graph_builder._call_with_fallback("Embedding", mock_embed, content="test")
        self.assertEqual(result, {"embedding": [0.1, 0.2]})
        self.assertEqual(mock_embed.call_count, 2)

if __name__ == '__main__':
    unittest.main()
