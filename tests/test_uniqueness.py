
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to sys.path
sys.path.append('/home/abtin/Phylos')

# Mock environment variables
os.environ['PHYLOS_ECO_MODE'] = 'false'
os.environ['GEMINI_API_KEY'] = 'fake_key'

# Mock external dependencies
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()
sys.modules['langgraph.graph'].StateGraph = MagicMock()
sys.modules['langgraph.graph'].END = "END"

sys.modules['google'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['bs4'] = MagicMock()
sys.modules['pydantic'] = MagicMock()

import graph_builder

class TestUniqueness(unittest.TestCase):

    @patch('graph_builder.llm')
    def test_unique_fallback_on_error(self, mock_llm):
        # Setup mock to raise an exception (simulating 404 or other error)
        mock_llm.generate_content.side_effect = Exception("404 Model not found")
        
        # Define two different pairs of content
        parent1 = "The quick brown fox jumps over the lazy dog."
        child1 = "The quick brown fox jumps over the active cat."
        
        parent2 = "Python is a programming language."
        child2 = "Python is a snake."
        
        # Generate summaries
        summary1 = graph_builder.summarize_mutation(parent1, child1, "url1", "url2")
        summary2 = graph_builder.summarize_mutation(parent2, child2, "url3", "url4")
        
        print(f"Summary 1: {summary1}")
        print(f"Summary 2: {summary2}")
        
        # Verify they are NOT equal (uniqueness check)
        self.assertNotEqual(summary1, summary2)
        
        # Verify they contain the offline summary content (fallback)
        self.assertIn("Offline summary", summary1)
        self.assertIn("Offline summary", summary2)
        
        # Verify they do NOT contain the static error message
        self.assertNotIn("AI Error", summary1)
        self.assertNotIn("AI Error", summary2)

if __name__ == '__main__':
    unittest.main()
