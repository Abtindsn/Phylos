
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

class TestDeepAnalysis(unittest.TestCase):

    @patch('graph_builder.generate_text_response')
    def test_deep_analysis_prompt_trigger(self, mock_generate):
        # Test that high drift triggers the deep analysis prompt
        graph_builder.summarize_mutation(
            "Parent Content", 
            "Child Content", 
            "http://parent.com", 
            "http://child.com", 
            domains_related=False, 
            drift_score=0.95
        )
        
        # Check if the prompt contains the deep analysis keywords
        args, _ = mock_generate.call_args
        prompt = args[0]
        self.assertIn("PERFORM A DEEP NARRATIVE ANALYSIS", prompt)
        self.assertIn("Structure your response exactly as follows", prompt)

    @patch('graph_builder.generate_text_response')
    def test_standard_prompt_trigger(self, mock_generate):
        # Test that low drift triggers the standard prompt
        graph_builder.summarize_mutation(
            "Parent Content", 
            "Child Content", 
            "http://parent.com", 
            "http://child.com", 
            domains_related=True, 
            drift_score=0.4
        )
        
        args, _ = mock_generate.call_args
        prompt = args[0]
        self.assertNotIn("PERFORM A DEEP NARRATIVE ANALYSIS", prompt)
        self.assertIn("Concisely describe the mutation", prompt)

if __name__ == '__main__':
    unittest.main()
