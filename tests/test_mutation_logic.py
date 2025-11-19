
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to sys.path
sys.path.append('/home/abtin/Phylos')

# Mock environment variables before importing graph_builder
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

# Import the module to test
import graph_builder

class TestMutationLogic(unittest.TestCase):

    def test_domain_relation(self):
        # Test the logic for detecting related domains (to be implemented)
        # For now, we just check if we can define the logic here or if we need to mock it
        pass

    @patch('graph_builder.calculate_semantic_drift')
    @patch('graph_builder.summarize_mutation')
    def test_node_sequence_high_divergence_related_domains(self, mock_summarize, mock_drift):
        # Setup
        mock_drift.return_value = 0.95 # High drift
        mock_summarize.return_value = "Summary"
        
        state = {
            "current_article": {
                "id": "https://www.bbc.co.uk/news/world-123",
                "content": "Content B",
                "embedding": [0.2] * 128
            },
            "parent_article_id": "https://www.bbc.com/news/world-123",
            "knowledge_graph": {
                "nodes": {
                    "https://www.bbc.com/news/world-123": {
                        "id": "https://www.bbc.com/news/world-123",
                        "content": "Content A",
                        "embedding": [0.1] * 128
                    }
                },
                "edges": []
            },
            "global_context": [0.0] * 128
        }

        # Execute
        # We need to see how node_sequence handles this. 
        # Currently it should just report high mutation.
        result = graph_builder.node_sequence(state)
        
        edge = result['knowledge_graph']['edges'][0]
        print(f"Edge attributes: {edge['attributes']}")
        
        # In the refined implementation, this should be a Major Variation with score 0.95
        self.assertEqual(edge['attributes']['mutation_score'], 0.95)
        self.assertEqual(edge['attributes']['relation_type'], "Major Variation")

    @patch('graph_builder._jaccard_drift')
    @patch('graph_builder._stub_summary')
    def test_node_sequence_eco_mode(self, mock_summary, mock_drift):
        # Test that Eco Mode runs without UnboundLocalError
        mock_drift.return_value = 0.1
        mock_summary.return_value = "Stub Summary"
        
        # Temporarily set ECO_MODE to True for this test
        with patch('graph_builder.ECO_MODE', True):
            state = {
                "current_article": {
                    "id": "https://example.com/2",
                    "content": "Content B",
                    "embedding": [0.2] * 128
                },
                "parent_article_id": "https://example.com/1",
                "knowledge_graph": {
                    "nodes": {
                        "https://example.com/1": {
                            "id": "https://example.com/1",
                            "content": "Content A",
                            "embedding": [0.1] * 128
                        }
                    },
                    "edges": []
                },
                "global_context": [0.0] * 128
            }
            
            result = graph_builder.node_sequence(state)
            edge = result['knowledge_graph']['edges'][0]
            
            # Check if it ran successfully and set default values
            self.assertEqual(edge['attributes']['relation_type'], "Replication")
            self.assertEqual(edge['attributes']['divergence_reason'], "Jaccard Drift (Eco Mode)")

    @patch('graph_builder.calculate_semantic_drift')
    def test_node_sequence_patient_zero(self, mock_drift):
        # Test Patient Zero case (no parent)
        mock_drift.return_value = 0.0
        
        state = {
            "current_article": {
                "id": "https://example.com/root",
                "content": "Root Content",
                "embedding": [0.1] * 128
            },
            "parent_article_id": None,
            "knowledge_graph": {
                "nodes": {},
                "edges": []
            },
            "global_context": [0.0] * 128
        }
        
        result = graph_builder.node_sequence(state)
        edge = result['knowledge_graph']['edges'][0]
        
        self.assertEqual(edge['attributes']['relation_type'], "Replication")
        self.assertEqual(edge['attributes']['divergence_reason'], "Patient Zero (Root)")

if __name__ == '__main__':
    unittest.main()
