
import unittest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock environment variables
os.environ['PHYLOS_ECO_MODE'] = 'true'

import graph_builder
import server

class TestIndicator(unittest.TestCase):

    def test_graph_builder_stub_indicator(self):
        summary = graph_builder._stub_summary("content A", "content B")
        self.assertIn("[Non-AI Analysis]", summary)
        print(f"GraphBuilder Stub: {summary}")

    def test_server_fallback_indicator(self):
        summary, hidden = server._fallback_origin_summary("content A", "content B", "Title")
        self.assertIn("[Non-AI Analysis]", summary)
        print(f"Server Fallback: {summary}")

if __name__ == '__main__':
    unittest.main()
