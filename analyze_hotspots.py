
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# Add project root to sys.path
sys.path.append('/home/abtin/Phylos')

# Load env to ensure key is present
load_dotenv('/home/abtin/Phylos/.env')

# Import graph_builder after loading env
import graph_builder

def analyze(parent_url, child_url):
    print(f"\n--- Analyzing {parent_url} -> {child_url} ---")
    
    # Fetch content
    print("Fetching parent...")
    parent_data, _ = graph_builder.fetch_article_content(parent_url)
    if not parent_data:
        print("Failed to fetch parent.")
        return

    print("Fetching child...")
    child_data, _ = graph_builder.fetch_article_content(child_url)
    if not child_data:
        print("Failed to fetch child.")
        return

    # Embed
    print("Embedding...")
    parent_data["embedding"] = graph_builder.embedder(parent_data["content"])
    child_data["embedding"] = graph_builder.embedder(child_data["content"])

    # Compare
    drift_score = graph_builder.calculate_semantic_drift(parent_data["embedding"], child_data["embedding"])
    domains_related = graph_builder._are_domains_related(parent_url, child_url)
    
    print(f"Drift Score: {drift_score}")
    print(f"Domains Related: {domains_related}")
    
    # Logic from node_sequence
    divergence_reason = "Standard Drift"
    if drift_score > 0.90:
        if domains_related:
            divergence_reason = "Regional/Format Variation (Related Sources)"
        else:
            divergence_reason = "High Mutation Hotspot (Unrelated Sources)"
    elif domains_related and drift_score > 0.5:
            divergence_reason = "Routine Update/Variation (Related Sources)"
            
    print(f"Divergence Reason: {divergence_reason}")
    
    # Summary
    print("Generating Summary...")
    summary = graph_builder.summarize_mutation(
        parent_data["content"], 
        child_data["content"],
        parent_url=parent_url,
        child_url=child_url,
        domains_related=domains_related
    )
    print(f"Summary: {summary}")

if __name__ == "__main__":
    parent = "https://www.bbc.com/news/articles/cn09n94qg92o"
    children = [
        "https://www.bbc.co.uk/news/articles/c3rj0d97ynvo",
        "https://www.bbc.co.uk/news/articles/cddr199d3z0o",
        "https://www.cnbc.com/2025/11/13/goldman-sachs-jeffrey-epstein-emails-ruemmler.html"
    ]
    
    for child in children:
        analyze(parent, child)
