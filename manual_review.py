
import sys
import os
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append('/home/abtin/Phylos')

# Load env
load_dotenv('/home/abtin/Phylos/.env')

import graph_builder

def fetch_and_print(url):
    print(f"\n\n=== FETCHING: {url} ===")
    data, _ = graph_builder.fetch_article_content(url)
    if data:
        print(f"TITLE: {data.get('content', '').splitlines()[0]}") # Assuming title is first line
        print("-" * 20)
        # Print first 2000 chars to get the gist without overwhelming logs
        print(data.get('content', '')[:2000]) 
        print("-" * 20)
    else:
        print("FAILED TO FETCH")

if __name__ == "__main__":
    urls = [
        "https://www.bbc.com/news/articles/cn09n94qg92o",
        "https://www.cnbc.com/2025/11/13/goldman-sachs-jeffrey-epstein-emails-ruemmler.html"
    ]
    for url in urls:
        fetch_and_print(url)
