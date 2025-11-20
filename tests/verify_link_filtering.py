import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_builder import _is_blocked_link, URL_PATH_BLOCKLIST

def test_link_filtering():
    print("Testing Link Filtering Logic...")
    
    blocked_urls = [
        "https://store.nytimes.com/collections/new-york-times-page-reprints",
        "https://help.nytimes.com/hc/en-us",
        "https://www.nytimes.com/privacy/cookie-policy",
        "https://www.nytimes.com/subscription",
        "https://myaccount.nytimes.com/auth/login",
        "https://www.nytimes.com/section/world?action=click&module=Ribbon&pgtype=Article", # Should ideally be blocked if we had query param blocking, but path blocking might miss this specific one if it's just /section/world. Wait, /section/world is a section front, maybe we want to allow it? The user complained about it.
        # Let's check if our path blocklist catches these:
        "https://nytimesarticles.store/cart",
        "https://www.nytimes.com/ads/marketing",
    ]
    
    allowed_urls = [
        "https://www.nytimes.com/1987/06/23/world/from-an-iranian-middleman-his-side-of-the-story.html",
        "https://www.bbc.com/news/world-middle-east-12345",
        "https://www.washingtonpost.com/politics/2023/10/01/article-name",
    ]

    print(f"\nBlocked Path Patterns: {URL_PATH_BLOCKLIST}\n")

    all_passed = True
    
    print("--- Checking Blocked URLs (Should be True) ---")
    for url in blocked_urls:
        is_blocked = _is_blocked_link(url)
        status = "✅ BLOCKED" if is_blocked else "❌ ALLOWED"
        print(f"{status}: {url}")
        if not is_blocked:
            all_passed = False

    print("\n--- Checking Allowed URLs (Should be False) ---")
    for url in allowed_urls:
        is_blocked = _is_blocked_link(url)
        status = "✅ ALLOWED" if not is_blocked else "❌ BLOCKED"
        print(f"{status}: {url}")
        if is_blocked:
            all_passed = False
            
    if all_passed:
        print("\n✅ All link filtering tests passed!")
    else:
        print("\n❌ Some link filtering tests failed.")

if __name__ == "__main__":
    test_link_filtering()
