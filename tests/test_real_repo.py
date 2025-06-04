#!/usr/bin/env python3
"""
Test with real Python repositories
"""

import asyncio
import uuid
import shutil
import os
from test_clone_analyze import test_full_pipeline, test_clone, test_analyze

async def test_real_repos():
    """Test with real Python repositories"""
    
    # Small Python repositories
    test_repos = [
        "https://github.com/psf/requests",  # Popular but might be large
        "https://github.com/kennethreitz/requests",  # Alternative
        "https://github.com/pytoolz/toolz",  # Small functional library
        "https://github.com/mahmoud/boltons",  # Utility library
        "https://github.com/grantjenks/python-sortedcontainers",  # Small and clean
    ]
    
    print("🐍 Testing with real Python repositories")
    print("="*60)
    
    for repo_url in test_repos:
        print(f"\n⏯️ Testing: {repo_url}")
        try:
            await test_full_pipeline(repo_url)
            
            # Just test one successful repo for now
            break
            
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted")
            break
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(test_real_repos()) 