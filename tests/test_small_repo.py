#!/usr/bin/env python3

import asyncio
import sys
import os
import signal
from typing import Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import CodeAnalyzer
from storage import StorageManager, GraphManager

class TimeoutError(Exception):
    """Raised when operation times out"""
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

async def test_with_timeout(repo_path: str, repo_id: str, timeout_seconds: int = 30):
    """Test repository analysis with timeout protection"""
    
    print(f"⏰ Starting analysis with {timeout_seconds}s timeout")
    print(f"📁 Repository: {repo_path}")
    print(f"🆔 Repo ID: {repo_id}")
    print("=" * 60)
    
    # Set up timeout protection for Windows (different approach needed)
    storage = StorageManager('')
    graph_mgr = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_mgr)
    
    try:
        # Use asyncio.wait_for for timeout
        result = await asyncio.wait_for(
            analyzer.analyze_repository(repo_path, repo_id),
            timeout=timeout_seconds
        )
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED!")
        print("=" * 60)
        print(f"✅ Nodes created: {result['nodes_created']}")
        print(f"✅ Edges created: {result['edges_created']}")
        print(f"✅ Files analyzed: {result['files_analyzed']}")
        
        if result.get('errors'):
            print(f"⚠️ Errors: {len(result['errors'])}")
            for error in result['errors'][:3]:
                print(f"   - {error}")
        
        print(f"\n🔧 Global entities: {len(analyzer.global_entities)}")
        print(f"🔧 Global imports: {len(analyzer.global_imports)}")
        print(f"🔧 Call relationships: {len(analyzer.call_relationships)}")
        
        return result
        
    except asyncio.TimeoutError:
        print(f"\n❌ TIMEOUT! Analysis took longer than {timeout_seconds} seconds")
        print("🔧 This might indicate:")
        print("   - Repository is too large")
        print("   - Infinite loop in analysis")
        print("   - Blocking I/O operations")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None

async def test_progression():
    """Test on progressively larger repositories"""
    
    print("🧪 Progressive Repository Testing")
    print("=" * 60)
    
    # Test 1: Our tiny test repo (should work fast)
    print("\n🔸 Test 1: Tiny repository (1 file)")
    result1 = await test_with_timeout("./test_repo", "tiny_test", 10)
    
    if not result1:
        print("❌ Tiny repository failed - something is seriously wrong!")
        return
    
    print("✅ Tiny repository passed!")
    
    # Test 2: Try with a small cloned repo
    print("\n🔸 Test 2: Small cloned repository")
    
    # Let's clone something really small
    from server import clone_repository
    
    try:
        small_repo_path = "small_test_repo"
        success = await clone_repository(
            "https://github.com/gvanrossum/pep8.git",  # Small Python repo
            small_repo_path
        )
        
        if not success:
            print("❌ Failed to clone repository")
            return
        
        result2 = await test_with_timeout(small_repo_path, "small_test", 30)
        
        if result2:
            print("✅ Small repository passed!")
        else:
            print("❌ Small repository failed/timed out")
            
    except Exception as e:
        print(f"❌ Could not clone small repo: {e}")
    
    print("\n🎯 Testing complete!")

if __name__ == "__main__":
    asyncio.run(test_progression()) 