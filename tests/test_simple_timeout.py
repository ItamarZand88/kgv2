#!/usr/bin/env python3

import asyncio
import sys
import os
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import CodeAnalyzer
from storage import StorageManager, GraphManager

async def simple_timeout_test():
    """Simple test with progress reporting"""
    
    print("🔧 Simple Timeout Test")
    print("=" * 60)
    
    storage = StorageManager('')
    graph_mgr = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_mgr)
    
    # Test on our tiny repo
    test_path = "./test_repo"
    repo_id = f"timeout_test_{int(time.time())}"
    
    print(f"📁 Testing on: {test_path}")
    print(f"🆔 Repo ID: {repo_id}")
    
    start_time = time.time()
    
    try:
        # Run with a very short timeout first
        print("\n⏰ Starting analysis with 5 second timeout...")
        result = await asyncio.wait_for(
            analyzer.analyze_repository(test_path, repo_id),
            timeout=5.0
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS! Completed in {elapsed:.2f} seconds")
        print(f"   - Nodes: {result['nodes_created']}")
        print(f"   - Edges: {result['edges_created']}")
        print(f"   - Files: {result['files_analyzed']}")
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"\n❌ TIMEOUT after {elapsed:.2f} seconds")
        print("   - The process is hanging somewhere...")
        
        # Try to analyze what might be causing the hang
        print("\n🔍 Potential causes:")
        print("   1. Blocking I/O in file operations")
        print("   2. Infinite loop in parsing")
        print("   3. Database operations taking too long")
        print("   4. NetworkX algorithms hanging")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR after {elapsed:.2f} seconds: {e}")

if __name__ == "__main__":
    asyncio.run(simple_timeout_test()) 