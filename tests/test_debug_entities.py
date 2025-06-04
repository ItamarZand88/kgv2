#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import CodeAnalyzer
from storage import StorageManager, GraphManager

async def test_debug():
    """Test debug for global entities registration"""
    
    print("🔧 DEBUG: Testing global entities registration")
    print("=" * 60)
    
    storage = StorageManager('')
    graph_mgr = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_mgr)
    
    # Test on our simple test repo
    test_path = "./test_repo"
    repo_id = "debug_test_12345"
    
    print(f"📁 Analyzing: {test_path}")
    print(f"🆔 Repo ID: {repo_id}")
    
    result = await analyzer.analyze_repository(test_path, repo_id)
    
    print("\n" + "=" * 60)
    print("🔍 FINAL RESULTS:")
    print(f"✅ Nodes created: {result['nodes_created']}")
    print(f"✅ Edges created: {result['edges_created']}")
    print(f"✅ Files analyzed: {result['files_analyzed']}")
    
    if result.get('errors'):
        print(f"❌ Errors: {len(result['errors'])}")
        for error in result['errors'][:3]:
            print(f"   - {error}")
    
    print("\n" + "=" * 60)
    print("🔧 DEBUG: Global state at end:")
    print(f"   - Global entities: {len(analyzer.global_entities)}")
    print(f"   - Global imports: {len(analyzer.global_imports)}")
    print(f"   - Call relationships: {len(analyzer.call_relationships)}")
    
    if analyzer.global_entities:
        print("\n🎯 Global entities found:")
        for i, (key, entity) in enumerate(list(analyzer.global_entities.items())[:10]):
            print(f"   {i+1}. {key} -> {entity.name} ({entity.type})")
    else:
        print("\n❌ NO GLOBAL ENTITIES REGISTERED!")
        
    return result

if __name__ == "__main__":
    asyncio.run(test_debug()) 