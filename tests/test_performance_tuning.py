#!/usr/bin/env python3

import asyncio
import sys
import os
import time
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import CodeAnalyzer
from storage import StorageManager, GraphManager

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)

async def test_performance_steps():
    """Test each step of the analysis process to find bottlenecks"""
    
    print("🚀 Performance Tuning Test")
    print("=" * 60)
    
    storage = StorageManager('')
    graph_mgr = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_mgr)
    
    test_path = "./test_repo"
    repo_id = f"perf_test_{int(time.time())}"
    
    print(f"📁 Repository: {test_path}")
    print(f"🆔 Repo ID: {repo_id}")
    
    # Step 1: File discovery
    print("\n🔍 Step 1: File Discovery")
    start = time.time()
    files = await analyzer._find_source_files(test_path)
    step1_time = time.time() - start
    print(f"   ✅ Found {len(files)} files in {step1_time:.3f}s")
    
    # Step 2: File parsing (individual files)
    print("\n🔍 Step 2: File Parsing")
    start = time.time()
    all_nodes = []
    all_edges = []
    
    for file_path in files:
        relative_path = analyzer._get_relative_path(test_path, file_path)
        nodes, edges = await analyzer._analyze_file(file_path, relative_path, repo_id)
        all_nodes.extend(nodes)
        all_edges.extend(edges)
    
    step2_time = time.time() - start
    print(f"   ✅ Parsed {len(files)} files in {step2_time:.3f}s")
    print(f"   📊 Found {len(all_nodes)} nodes, {len(all_edges)} edges")
    print(f"   📊 Global entities: {len(analyzer.global_entities)}")
    
    # Step 3: NetworkX graph building
    print("\n🔍 Step 3: NetworkX Graph Building")
    start = time.time()
    analyzer._build_networkx_graph(all_nodes, all_edges)
    step3_time = time.time() - start
    print(f"   ✅ Built NetworkX graph in {step3_time:.3f}s")
    print(f"   📊 Nodes: {analyzer.nx_graph.number_of_nodes()}")
    print(f"   📊 Edges: {analyzer.nx_graph.number_of_edges()}")
    
    # Step 4: PageRank calculation
    print("\n🔍 Step 4: PageRank Calculation")
    start = time.time()
    ranked_entities = analyzer._apply_pagerank_ranking()
    step4_time = time.time() - start
    print(f"   ✅ PageRank completed in {step4_time:.3f}s")
    print(f"   📊 Ranked {len(ranked_entities)} entities")
    
    # Step 5: Cross-file relationship analysis
    print("\n🔍 Step 5: Cross-file Relationships")
    start = time.time()
    cross_file_edges = analyzer._create_cross_file_edges_with_ranking(ranked_entities)
    step5_time = time.time() - start
    print(f"   ✅ Cross-file analysis in {step5_time:.3f}s")
    print(f"   📊 Created {len(cross_file_edges)} cross-file edges")
    
    # Step 6: Database persistence (this might be the bottleneck!)
    print("\n🔍 Step 6: Database Persistence")
    start = time.time()
    
    # Create a mock results object
    results = {
        'nodes_created': 0,
        'edges_created': 0,
        'errors': []
    }
    
    await analyzer._persist_analysis_results(repo_id, all_nodes, all_edges + cross_file_edges, results)
    step6_time = time.time() - start
    print(f"   ✅ Persisted to database in {step6_time:.3f}s")
    print(f"   📊 Stored {results['nodes_created']} nodes, {results['edges_created']} edges")
    
    # Summary
    total_time = step1_time + step2_time + step3_time + step4_time + step5_time + step6_time
    
    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"1. File Discovery:       {step1_time:.3f}s ({step1_time/total_time*100:.1f}%)")
    print(f"2. File Parsing:         {step2_time:.3f}s ({step2_time/total_time*100:.1f}%)")
    print(f"3. NetworkX Building:    {step3_time:.3f}s ({step3_time/total_time*100:.1f}%)")
    print(f"4. PageRank:             {step4_time:.3f}s ({step4_time/total_time*100:.1f}%)")
    print(f"5. Cross-file Analysis:  {step5_time:.3f}s ({step5_time/total_time*100:.1f}%)")
    print(f"6. Database Persistence: {step6_time:.3f}s ({step6_time/total_time*100:.1f}%)")
    print(f"   TOTAL:                {total_time:.3f}s")
    
    # Identify bottlenecks
    times = [
        ("File Discovery", step1_time),
        ("File Parsing", step2_time), 
        ("NetworkX Building", step3_time),
        ("PageRank", step4_time),
        ("Cross-file Analysis", step5_time),
        ("Database Persistence", step6_time)
    ]
    
    # Find the slowest step
    slowest_step, slowest_time = max(times, key=lambda x: x[1])
    
    print(f"\n🐌 BOTTLENECK: {slowest_step} ({slowest_time:.3f}s)")
    
    if slowest_step == "Database Persistence":
        print("   💡 Suggestion: Optimize batch sizes or use bulk operations")
    elif slowest_step == "PageRank":
        print("   💡 Suggestion: Skip PageRank for large repos or use approximation")
    elif slowest_step == "Cross-file Analysis":
        print("   💡 Suggestion: Limit cross-file analysis scope")
    elif slowest_step == "File Parsing":
        print("   💡 Suggestion: Increase parallelism or filter files")

if __name__ == "__main__":
    asyncio.run(test_performance_steps()) 