#!/usr/bin/env python3
"""
Performance benchmark for repository analysis
"""

import asyncio
import time
import uuid
import shutil
import os
from datetime import datetime, timedelta
from test_clone_analyze import test_clone, test_analyze

async def benchmark_repository(repo_url: str, repo_name: str):
    """Benchmark repository analysis performance"""
    
    print(f"\n{'='*80}")
    print(f"🏁 PERFORMANCE BENCHMARK: {repo_name}")
    print(f"📍 URL: {repo_url}")
    print(f"{'='*80}")
    
    repo_id = str(uuid.uuid4())
    temp_dir = f"bench_{repo_id}"
    
    total_start = time.time()
    
    try:
        # Phase 1: Clone
        print(f"\n📥 Phase 1: Cloning...")
        clone_start = time.time()
        
        success = await test_clone(repo_url, temp_dir)
        if not success:
            print("❌ Clone failed, skipping benchmark")
            return None
        
        clone_time = time.time() - clone_start
        print(f"✅ Clone completed in {clone_time:.2f} seconds")
        
        # Count files
        total_files = 0
        py_files = 0
        for root, dirs, files in os.walk(temp_dir):
            total_files += len(files)
            py_files += len([f for f in files if f.endswith('.py')])
        
        print(f"📊 Repository stats:")
        print(f"   - Total files: {total_files}")
        print(f"   - Python files: {py_files}")
        
        # Phase 2: Analysis
        print(f"\n🔍 Phase 2: Analysis...")
        analysis_start = time.time()
        
        result = await test_analyze(temp_dir, repo_id)
        if not result:
            print("❌ Analysis failed")
            return None
        
        analysis_time = time.time() - analysis_start
        total_time = time.time() - total_start
        
        # Calculate metrics
        files_per_second = result['files_analyzed'] / analysis_time if analysis_time > 0 else 0
        nodes_per_second = result['nodes_created'] / analysis_time if analysis_time > 0 else 0
        
        print(f"\n📈 PERFORMANCE RESULTS:")
        print(f"   ⏱️  Times:")
        print(f"      - Clone time: {clone_time:.2f}s")
        print(f"      - Analysis time: {analysis_time:.2f}s")
        print(f"      - Total time: {total_time:.2f}s")
        print(f"   📊 Analysis Results:")
        print(f"      - Files analyzed: {result['files_analyzed']}")
        print(f"      - Nodes created: {result['nodes_created']}")
        print(f"      - Edges created: {result['edges_created']}")
        print(f"   🚀 Performance:")
        print(f"      - Files/second: {files_per_second:.2f}")
        print(f"      - Nodes/second: {nodes_per_second:.2f}")
        print(f"      - Seconds/file: {analysis_time/result['files_analyzed'] if result['files_analyzed'] > 0 else 0:.2f}")
        
        if result.get('errors'):
            print(f"   ⚠️  Errors: {len(result['errors'])}")
        
        return {
            'repo_name': repo_name,
            'clone_time': clone_time,
            'analysis_time': analysis_time,
            'total_time': total_time,
            'files_analyzed': result['files_analyzed'],
            'nodes_created': result['nodes_created'],
            'edges_created': result['edges_created'],
            'files_per_second': files_per_second,
            'nodes_per_second': nodes_per_second,
            'errors': len(result.get('errors', []))
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            print(f"\n🧹 Cleaning up {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

async def run_benchmark_suite():
    """Run benchmark on multiple repositories"""
    
    # Test repositories of different sizes - including larger ones
    test_repos = [
        ("https://github.com/grantjenks/python-sortedcontainers", "sortedcontainers (small)"),
        ("https://github.com/pytoolz/toolz", "toolz (medium)"),
        ("https://github.com/psf/requests", "requests (large)"),
        ("https://github.com/django/django", "django (huge)"),
    ]
    
    print("🏁 REPOSITORY ANALYSIS PERFORMANCE BENCHMARK")
    print("="*80)
    print("Testing with improved parallel processing...")
    
    results = []
    
    for repo_url, repo_name in test_repos:
        try:
            result = await benchmark_repository(repo_url, repo_name)
            if result:
                results.append(result)
                
        except KeyboardInterrupt:
            print("\n⏹️ Benchmark interrupted")
            break
        except Exception as e:
            print(f"❌ Error benchmarking {repo_name}: {e}")
            continue
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print("📊 BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Repository':<25} {'Files':<8} {'Nodes':<8} {'Time':<8} {'Files/s':<8} {'Nodes/s':<8}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['repo_name']:<25} {r['files_analyzed']:<8} {r['nodes_created']:<8} "
                  f"{r['analysis_time']:<8.1f} {r['files_per_second']:<8.1f} {r['nodes_per_second']:<8.0f}")
        
        # Calculate averages
        avg_files_per_sec = sum(r['files_per_second'] for r in results) / len(results)
        avg_nodes_per_sec = sum(r['nodes_per_second'] for r in results) / len(results)
        
        print(f"\n🎯 AVERAGE PERFORMANCE:")
        print(f"   - Files per second: {avg_files_per_sec:.2f}")
        print(f"   - Nodes per second: {avg_nodes_per_sec:.0f}")

if __name__ == "__main__":
    print("Starting performance benchmark...")
    try:
        asyncio.run(run_benchmark_suite())
    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrupted")
    except Exception as e:
        print(f"\n💥 Benchmark error: {e}")
    
    print("\n🏁 Benchmark completed!") 