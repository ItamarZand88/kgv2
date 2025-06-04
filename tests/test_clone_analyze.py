#!/usr/bin/env python3
"""
Test script for clone and analyze functions
"""

import asyncio
import uuid
import shutil
import os
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from storage import StorageManager, GraphManager
    from core import CodeAnalyzer
    from server import clone_repository  # Import the clone function
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

async def test_clone(git_url: str, target_dir: str):
    """Test the clone functionality"""
    print(f"\n=== Testing Clone: {git_url} ===")
    
    # Clean up if directory exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    
    start_time = datetime.now()
    success = await clone_repository(git_url, target_dir)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    if success:
        print(f"✅ Clone successful in {duration:.2f} seconds")
        
        # Check what was cloned
        if os.path.exists(target_dir):
            files = []
            for root, dirs, filenames in os.walk(target_dir):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
            
            print(f"📁 Cloned {len(files)} files")
            
            # Show some Python files
            py_files = [f for f in files if f.endswith('.py')]
            print(f"🐍 Found {len(py_files)} Python files")
            
            if py_files:
                print("First few Python files:")
                for f in py_files[:5]:
                    rel_path = os.path.relpath(f, target_dir)
                    print(f"  - {rel_path}")
        
        return True
    else:
        print(f"❌ Clone failed after {duration:.2f} seconds")
        return False

async def test_analyze(repo_path: str, repo_id: str):
    """Test the analyze functionality"""
    print(f"\n=== Testing Analysis: {repo_path} ===")
    
    try:
        # Initialize components
        storage = StorageManager()
        graph_manager = GraphManager(storage)
        analyzer = CodeAnalyzer(storage, graph_manager)
        
        # Ensure repo directory
        storage.ensure_repo_dir(repo_id)
        
        start_time = datetime.now()
        result = await analyzer.analyze_repository(repo_path, repo_id)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Analysis completed in {duration:.2f} seconds")
        print(f"📊 Results:")
        print(f"  - Nodes created: {result.get('nodes_created', 0)}")
        print(f"  - Edges created: {result.get('edges_created', 0)}")
        print(f"  - Files analyzed: {result.get('files_analyzed', 0)}")
        
        if result.get('errors'):
            print(f"⚠️ Errors: {len(result['errors'])}")
            for error in result['errors'][:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

async def test_full_pipeline(git_url: str):
    """Test the full clone + analyze pipeline"""
    repo_id = str(uuid.uuid4())
    temp_dir = f"temp_test_{repo_id}"
    
    print(f"\n{'='*60}")
    print(f"🧪 FULL PIPELINE TEST")
    print(f"📁 Repo ID: {repo_id}")
    print(f"🌐 Git URL: {git_url}")
    print(f"📂 Temp Dir: {temp_dir}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Clone
        clone_success = await test_clone(git_url, temp_dir)
        
        if not clone_success:
            print("❌ Stopping - Clone failed")
            return
        
        # Step 2: Analyze
        analysis_result = await test_analyze(temp_dir, repo_id)
        
        if analysis_result:
            print(f"\n✅ Full pipeline successful!")
        else:
            print(f"\n❌ Analysis step failed")
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            print(f"\n🧹 Cleaning up {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

async def test_local_repo():
    """Test analysis with local repo"""
    repo_path = "./test_repo"
    repo_id = f"local_test_{uuid.uuid4()}"
    
    print(f"\n{'='*60}")
    print(f"🏠 LOCAL REPO TEST")
    print(f"📁 Repo ID: {repo_id}")
    print(f"📂 Local Path: {repo_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(repo_path):
        print(f"❌ Local repo path {repo_path} doesn't exist")
        return
    
    result = await test_analyze(repo_path, repo_id)
    
    if result:
        print(f"✅ Local analysis successful!")
    else:
        print(f"❌ Local analysis failed")

async def main():
    """Main test function"""
    print("🧪 MCP Clone & Analyze Tester")
    print("="*60)
    
    # Test with local repo first
    await test_local_repo()
    
    # Test with small remote repos
    test_repos = [
        "https://github.com/octocat/Hello-World",  # Very small GitHub repo
        "https://github.com/microsoft/vscode-mcp",  # MCP example (if exists)
    ]
    
    for repo_url in test_repos:
        try:
            await test_full_pipeline(repo_url)
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            continue

if __name__ == "__main__":
    print("Starting tests...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
    
    print("\n�� Tests completed!") 