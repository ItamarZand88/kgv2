#!/usr/bin/env python3
"""
Test script for counting files in large repositories
"""

import os
import subprocess
import tempfile
import shutil
from datetime import datetime

def count_files_in_repo(repo_url: str, repo_name: str):
    """Count files in a repository to understand its size"""
    
    print(f"\n{'='*60}")
    print(f"📊 FILE COUNT ANALYSIS: {repo_name}")
    print(f"📍 URL: {repo_url}")
    print(f"{'='*60}")
    
    temp_dir = f"temp_count_{repo_name.replace(' ', '_')}"
    
    try:
        # Clone repository (shallow)
        print(f"📥 Cloning {repo_name}...")
        start_time = datetime.now()
        
        result = subprocess.run([
            'git', 'clone', '--depth', '1', repo_url, temp_dir
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Clone failed: {result.stderr}")
            return
        
        clone_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Clone completed in {clone_time:.2f} seconds")
        
        # Count different file types
        file_counts = {
            'total': 0,
            'python': 0,
            'javascript': 0,
            'typescript': 0,
            'test_files': 0,
            'docs': 0,
            'config': 0,
            'other': 0
        }
        
        python_files = []
        large_files = []
        
        for root, dirs, files in os.walk(temp_dir):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            for file in files:
                file_path = os.path.join(root, file)
                file_counts['total'] += 1
                
                # Get file size
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > 50000:  # Files larger than 50KB
                        relative_path = os.path.relpath(file_path, temp_dir)
                        large_files.append((relative_path, file_size))
                except:
                    pass
                
                # Categorize by extension
                _, ext = os.path.splitext(file)
                ext = ext.lower()
                
                if ext == '.py':
                    file_counts['python'] += 1
                    relative_path = os.path.relpath(file_path, temp_dir)
                    python_files.append(relative_path)
                    
                    # Check if it's a test file
                    if 'test' in file.lower():
                        file_counts['test_files'] += 1
                        
                elif ext in ['.js', '.jsx']:
                    file_counts['javascript'] += 1
                elif ext in ['.ts', '.tsx']:
                    file_counts['typescript'] += 1
                elif ext in ['.md', '.rst', '.txt']:
                    file_counts['docs'] += 1
                elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']:
                    file_counts['config'] += 1
                else:
                    file_counts['other'] += 1
        
        # Print results
        print(f"\n📈 FILE STATISTICS:")
        print(f"   📁 Total files: {file_counts['total']}")
        print(f"   🐍 Python files: {file_counts['python']}")
        print(f"   📜 JavaScript files: {file_counts['javascript']}")
        print(f"   🔷 TypeScript files: {file_counts['typescript']}")
        print(f"   🧪 Test files: {file_counts['test_files']}")
        print(f"   📚 Documentation: {file_counts['docs']}")
        print(f"   ⚙️ Config files: {file_counts['config']}")
        print(f"   📄 Other files: {file_counts['other']}")
        
        # Show some Python files as examples
        if python_files:
            print(f"\n🐍 Sample Python files:")
            for i, py_file in enumerate(python_files[:10]):
                print(f"   {i+1:2d}. {py_file}")
            if len(python_files) > 10:
                print(f"   ... and {len(python_files) - 10} more")
        
        # Show large files
        if large_files:
            large_files.sort(key=lambda x: x[1], reverse=True)
            print(f"\n📋 Largest files (>50KB):")
            for i, (file_path, size) in enumerate(large_files[:5]):
                size_kb = size / 1024
                print(f"   {i+1}. {file_path} ({size_kb:.1f} KB)")
        
        # Estimate processing time
        processing_estimate = file_counts['python'] * 0.5  # 0.5 seconds per Python file
        print(f"\n⏱️ ESTIMATED PROCESSING:")
        print(f"   📊 Python files to analyze: {file_counts['python']}")
        print(f"   ⏱️ Estimated analysis time: {processing_estimate:.1f} seconds")
        print(f"   📈 With parallel processing: {processing_estimate/4:.1f} seconds")
        
        return file_counts
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            print(f"\n🧹 Cleaning up {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # Test different sized repositories
    test_repos = [
        ("https://github.com/grantjenks/python-sortedcontainers", "sortedcontainers"),
        ("https://github.com/pytoolz/toolz", "toolz"),
        ("https://github.com/psf/requests", "requests"),
        ("https://github.com/pallets/flask", "flask"),
        ("https://github.com/django/django", "django"),
        ("https://github.com/microsoft/vscode", "vscode"),
    ]
    
    print("🔍 REPOSITORY SIZE ANALYSIS")
    print("="*80)
    print("Analyzing file counts in popular repositories...")
    
    results = []
    
    for repo_url, repo_name in test_repos:
        try:
            result = count_files_in_repo(repo_url, repo_name)
            if result:
                results.append((repo_name, result))
        except KeyboardInterrupt:
            print("\n⏹️ Analysis interrupted")
            break
        except Exception as e:
            print(f"❌ Error analyzing {repo_name}: {e}")
            continue
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print("📊 REPOSITORY SIZE SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Repository':<15} {'Total':<8} {'Python':<8} {'JS/TS':<8} {'Tests':<8} {'Est. Time':<10}")
        print("-" * 80)
        
        for repo_name, counts in results:
            total = counts['total']
            python = counts['python']
            js_ts = counts['javascript'] + counts['typescript']
            tests = counts['test_files']
            est_time = python * 0.5
            
            print(f"{repo_name:<15} {total:<8} {python:<8} {js_ts:<8} {tests:<8} {est_time:<10.1f}s")
    
    print("\n🏁 Analysis completed!") 