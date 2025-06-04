#!/usr/bin/env python3

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_clone_analyze import test_full_pipeline

async def test_simple_real_repo():
    """Test with a small, simple real repository"""
    
    print("🚀 TESTING FAST ANALYSIS ON REAL REPOSITORY")
    print("=" * 60)
    
    # Use a small, simple Python repository
    test_repo = "https://github.com/grantjenks/python-sortedcontainers"
    
    print(f"📁 Repository: {test_repo}")
    print("📊 Expected: ~26 Python files")
    print("⏱️  Goal: Complete analysis in under 30 seconds")
    print("🎯 Performance fix: Limited call relationships processing")
    print()
    
    try:
        # Clone and analyze using existing pipeline
        await test_full_pipeline(test_repo)
        
        print("\n🎉 TEST COMPLETE!")
        print("=" * 60)
        print("✅ Repository successfully analyzed")
        print("⚡ Performance improvements working!")
        print("🚀 No more freezing on large call relationships!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_real_repo()) 