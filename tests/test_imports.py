#!/usr/bin/env python3
"""Test script to check which imports are failing."""

import sys

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def main():
    print("🔍 Testing imports...")
    
    # Test all imports needed for the unified server
    modules_to_test = [
        "mcp",
        "mcp.server",
        "mcp.server.fastmcp",
        "fastapi",
        "networkx",
        "models",
        "storage", 
        "core",
        "search",
        "context"
    ]
    
    successful = 0
    total = len(modules_to_test)
    
    for module in modules_to_test:
        if test_import(module):
            successful += 1
    
    print(f"\n📊 Results: {successful}/{total} modules imported successfully")
    
    if successful == total:
        print("🎉 All imports successful!")
        print("✨ The unified Knowledge Graph MCP server is ready to run!")
    else:
        print("❌ Some imports failed. Please check the errors above.")
        
        # Try to give specific guidance
        missing_modules = []
        if not test_import("mcp"):
            missing_modules.append("mcp")
        
        if not test_import("fastapi"):
            missing_modules.append("fastapi")
            
        if not test_import("networkx"):
            missing_modules.append("networkx")
        
        if missing_modules:
            print(f"\n💡 Install missing packages: pip install {' '.join(missing_modules)}")

if __name__ == "__main__":
    main() 