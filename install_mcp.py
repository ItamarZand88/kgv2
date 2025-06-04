#!/usr/bin/env python3
"""
Installation script for Knowledge Graph MCP server.
This script registers the unified MCP server with the local MCP registry.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def install_mcp_server():
    """Install the unified MCP server"""
    
    # Check if mcp command is available
    if not run_command("mcp --help", "Checking MCP CLI availability"):
        print("❌ MCP CLI not found. Please install it first:")
        print("   pip install mcp")
        return False
    
    current_dir = Path.cwd()
    
    # Unified MCP server configuration
    server_config = {
        "file": "server.py",
        "name": "knowledge-graph-unified",
        "description": "Unified Knowledge Graph MCP Server (All-in-One)"
    }
    
    print("🚀 Installing Unified Knowledge Graph MCP Server...")
    print(f"📁 Working directory: {current_dir}")
    
    server_path = current_dir / server_config["file"]
    
    if not server_path.exists():
        print(f"⚠️  {server_config['file']} not found!")
        return False
        
    print(f"\n📦 Installing {server_config['description']}...")
    
    # Install the MCP server
    install_command = f"mcp install {server_path} --name {server_config['name']}"
    
    if run_command(install_command, f"Installing {server_config['name']}"):
        print(f"🎉 Installation complete! Unified MCP server installed successfully.")
        
        print("\n📋 Next steps:")
        print("1. Your unified MCP server is now installed and available")
        print("2. You can use it with MCP-compatible clients like Claude Desktop")
        print("3. Check server status with: mcp list")
        print("4. Start the server with: mcp run knowledge-graph-unified")
        
        print(f"\n🔍 Available server:")
        print(f"   • {server_config['name']}: {server_config['description']}")
        
        print("\n💡 Example usage with MCP Inspector:")
        print("   mcp inspect knowledge-graph-unified")
        
        print("\n🛠️ Available features:")
        print("   • Repository Management: create_repository, delete_repository")
        print("   • Entity CRUD: create_entity, update_entity, delete_entity")
        print("   • Relationships: create_relation")
        print("   • Search & Analysis: get_bulk_code, get_probable_code")
        print("   • Analytics: complexity-analysis, hotspots, dependencies")
        print("   • Code Quality: language-analysis, code-quality-metrics")
        print("   • Graph Operations: subgraph, tree, neighbors")
        print("   • Audit & History: audit logs and entity tracking")
        
        return True
    else:
        print(f"❌ Failed to install {server_config['name']}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi", 
        "networkx",
        "asyncio"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are satisfied")
    return True

def main():
    """Main installation function"""
    print("🚀 Unified Knowledge Graph MCP Installation")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before proceeding.")
        sys.exit(1)
    
    # Install unified MCP server
    if install_mcp_server():
        print("\n🎉 Unified MCP server has been installed successfully!")
        print("\n📚 All functionality is now available in a single server:")
        print("   - Repository analysis and management")
        print("   - Advanced code analytics and metrics")
        print("   - Search, context, and audit capabilities")
        print("   - Graph operations and relationship tracking")
        
        print("\n🚀 Ready to use!")
        print("Run the server directly with: python server.py")
        print("Or use MCP tools to interact with it programmatically")
        
        sys.exit(0)
    else:
        print("\n❌ Installation failed. Please check the errors above.")
        print("\n💡 Alternative: You can run the server directly with:")
        print("   python server.py")
        sys.exit(1)

if __name__ == "__main__":
    main() 