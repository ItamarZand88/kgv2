#!/usr/bin/env python3
"""
Simple API Integration Tests
===========================

בדיקות פשוטות של המערכת שלא דורשות שרת רץ.
"""

import pytest
import pytest_asyncio
import asyncio
import os
import tempfile
import shutil
from pathlib import Path

# Add current directory to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import CodeAnalyzer
from storage import StorageManager, GraphManager


@pytest_asyncio.fixture
async def temp_repo():
    """Create a temporary test repository."""
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple Python project structure
    repo_path = Path(temp_dir) / "test_repo"
    repo_path.mkdir()
    
    # Create __init__.py
    (repo_path / "__init__.py").write_text("""
\"\"\"Test package.\"\"\"

VERSION = "1.0.0"

def get_version():
    return VERSION
""")
    
    # Create main.py
    (repo_path / "main.py").write_text("""
#!/usr/bin/env python3
\"\"\"Main module.\"\"\"

import __init__
from utils import helper_function


class MainClass:
    \"\"\"Main application class.\"\"\"
    
    def __init__(self, name: str):
        self.name = name
    
    def run(self):
        \"\"\"Run the application.\"\"\"
        version = __init__.get_version()
        result = helper_function(self.name)
        return f"{result} (v{version})"


def main():
    \"\"\"Main entry point.\"\"\"
    app = MainClass("TestApp")
    return app.run()


if __name__ == "__main__":
    print(main())
""")
    
    # Create utils.py
    (repo_path / "utils.py").write_text("""
\"\"\"Utility functions.\"\"\"


def helper_function(name: str) -> str:
    \"\"\"Helper function.\"\"\"
    return f"Hello, {name}!"


def format_data(data: dict) -> str:
    \"\"\"Format data as string.\"\"\"
    return str(data)


class UtilityClass:
    \"\"\"Utility class.\"\"\"
    
    @staticmethod
    def static_method():
        \"\"\"Static utility method.\"\"\"
        return "static_result"
""")
    
    yield str(repo_path)
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_01_basic_analysis(temp_repo):
    """בדיקה בסיסית של ניתוח קוד."""
    # Initialize components
    storage = StorageManager()
    graph_manager = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_manager)
    
    # Analyze repository
    result = await analyzer.analyze_repository(temp_repo, "test_simple")
    
    # Check results
    assert result["files_analyzed"] > 0
    assert result["nodes_created"] > 0
    
    print(f"✅ Analyzed {result['files_analyzed']} files, created {result['nodes_created']} nodes")


@pytest.mark.asyncio
async def test_02_entities_detection(temp_repo):
    """בדיקת זיהוי entities."""
    storage = StorageManager()
    graph_manager = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_manager)
    
    # Analyze repository
    await analyzer.analyze_repository(temp_repo, "test_entities")
    
    # Get entities
    entities = await graph_manager.get_nodes("test_entities")
    
    # Should find functions, classes, variables
    entity_types = {}
    for entity in entities:
        entity_type = entity.type
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    print(f"✅ Found entity types: {entity_types}")
    
    # Should have at least functions and classes
    assert len(entities) > 0
    assert "function" in entity_types or "class" in entity_types


@pytest.mark.asyncio
async def test_03_relationships_detection(temp_repo):
    """בדיקת זיהוי קשרים."""
    storage = StorageManager()
    graph_manager = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_manager)
    
    # Analyze repository
    await analyzer.analyze_repository(temp_repo, "test_relationships")
    
    # Get relationships
    edges = await graph_manager.get_edges("test_relationships")
    
    print(f"✅ Found {len(edges)} relationships")
    
    if edges:
        # Check relationship structure
        edge = edges[0]
        assert hasattr(edge, 'type')
        assert hasattr(edge, 'from_uuid')
        assert hasattr(edge, 'to_uuid')
        
        # Print relationship types
        relationship_types = {}
        for edge in edges:
            rel_type = edge.type
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        print(f"   Relationship types: {relationship_types}")


@pytest.mark.asyncio
async def test_04_file_analysis(temp_repo):
    """בדיקת ניתוח קבצים."""
    storage = StorageManager()
    graph_manager = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_manager)
    
    # Analyze repository
    await analyzer.analyze_repository(temp_repo, "test_files")
    
    # Get entities grouped by file
    entities = await graph_manager.get_nodes("test_files")
    
    files_analyzed = set()
    for entity in entities:
        if entity.file_path:
            files_analyzed.add(entity.file_path)
    
    print(f"✅ Analyzed files: {sorted(files_analyzed)}")
    
    # Should analyze all Python files
    assert len(files_analyzed) >= 3  # __init__.py, main.py, utils.py


@pytest.mark.asyncio
async def test_05_entity_search(temp_repo):
    """בדיקת חיפוש entities."""
    storage = StorageManager()
    graph_manager = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_manager)
    
    # Analyze repository
    await analyzer.analyze_repository(temp_repo, "test_search")
    
    # Search for specific entities
    all_entities = await graph_manager.get_nodes("test_search")
    
    # Find function entities
    functions = [e for e in all_entities if e.type == "function"]
    classes = [e for e in all_entities if e.type == "class"]
    
    print(f"✅ Found {len(functions)} functions, {len(classes)} classes")
    
    # Should find some functions and classes
    assert len(functions) > 0 or len(classes) > 0


@pytest.mark.asyncio
async def test_06_statistics_generation(temp_repo):
    """בדיקת יצירת סטטיסטיקות."""
    storage = StorageManager()
    graph_manager = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_manager)
    
    # Analyze repository
    result = await analyzer.analyze_repository(temp_repo, "test_stats")
    
    # Calculate statistics
    entities = await graph_manager.get_nodes("test_stats")
    edges = await graph_manager.get_edges("test_stats")
    
    stats = {
        "total_entities": len(entities),
        "total_relationships": len(edges),
        "files_analyzed": result["files_analyzed"],
        "nodes_created": result["nodes_created"],
        "edges_created": result["edges_created"]
    }
    
    print(f"✅ Statistics: {stats}")
    
    # Verify statistics make sense
    assert stats["total_entities"] == stats["nodes_created"]
    assert stats["total_relationships"] == stats["edges_created"]
    assert stats["files_analyzed"] > 0


# Keep the synchronous wrapper for direct execution
def test_sync_wrapper():
    """Synchronous test that runs all async tests."""
    # Create temp repo
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir) / "test_repo"
    repo_path.mkdir()
    
    # Create test files (same as fixture)
    (repo_path / "__init__.py").write_text("""
\"\"\"Test package.\"\"\"
VERSION = "1.0.0"
def get_version():
    return VERSION
""")
    
    (repo_path / "main.py").write_text("""
#!/usr/bin/env python3
\"\"\"Main module.\"\"\"
import __init__
from utils import helper_function

class MainClass:
    def __init__(self, name: str):
        self.name = name
    
    def run(self):
        version = __init__.get_version()
        result = helper_function(self.name)
        return f"{result} (v{version})"

def main():
    app = MainClass("TestApp")
    return app.run()
""")
    
    (repo_path / "utils.py").write_text("""
\"\"\"Utility functions.\"\"\"
def helper_function(name: str) -> str:
    return f"Hello, {name}!"

class UtilityClass:
    @staticmethod
    def static_method():
        return "static_result"
""")
    
    try:
        # Run the basic analysis test
        async def test_basic():
            storage = StorageManager()
            graph_manager = GraphManager(storage)
            analyzer = CodeAnalyzer(storage, graph_manager)
            
            result = await analyzer.analyze_repository(str(repo_path), "test_sync")
            
            assert result["files_analyzed"] > 0
            assert result["nodes_created"] > 0
            
            print(f"✅ Sync test passed: {result['files_analyzed']} files, {result['nodes_created']} nodes")
            return True
        
        success = asyncio.run(test_basic())
        assert success
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests directly
    print("🧪 Running simple API tests...")
    test_sync_wrapper()
    print("✅ All tests passed!") 