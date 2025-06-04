#!/usr/bin/env python3
"""
Comprehensive API Tests with Real Repository
===========================================

בדיקות מקיפות של כל ה-API endpoints שלנו עם repository אמיתי של AgentOps-AI.
מבדיק את כל הפונקציונליות מקצה לקצה.
"""

import pytest
import asyncio
import httpx
import json
from typing import Dict, List, Any

# API Base URL
BASE_URL = "http://localhost:8000"
REPO_URL = "https://github.com/AgentOps-AI/agentops"
REPO_ID = "agentops-test"


class TestAPIComprehensive:
    """בדיקות מקיפות של כל ה-API endpoints."""
    
    @pytest.fixture(scope="session")
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
    @pytest.fixture(scope="session") 
    async def client(self):
        """HTTP client for API testing."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            yield client
    
    @pytest.fixture(scope="session")
    async def analyzed_repo(self, client):
        """Analyze AgentOps repository once for all tests."""
        print(f"\n🔍 Analyzing repository: {REPO_URL}")
        
        response = await client.post(
            f"{BASE_URL}/repo",
            json={
                "repo_url": REPO_URL,
                "repo_id": REPO_ID
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        print(f"✅ Repository analyzed: {result}")
        return result
    
    async def test_01_health_check(self, client):
        """בדיקת בסיס - האם השרת עובד."""
        response = await client.get(f"{BASE_URL}/mcp")
        assert response.status_code == 200
        print("✅ Server is running")
    
    async def test_02_analyze_repository(self, analyzed_repo):
        """בדיקת ניתוח repository."""
        result = analyzed_repo
        
        # Verify basic structure
        assert "nodes_created" in result
        assert "edges_created" in result
        assert "files_analyzed" in result
        
        # Should have analyzed some files
        assert result["files_analyzed"] > 0
        assert result["nodes_created"] > 0
        
        print(f"✅ Repository analysis: {result['files_analyzed']} files, {result['nodes_created']} nodes, {result['edges_created']} edges")
    
    async def test_03_get_entities(self, client, analyzed_repo):
        """בדיקת קבלת entities."""
        response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entities")
        assert response.status_code == 200
        
        entities = response.json()
        assert isinstance(entities, list)
        assert len(entities) > 0
        
        # Check entity structure
        entity = entities[0]
        required_fields = ["uuid", "name", "type", "file_path"]
        for field in required_fields:
            assert field in entity
        
        print(f"✅ Found {len(entities)} entities")
        return entities
    
    async def test_04_get_statistics(self, client, analyzed_repo):
        """בדיקת סטטיסטיקות."""
        response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/statistics")
        assert response.status_code == 200
        
        stats = response.json()
        required_stats = ["total_entities", "total_relationships", "entity_types", "file_count"]
        for stat in required_stats:
            assert stat in stats
        
        assert stats["total_entities"] > 0
        assert stats["file_count"] > 0
        
        print(f"✅ Statistics: {stats['total_entities']} entities, {stats['total_relationships']} relationships")
        return stats
    
    async def test_05_get_entities_index(self, client, analyzed_repo):
        """בדיקת אינדקס entities."""
        response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entities-index")
        assert response.status_code == 200
        
        index = response.json()
        assert isinstance(index, list)
        
        if index:  # If we have entities
            entry = index[0]
            assert "name" in entry
            assert "type" in entry
            assert "file_path" in entry
        
        print(f"✅ Entities index: {len(index)} entries")
    
    async def test_06_filter_entities_by_type(self, client, analyzed_repo):
        """בדיקת סינון entities לפי סוג."""
        # Get functions only
        response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entities-index?type=function")
        assert response.status_code == 200
        
        functions = response.json()
        if functions:
            for func in functions:
                assert func["type"] == "function"
        
        print(f"✅ Found {len(functions)} functions")
    
    async def test_07_get_specific_entity(self, client, analyzed_repo):
        """בדיקת קבלת entity ספציפי."""
        # First get entities list
        entities_response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entities")
        entities = entities_response.json()
        
        if entities:
            entity_uuid = entities[0]["uuid"]
            
            response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entity/{entity_uuid}")
            assert response.status_code == 200
            
            entity = response.json()
            assert entity["uuid"] == entity_uuid
            
            print(f"✅ Retrieved entity: {entity['name']} ({entity['type']})")
    
    async def test_08_get_entity_neighbors(self, client, analyzed_repo):
        """בדיקת קבלת שכנים של entity."""
        # Get entities and find one with potential neighbors
        entities_response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entities")
        entities = entities_response.json()
        
        for entity in entities[:3]:  # Check first few entities
            entity_uuid = entity["uuid"]
            
            response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entity/{entity_uuid}/neighbors")
            assert response.status_code == 200
            
            neighbors = response.json()
            assert isinstance(neighbors, list)
            
            if neighbors:
                print(f"✅ Entity {entity['name']} has {len(neighbors)} neighbors")
                break
        else:
            print("✅ No neighbors found (expected for simple analysis)")
    
    async def test_09_get_entity_context(self, client, analyzed_repo):
        """בדיקת קבלת הקשר של entity."""
        entities_response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entities")
        entities = entities_response.json()
        
        if entities:
            entity_uuid = entities[0]["uuid"]
            
            response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entity/{entity_uuid}/context")
            assert response.status_code == 200
            
            context = response.json()
            assert "entity" in context
            assert "code_snippet" in context
            
            print(f"✅ Retrieved context for entity: {context['entity']['name']}")
    
    async def test_10_search_entities(self, client, analyzed_repo):
        """בדיקת חיפוש entities."""
        # Search by file
        response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/search?file_path=__init__.py")
        assert response.status_code == 200
        
        results = response.json()
        assert isinstance(results, list)
        
        print(f"✅ Search by file: {len(results)} results")
        
        # Search by type
        response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/search?type=function")
        assert response.status_code == 200
        
        results = response.json()
        if results:
            for result in results:
                assert result["type"] == "function"
        
        print(f"✅ Search by type: {len(results)} results")
    
    async def test_11_get_hotspots(self, client, analyzed_repo):
        """בדיקת קבלת hotspots."""
        response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/hotspots")
        assert response.status_code == 200
        
        hotspots = response.json()
        assert isinstance(hotspots, list)
        
        if hotspots:
            hotspot = hotspots[0]
            assert "entity" in hotspot
            assert "score" in hotspot
            assert "reasons" in hotspot
        
        print(f"✅ Found {len(hotspots)} hotspots")
    
    async def test_12_file_analysis(self, client, analyzed_repo):
        """בדיקת ניתוח קבצים ספציפיים."""
        # Get statistics to see what files were analyzed
        stats_response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/statistics")
        stats = stats_response.json()
        
        # Get entities to see file paths
        entities_response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entities")
        entities = entities_response.json()
        
        file_paths = set()
        for entity in entities:
            if entity.get("file_path"):
                file_paths.add(entity["file_path"])
        
        print(f"✅ Analyzed files in {len(file_paths)} different files")
        
        # Show some example files
        for i, file_path in enumerate(list(file_paths)[:5]):
            print(f"   📄 {file_path}")
    
    async def test_13_entity_types_analysis(self, client, analyzed_repo):
        """בדיקת סוגי entities שנמצאו."""
        entities_response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entities")
        entities = entities_response.json()
        
        entity_types = {}
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        print(f"✅ Entity types found:")
        for entity_type, count in entity_types.items():
            print(f"   {entity_type}: {count}")
        
        # Should have at least functions and classes in a Python project
        assert len(entity_types) > 0
    
    async def test_14_error_handling(self, client):
        """בדיקת טיפול בשגיאות."""
        # Test non-existent repo
        response = await client.get(f"{BASE_URL}/repo/non-existent/entities")
        assert response.status_code == 404
        
        # Test non-existent entity
        response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entity/non-existent-uuid")
        assert response.status_code == 404
        
        print("✅ Error handling works correctly")
    
    async def test_15_performance_check(self, client, analyzed_repo):
        """בדיקת ביצועים בסיסית."""
        import time
        
        # Measure response time for entities endpoint
        start_time = time.time()
        response = await client.get(f"{BASE_URL}/repo/{REPO_ID}/entities")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Should respond within reasonable time (5 seconds)
        assert response_time < 5.0
        
        print(f"✅ Entities endpoint responded in {response_time:.2f} seconds")


async def main():
    """Run all tests manually (for debugging)."""
    print("🚀 Starting comprehensive API tests...")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Test basic connectivity
        try:
            response = await client.get(f"{BASE_URL}/mcp")
            print(f"✅ Server is running: {response.status_code}")
        except Exception as e:
            print(f"❌ Server not running: {e}")
            return
        
        # Analyze repository
        print(f"\n🔍 Analyzing repository: {REPO_URL}")
        try:
            response = await client.post(
                f"{BASE_URL}/repo",
                json={
                    "repo_url": REPO_URL,
                    "repo_id": REPO_ID
                }
            )
            result = response.json()
            print(f"✅ Analysis complete: {result}")
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return
        
        # Test basic endpoints
        endpoints = [
            f"/repo/{REPO_ID}/entities",
            f"/repo/{REPO_ID}/statistics", 
            f"/repo/{REPO_ID}/entities-index",
            f"/repo/{REPO_ID}/hotspots"
        ]
        
        for endpoint in endpoints:
            try:
                response = await client.get(f"{BASE_URL}{endpoint}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {endpoint}: {len(data) if isinstance(data, list) else 'OK'}")
                else:
                    print(f"❌ {endpoint}: {response.status_code}")
            except Exception as e:
                print(f"❌ {endpoint}: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 