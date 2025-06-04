import os
import logging
import asyncio
import uuid
import shutil
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path as PathLib
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.responses import JSONResponse
import networkx as nx

# Add FastAPI-MCP import
from fastapi_mcp import FastApiMCP

from models import (
    GraphNode, GraphEdge, AuditEvent, EntityContext,
    CreateRepoRequest, CreateRepoResponse, CreateNodeRequest, UpdateNodeRequest,
    CreateEdgeRequest, UpdateEdgeRequest, BulkCodeRequest, ProbableCodeRequest,
    SearchParams, EntityIndexEntry
)
from storage import StorageManager, GraphManager
from core import CodeAnalyzer
from search import SearchEngine
from context import ContextManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Graph API",
    description="A modular API for parsing Python repositories into a rich knowledge graph with advanced analytics capabilities",
    version="1.0.0"
)

# Initialize core components
storage = StorageManager()
graph_manager = GraphManager(storage)
analyzer = CodeAnalyzer(storage, graph_manager)
search_engine = SearchEngine(storage, graph_manager)
context_manager = ContextManager(storage, graph_manager)

# Initialize FastAPI-MCP
mcp = FastApiMCP(
    app,
    name="Knowledge Graph MCP Server",
    description="Advanced Knowledge Graph API for code analysis and insights via MCP",
    # Include only the most useful endpoints for MCP clients
    include_operations=[
        # Repository management
        "create_repository",
        "list_repositories", 
        "delete_repository",
        
        # Entity search and retrieval
        "get_entities_index",
        "search_entities",
        "get_entity",
        "get_entity_context",
        
        # Analytics endpoints
        "get_repository_statistics",
        "analyze_repository_complexity", 
        "find_code_hotspots",
        "analyze_dependencies",
        "analyze_repository_languages",
        "analyze_code_quality",
        "analyze_impact",
        
        # Neighborhood exploration
        "get_entity_neighbors",
        "get_entity_subgraph",
        
        # Code retrieval
        "get_bulk_code",
        "get_probable_code"
    ],
    # Provide detailed response schemas for better tool understanding
    describe_all_responses=True,
    describe_full_response_schema=True
)

# Mount the MCP server
mcp.mount()

# Repository Lifecycle Endpoints

@app.post("/repo", response_model=CreateRepoResponse, operation_id="create_repository")
async def create_repository(request: CreateRepoRequest):
    """Create or clone a repository and analyze it"""
    try:
        # Generate repo_id if not provided
        repo_id = request.repo_id or str(uuid.uuid4())
        
        # Check if repo already exists
        if await storage.repo_exists(repo_id):
            raise HTTPException(status_code=400, detail=f"Repository {repo_id} already exists")
        
        # Handle local path vs git URL
        if request.repo_path.startswith(('http://', 'https://', 'git@')):
            # Clone repository
            temp_dir = f"temp_{repo_id}"
            if not await clone_repository(request.repo_path, temp_dir):
                raise HTTPException(status_code=400, detail="Failed to clone repository")
            repo_path = temp_dir
        else:
            # Use local path
            if not os.path.exists(request.repo_path):
                raise HTTPException(status_code=400, detail="Repository path does not exist")
            repo_path = request.repo_path
        
        # Analyze repository
        analysis_result = await analyzer.analyze_repository(repo_path, repo_id)
        
        # Clean up temp directory if created
        if request.repo_path.startswith(('http://', 'https://', 'git@')):
            shutil.rmtree(repo_path, ignore_errors=True)
        
        # Log repository creation
        await storage.append_audit_event(repo_id, AuditEvent(
            user="system",
            action="create_repo",
            entity_type="repo",
            entity_uuid=repo_id,
            details={
                "repo_path": request.repo_path,
                "analysis_result": analysis_result
            }
        ))
        
        return CreateRepoResponse(
            repo_id=repo_id,
            message=f"Repository analyzed successfully. Created {analysis_result['nodes_created']} nodes and {analysis_result['edges_created']} edges."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/repo/{repo_id}", operation_id="delete_repository")
async def delete_repository(repo_id: str = Path(..., description="Repository ID")):
    """Delete all data for a repository"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        success = await storage.delete_repo(repo_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete repository")
        
        return {"message": f"Repository {repo_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repos", operation_id="list_repositories")
async def list_repositories():
    """List all available repositories"""
    try:
        repos = await storage.list_repos()
        return {"repositories": repos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Entity Index and Search Endpoints

@app.get("/repo/{repo_id}/entities-index", operation_id="get_entities_index")
async def get_entities_index(
    repo_id: str = Path(..., description="Repository ID"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    name: Optional[str] = Query(None, description="Filter by name"),
    type: Optional[str] = Query(None, description="Filter by type"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags")
):
    """Get flat index of all entities with filtering and pagination"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        params = SearchParams(
            name=name,
            type=type,
            file_path=file_path,
            tags=tags or [],
            limit=limit,
            offset=offset
        )
        
        entities = await search_engine.search_entities(repo_id, params)
        return {"entities": entities, "count": len(entities)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repo/{repo_id}/search", operation_id="search_entities")
async def search_entities(
    repo_id: str = Path(..., description="Repository ID"),
    name: Optional[str] = Query(None, description="Search by name (fuzzy)"),
    type: Optional[str] = Query(None, description="Filter by type"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Advanced search with fuzzy matching and filtering"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        params = SearchParams(
            name=name,
            type=type,
            file_path=file_path,
            tags=tags or [],
            limit=limit,
            offset=offset
        )
        
        results = await search_engine.search_entities(repo_id, params)
        return {"results": results, "count": len(results)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Node (Entity) CRUD Endpoints

@app.get("/repo/{repo_id}/entities", operation_id="list_entities")
async def list_entities(
    repo_id: str = Path(..., description="Repository ID"),
    type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all entities with optional filtering"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        params = SearchParams(type=type, limit=limit, offset=offset)
        entities = await search_engine.search_entities(repo_id, params)
        return {"entities": entities}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repo/{repo_id}/entity/{entity_uuid}", operation_id="get_entity")
async def get_entity(
    repo_id: str = Path(..., description="Repository ID"),
    entity_uuid: str = Path(..., description="Entity UUID")
):
    """Get entity details by UUID"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        entity = await graph_manager.get_node(repo_id, entity_uuid)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        return entity
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repo/{repo_id}/entity", operation_id="create_entity")
async def create_entity(
    request: CreateNodeRequest,
    repo_id: str = Path(..., description="Repository ID")
):
    """Create a new entity manually"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Create new node
        node = GraphNode(
            repo_id=repo_id,
            file_path=request.file_path,
            type=request.type,
            name=request.name,
            start_line=request.start_line,
            end_line=request.end_line,
            docstring=request.docstring,
            tags=request.tags,
            created_by=request.created_by,
            manual=True
        )
        
        # Add to graph
        await graph_manager.add_node(repo_id, node)
        
        # Update index
        index = await storage.load_index(repo_id)
        index_entry = EntityIndexEntry(
            uuid=node.uuid,
            name=node.name,
            type=node.type,
            file_path=node.file_path,
            start_line=node.start_line,
            tags=node.tags,
            short_doc=node.docstring.split('\n')[0][:100] if node.docstring else None
        )
        index.append(index_entry)
        await storage.save_index(repo_id, index)
        
        # Log creation
        await storage.append_audit_event(repo_id, AuditEvent(
            user=request.created_by,
            action="create_entity",
            entity_type="node",
            entity_uuid=node.uuid,
            details={"name": node.name, "type": node.type, "manual": True}
        ))
        
        return node
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/repo/{repo_id}/entity/{entity_uuid}", operation_id="update_entity")
async def update_entity(
    request: UpdateNodeRequest,
    repo_id: str = Path(..., description="Repository ID"),
    entity_uuid: str = Path(..., description="Entity UUID")
):
    """Update entity fields"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Get existing entity
        existing_node = await graph_manager.get_node(repo_id, entity_uuid)
        if not existing_node:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        # Update fields
        update_data = request.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(existing_node, field, value)
        
        # Add to history
        existing_node.history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "update",
            "changes": update_data
        })
        
        # Update in graph
        await graph_manager.update_node(repo_id, existing_node)
        
        # Update index
        index = await storage.load_index(repo_id)
        for i, entry in enumerate(index):
            if entry.uuid == entity_uuid:
                # Update index entry
                index[i] = EntityIndexEntry(
                    uuid=existing_node.uuid,
                    name=existing_node.name,
                    type=existing_node.type,
                    file_path=existing_node.file_path,
                    start_line=existing_node.start_line,
                    tags=existing_node.tags,
                    short_doc=existing_node.docstring.split('\n')[0][:100] if existing_node.docstring else None
                )
                break
        await storage.save_index(repo_id, index)
        
        # Log update
        await storage.append_audit_event(repo_id, AuditEvent(
            user="user",
            action="update_entity",
            entity_type="node",
            entity_uuid=entity_uuid,
            details=update_data
        ))
        
        return existing_node
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/repo/{repo_id}/entity/{entity_uuid}", operation_id="delete_entity")
async def delete_entity(
    repo_id: str = Path(..., description="Repository ID"),
    entity_uuid: str = Path(..., description="Entity UUID")
):
    """Delete an entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Check if entity exists
        entity = await graph_manager.get_node(repo_id, entity_uuid)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        # Remove from graph
        success = await graph_manager.remove_node(repo_id, entity_uuid)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete entity")
        
        # Update index
        index = await storage.load_index(repo_id)
        index = [entry for entry in index if entry.uuid != entity_uuid]
        await storage.save_index(repo_id, index)
        
        # Log deletion
        await storage.append_audit_event(repo_id, AuditEvent(
            user="user",
            action="delete_entity",
            entity_type="node",
            entity_uuid=entity_uuid,
            details={"name": entity.name, "type": entity.type}
        ))
        
        return {"message": "Entity deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Edge (Relationship) CRUD Endpoints

@app.get("/repo/{repo_id}/relations", operation_id="list_relations")
async def list_relations(
    repo_id: str = Path(..., description="Repository ID"),
    type: Optional[str] = Query(None, description="Filter by relation type"),
    from_uuid: Optional[str] = Query(None, description="Filter by source entity"),
    to_uuid: Optional[str] = Query(None, description="Filter by target entity"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all relationships with filtering"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        graph = await graph_manager.get_graph(repo_id)
        edges = []
        
        for from_id, to_id, edge_data in graph.edges(data=True):
            edge = GraphEdge(**edge_data)
            
            # Apply filters
            if type and edge.type != type:
                continue
            if from_uuid and edge.from_uuid != from_uuid:
                continue
            if to_uuid and edge.to_uuid != to_uuid:
                continue
            
            edges.append(edge)
        
        # Apply pagination
        start = offset
        end = start + limit
        return {"relations": edges[start:end], "total": len(edges)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repo/{repo_id}/relation", operation_id="create_relation")
async def create_relation(
    request: CreateEdgeRequest,
    repo_id: str = Path(..., description="Repository ID")
):
    """Create a new relationship manually"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Verify both entities exist
        from_node = await graph_manager.get_node(repo_id, request.from_uuid)
        to_node = await graph_manager.get_node(repo_id, request.to_uuid)
        
        if not from_node:
            raise HTTPException(status_code=400, detail="Source entity not found")
        if not to_node:
            raise HTTPException(status_code=400, detail="Target entity not found")
        
        # Create edge
        edge = GraphEdge(
            from_uuid=request.from_uuid,
            to_uuid=request.to_uuid,
            type=request.type,
            created_by=request.created_by,
            manual=True
        )
        
        # Add to graph
        await graph_manager.add_edge(repo_id, edge)
        
        # Log creation
        await storage.append_audit_event(repo_id, AuditEvent(
            user=request.created_by,
            action="create_relation",
            entity_type="edge",
            entity_uuid=edge.uuid,
            details={"type": edge.type, "from_uuid": edge.from_uuid, "to_uuid": edge.to_uuid}
        ))
        
        return edge
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Context and Code Retrieval Endpoints

@app.get("/repo/{repo_id}/entity/{entity_uuid}/context", operation_id="get_entity_context")
async def get_entity_context(
    repo_id: str = Path(..., description="Repository ID"),
    entity_uuid: str = Path(..., description="Entity UUID")
):
    """Get full context for an entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        context = await context_manager.get_entity_context(repo_id, entity_uuid)
        if not context:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        return context
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repo/{repo_id}/bulk-code", operation_id="get_bulk_code")
async def get_bulk_code(
    request: BulkCodeRequest,
    repo_id: str = Path(..., description="Repository ID")
):
    """Get code and metadata for multiple entities"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        contexts = await context_manager.get_bulk_context(repo_id, request.uuids)
        return {"contexts": contexts}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repo/{repo_id}/probable-code", operation_id="get_probable_code")
async def get_probable_code(
    request: ProbableCodeRequest,
    repo_id: str = Path(..., description="Repository ID")
):
    """Get entities by probable names"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        entities = await search_engine.find_by_probable_names(repo_id, request.probable_names)
        return {"entities": entities}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Neighbor and Subgraph Endpoints

@app.get("/repo/{repo_id}/entity/{entity_uuid}/neighbors", operation_id="get_entity_neighbors")
async def get_entity_neighbors(
    repo_id: str = Path(..., description="Repository ID"),
    entity_uuid: str = Path(..., description="Entity UUID"),
    type: Optional[str] = Query(None, description="Filter by relation type"),
    direction: str = Query("both", pattern="^(in|out|both)$"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get neighbor entities"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        neighbor_uuids = await graph_manager.get_neighbors(repo_id, entity_uuid, direction)
        neighbors = []
        
        for uuid in neighbor_uuids[offset:offset + limit]:
            neighbor = await graph_manager.get_node(repo_id, uuid)
            if neighbor:
                neighbors.append(neighbor)
        
        return {"neighbors": neighbors, "total": len(neighbor_uuids)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repo/{repo_id}/entity/{entity_uuid}/subgraph", operation_id="get_entity_subgraph")
async def get_entity_subgraph(
    repo_id: str = Path(..., description="Repository ID"),
    entity_uuid: str = Path(..., description="Entity UUID"),
    depth: int = Query(2, ge=1, le=5)
):
    """Get subgraph around an entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        subgraph = await graph_manager.get_subgraph(repo_id, entity_uuid, depth)
        
        # Convert to serializable format
        nodes = []
        edges = []
        
        for node_id, node_data in subgraph.nodes(data=True):
            nodes.append(GraphNode(**node_data))
        
        for from_id, to_id, edge_data in subgraph.edges(data=True):
            edges.append(GraphEdge(**edge_data))
        
        return {"nodes": nodes, "edges": edges}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repo/{repo_id}/entity/{entity_uuid}/tree", operation_id="get_entity_tree")
async def get_entity_tree(
    repo_id: str = Path(..., description="Repository ID"),
    entity_uuid: str = Path(..., description="Entity UUID"),
    relation_type: str = Query("calls", description="Type of relation to traverse"),
    direction: str = Query("out", pattern="^(in|out)$"),
    max_depth: int = Query(3, ge=1, le=5)
):
    """Get call or inheritance tree for an entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        tree = await context_manager.get_call_tree(repo_id, entity_uuid, direction, max_depth)
        return {"tree": tree}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Audit Log Endpoints

@app.get("/repo/{repo_id}/audit", operation_id="get_audit_log")
async def get_audit_log(
    repo_id: str = Path(..., description="Repository ID"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get audit log for repository"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        events = await storage.load_audit_events(repo_id, limit, offset)
        return {"events": events}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/repo/{repo_id}/entity/{entity_uuid}/audit", operation_id="get_entity_audit")
async def get_entity_audit(
    repo_id: str = Path(..., description="Repository ID"),
    entity_uuid: str = Path(..., description="Entity UUID")
):
    """Get audit events for a specific entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        all_events = await storage.load_audit_events(repo_id, limit=10000)
        entity_events = [event for event in all_events if event.entity_uuid == entity_uuid]
        
        return {"events": entity_events}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.get("/repo/{repo_id}/statistics", operation_id="get_repository_statistics")
async def get_repository_statistics(repo_id: str):
    """Get comprehensive repository statistics and metrics"""
    try:
        # Check if repository exists
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        graph = await graph_manager.get_graph(repo_id)
        
        # Basic graph metrics
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        
        # Node type distribution
        node_types = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Edge type distribution
        edge_types = {}
        for source, target, data in graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # File distribution
        file_stats = {}
        for node, data in graph.nodes(data=True):
            file_path = data.get('file_path', 'unknown')
            if file_path != 'unknown':
                ext = os.path.splitext(file_path)[1].lower()
                if ext:
                    file_stats[ext] = file_stats.get(ext, 0) + 1
        
        # Complexity metrics
        most_connected_nodes = sorted(
            [(node, graph.degree(node)) for node in graph.nodes()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate graph density
        density = nx.density(graph) if node_count > 0 else 0
        
        # Find strongly connected components
        if graph.is_directed():
            scc_count = nx.number_strongly_connected_components(graph)
        else:
            scc_count = nx.number_connected_components(graph)
        
        return {
            "basic_metrics": {
                "total_nodes": node_count,
                "total_edges": edge_count,
                "graph_density": density,
                "connected_components": scc_count
            },
            "node_distribution": node_types,
            "edge_distribution": edge_types,
            "file_distribution": file_stats,
            "complexity_metrics": {
                "most_connected_nodes": [
                    {"node": node, "connections": degree} 
                    for node, degree in most_connected_nodes
                ],
                "average_degree": sum(dict(graph.degree()).values()) / node_count if node_count > 0 else 0
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        logger.error(f"Error getting repository statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repo/{repo_id}/complexity-analysis", operation_id="analyze_repository_complexity")
async def analyze_repository_complexity(repo_id: str):
    """Analyze repository complexity and patterns"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        graph = await graph_manager.get_graph(repo_id)
        
        # Find cyclomatic complexity (proxy using in-degree)
        complexity_scores = {}
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'function':
                # Use in-degree as a proxy for complexity
                in_degree = graph.in_degree(node)
                out_degree = graph.out_degree(node)
                complexity_scores[node] = {
                    "node": node,
                    "name": data.get('name', ''),
                    "file_path": data.get('file_path', ''),
                    "complexity_score": in_degree + out_degree,
                    "incoming_calls": in_degree,
                    "outgoing_calls": out_degree
                }
        
        # Sort by complexity
        most_complex = sorted(
            complexity_scores.values(),
            key=lambda x: x["complexity_score"],
            reverse=True
        )[:20]
        
        # Find potential code smells
        code_smells = []
        
        # God classes (classes with too many methods)
        class_method_count = {}
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'function':
                class_name = data.get('class_name')
                if class_name:
                    class_method_count[class_name] = class_method_count.get(class_name, 0) + 1
        
        god_classes = [(cls, count) for cls, count in class_method_count.items() if count > 10]
        
        # Long parameter lists (functions with many dependencies)
        long_param_functions = [
            item for item in most_complex 
            if item["incoming_calls"] > 5
        ]
        
        return {
            "most_complex_functions": most_complex,
            "code_smells": {
                "god_classes": [{"class": cls, "method_count": count} for cls, count in god_classes],
                "high_dependency_functions": long_param_functions[:10]
            },
            "recommendations": {
                "refactor_candidates": most_complex[:5],
                "complexity_distribution": {
                    "low_complexity": len([x for x in complexity_scores.values() if x["complexity_score"] <= 2]),
                    "medium_complexity": len([x for x in complexity_scores.values() if 2 < x["complexity_score"] <= 5]),
                    "high_complexity": len([x for x in complexity_scores.values() if x["complexity_score"] > 5])
                }
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        logger.error(f"Error analyzing repository complexity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repo/{repo_id}/hotspots", operation_id="find_code_hotspots")
async def find_code_hotspots(repo_id: str, limit: int = 10):
    """Find code hotspots - most important/central code elements"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        graph = await graph_manager.get_graph(repo_id)
        
        # Calculate centrality measures
        try:
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            pagerank = nx.pagerank(graph)
        except:
            # Fallback for disconnected graphs
            betweenness = {node: 0 for node in graph.nodes()}
            closeness = {node: 0 for node in graph.nodes()}
            pagerank = {node: 1/len(graph.nodes()) for node in graph.nodes()}
        
        # Combine metrics to find hotspots
        hotspots = []
        for node in graph.nodes():
            data = graph.nodes[node]
            hotspot_score = (
                betweenness.get(node, 0) * 0.4 +
                closeness.get(node, 0) * 0.3 +
                pagerank.get(node, 0) * 0.3
            )
            
            hotspots.append({
                "node": node,
                "name": data.get('name', ''),
                "type": data.get('type', ''),
                "file_path": data.get('file_path', ''),
                "hotspot_score": hotspot_score,
                "betweenness_centrality": betweenness.get(node, 0),
                "closeness_centrality": closeness.get(node, 0),
                "pagerank_score": pagerank.get(node, 0),
                "connections": graph.degree(node)
            })
        
        # Sort by hotspot score
        hotspots.sort(key=lambda x: x["hotspot_score"], reverse=True)
        
        return {
            "hotspots": hotspots[:limit],
            "summary": {
                "total_analyzed": len(hotspots),
                "average_hotspot_score": sum(h["hotspot_score"] for h in hotspots) / len(hotspots) if hotspots else 0,
                "top_types": {}
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        logger.error(f"Error finding code hotspots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repo/{repo_id}/dependencies", operation_id="analyze_dependencies")
async def analyze_dependencies(repo_id: str, depth: int = 3):
    """Analyze dependency relationships and patterns"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        graph = await graph_manager.get_graph(repo_id)
        
        # Find import relationships
        import_edges = []
        for source, target, data in graph.edges(data=True):
            if data.get('type') == 'imports':
                import_edges.append((source, target, data))
        
        # Build dependency graph
        dep_graph = nx.DiGraph()
        for source, target, data in import_edges:
            dep_graph.add_edge(source, target, **data)
        
        # Find circular dependencies
        try:
            cycles = list(nx.simple_cycles(dep_graph))
            circular_deps = cycles[:10]  # Limit to first 10 cycles
        except:
            circular_deps = []
        
        # Most depended upon (highest in-degree)
        in_degrees = dict(dep_graph.in_degree())
        most_depended = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Most dependent (highest out-degree)
        out_degrees = dict(dep_graph.out_degree())
        most_dependent = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Architectural issues
        issues = []
        
        # Find nodes with high fan-out (potential god objects)
        for node, out_degree in out_degrees.items():
            if out_degree > 10:
                node_data = graph.nodes.get(node, {})
                issues.append({
                    "type": "high_fan_out",
                    "node": node,
                    "name": node_data.get('name', ''),
                    "file_path": node_data.get('file_path', ''),
                    "dependency_count": out_degree,
                    "severity": "high" if out_degree > 20 else "medium"
                })
        
        # Find nodes with high fan-in (potential bottlenecks)
        for node, in_degree in in_degrees.items():
            if in_degree > 15:
                node_data = graph.nodes.get(node, {})
                issues.append({
                    "type": "high_fan_in",
                    "node": node,
                    "name": node_data.get('name', ''),
                    "file_path": node_data.get('file_path', ''),
                    "dependent_count": in_degree,
                    "severity": "high" if in_degree > 25 else "medium"
                })
        
        return {
            "summary": {
                "total_nodes": dep_graph.number_of_nodes(),
                "total_import_relationships": dep_graph.number_of_edges(),
                "circular_dependencies_count": len(circular_deps),
                "max_dependency_depth": depth,
                "architectural_issues_count": len(issues)
            },
            "circular_dependencies": [
                {
                    "cycle": cycle,
                    "length": len(cycle),
                    "severity": "high" if len(cycle) <= 3 else "medium"
                }
                for cycle in circular_deps
            ],
            "most_depended_upon": [
                {
                    "node": node,
                    "name": graph.nodes.get(node, {}).get('name', ''),
                    "file_path": graph.nodes.get(node, {}).get('file_path', ''),
                    "dependent_count": count
                }
                for node, count in most_depended
            ],
            "most_dependent": [
                {
                    "node": node,
                    "name": graph.nodes.get(node, {}).get('name', ''),
                    "file_path": graph.nodes.get(node, {}).get('file_path', ''),
                    "dependency_count": count
                }
                for node, count in most_dependent
            ],
            "architectural_issues": issues,
            "recommendations": [
                "Consider breaking down high fan-out components",
                "Review circular dependencies for potential refactoring",
                "Monitor high fan-in components for performance bottlenecks"
            ]
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        logger.error(f"Error analyzing dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repo/{repo_id}/impact-analysis/{node_id}", operation_id="analyze_impact")
async def analyze_impact(repo_id: str, node_id: str):
    """Analyze the impact of changes to a specific node"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        graph = await graph_manager.get_graph(repo_id)
        
        if not graph.has_node(node_id):
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Find all nodes that would be affected by changes to this node
        affected_nodes = set()
        
        # Direct dependents (immediate impact)
        direct_dependents = set()
        for source in graph.predecessors(node_id):
            edge_data = graph.get_edge_data(source, node_id, {})
            if edge_data.get('type') in ['imports', 'calls', 'uses']:
                direct_dependents.add(source)
                affected_nodes.add(source)
        
        # Transitive dependents (ripple effect)
        transitive_dependents = set()
        for dependent in direct_dependents:
            for source in graph.predecessors(dependent):
                edge_data = graph.get_edge_data(source, dependent, {})
                if edge_data.get('type') in ['imports', 'calls', 'uses']:
                    transitive_dependents.add(source)
                    affected_nodes.add(source)
        
        # Analyze impact by file
        files_affected = {}
        for node in affected_nodes:
            node_data = graph.nodes.get(node, {})
            file_path = node_data.get('file_path', 'unknown')
            if file_path not in files_affected:
                files_affected[file_path] = []
            files_affected[file_path].append({
                "node": node,
                "name": node_data.get('name', ''),
                "type": node_data.get('type', '')
            })
        
        # Calculate impact score
        impact_score = len(direct_dependents) * 3 + len(transitive_dependents) * 1
        
        # Risk assessment
        risk_level = "low"
        if impact_score > 20:
            risk_level = "high"
        elif impact_score > 10:
            risk_level = "medium"
        
        node_data = graph.nodes.get(node_id, {})
        
        return {
            "target_node": {
                "node": node_id,
                "name": node_data.get('name', ''),
                "type": node_data.get('type', ''),
                "file_path": node_data.get('file_path', '')
            },
            "impact_summary": {
                "total_affected_nodes": len(affected_nodes),
                "direct_dependents": len(direct_dependents),
                "transitive_dependents": len(transitive_dependents),
                "files_affected": len(files_affected),
                "impact_score": impact_score,
                "risk_level": risk_level
            },
            "affected_files": dict(list(files_affected.items())[:20]),  # Limit output
            "direct_dependents": [
                {
                    "node": node,
                    **graph.nodes.get(node, {})
                } for node in list(direct_dependents)[:10]
            ],
            "recommendations": {
                "requires_careful_testing": risk_level in ["medium", "high"],
                "consider_gradual_rollout": risk_level == "high",
                "suggested_test_files": list(files_affected.keys())[:10]
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        logger.error(f"Error analyzing impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repo/{repo_id}/language-analysis", operation_id="analyze_repository_languages")
async def analyze_repository_languages(repo_id: str):
    """Analyze programming languages used in the repository"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        # Get graph data to analyze file extensions and types
        graph = await graph_manager.get_graph(repo_id)
        
        # Analyze languages based on file extensions from the graph
        file_extensions = {}
        file_paths = set()
        total_nodes = 0
        
        for node, data in graph.nodes(data=True):
            file_path = data.get('file_path', '')
            if file_path:
                file_paths.add(file_path)
                total_nodes += 1
                ext = os.path.splitext(file_path)[1].lower()
                if ext:
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
        
        # Map extensions to languages
        extension_to_language = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.json': 'json',
            '.xml': 'xml',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.md': 'markdown',
            '.sql': 'sql',
            '.sh': 'shell'
        }
        
        # Count languages based on nodes (not unique files)
        language_counts = {}
        
        for ext, count in file_extensions.items():
            language = extension_to_language.get(ext, 'other')
            language_counts[language] = language_counts.get(language, 0) + count
        
        # Calculate percentages based on total nodes
        language_percentages = {}
        for lang, count in language_counts.items():
            if total_nodes > 0:
                language_percentages[lang] = (count / total_nodes) * 100
        
        # Find primary language
        primary_language = max(language_counts.items(), key=lambda x: x[1])[0] if language_counts else "unknown"
        
        # Build response
        response = {
            "primary_language": primary_language,
            "language_distribution": language_percentages,
            "file_counts": language_counts,
            "total_files": len(file_paths),
            "total_nodes": total_nodes,
            "file_extensions": file_extensions,
            "language_confidence": 0.9,  # High confidence based on file extension analysis
            "supported_languages": list(language_counts.keys()),
            "analysis_summary": {
                "is_multilingual": len([l for l in language_percentages.values() if l > 10]) > 1,
                "dominant_language_percentage": max(language_percentages.values()) if language_percentages else 0,
                "language_diversity_score": len(language_percentages),
                "total_languages": len(language_counts),
                "main_languages": sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        }
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        logger.error(f"Error analyzing repository languages: {e}")
        logger.error(f"Full traceback: ", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Language analysis failed: {str(e)}")

@app.get("/repo/{repo_id}/code-quality-metrics", operation_id="analyze_code_quality")
async def analyze_code_quality(repo_id: str):
    """Analyze overall code quality metrics"""
    try:
        if not await storage.repo_exists(repo_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        
        graph = await graph_manager.get_graph(repo_id)
        
        # Calculate various quality metrics
        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()
        
        # Maintainability metrics
        function_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'function']
        class_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'class']
        
        # Complexity metrics
        avg_degree = sum(dict(graph.degree()).values()) / total_nodes if total_nodes > 0 else 0
        max_degree = max(dict(graph.degree()).values()) if total_nodes > 0 else 0
        
        # Calculate complexity distribution
        degrees = list(dict(graph.degree()).values())
        low_complexity = len([d for d in degrees if d <= 3])
        medium_complexity = len([d for d in degrees if 3 < d <= 7])
        high_complexity = len([d for d in degrees if d > 7])
        
        # Documentation metrics (proxy using comments/docstrings)
        documented_functions = 0
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'function':
                # Simple heuristic: assume documented if has description
                if data.get('description') or data.get('docstring'):
                    documented_functions += 1
        
        documentation_ratio = documented_functions / len(function_nodes) if function_nodes else 0
        
        # Structure metrics
        file_count = len(set(d.get('file_path', '') for n, d in graph.nodes(data=True) if d.get('file_path')))
        avg_functions_per_file = len(function_nodes) / file_count if file_count > 0 else 0
        
        # Calculate component scores (0-100)
        maintainability_score = min(100, max(0, 100 - (avg_degree * 10)))  # Lower degree = higher maintainability
        complexity_score = min(100, max(0, 100 - (high_complexity / total_nodes * 100) if total_nodes > 0 else 100))
        documentation_score = documentation_ratio * 100
        structure_score = min(100, max(0, 100 - abs(avg_functions_per_file - 10) * 5))  # Optimal ~10 functions per file
        
        # Overall score (weighted average)
        overall_score = (
            maintainability_score * 0.3 +
            complexity_score * 0.3 +
            documentation_score * 0.2 +
            structure_score * 0.2
        )
        
        # Determine quality grade
        if overall_score >= 90:
            quality_grade = "A"
        elif overall_score >= 80:
            quality_grade = "B"
        elif overall_score >= 70:
            quality_grade = "C"
        elif overall_score >= 60:
            quality_grade = "D"
        else:
            quality_grade = "F"
        
        # Generate recommendations
        recommendations = []
        if complexity_score < 70:
            recommendations.append("Consider refactoring high-complexity functions")
        if documentation_score < 50:
            recommendations.append("Improve code documentation and comments")
        if maintainability_score < 70:
            recommendations.append("Reduce coupling between components")
        if structure_score < 70:
            recommendations.append("Review file organization and function distribution")
        
        return {
            "overall_score": round(overall_score, 2),
            "quality_grade": quality_grade,
            "metrics": {
                "maintainability": round(maintainability_score, 2),
                "complexity": round(complexity_score, 2),
                "documentation": round(documentation_score, 2),
                "structure": round(structure_score, 2)
            },
            "component_scores": {
                "maintainability": {
                    "score": round(maintainability_score, 2),
                    "description": "Code maintainability and coupling",
                    "details": {
                        "average_degree": round(avg_degree, 2),
                        "max_degree": max_degree
                    }
                },
                "complexity": {
                    "score": round(complexity_score, 2),
                    "description": "Code complexity distribution",
                    "details": {
                        "low_complexity_nodes": low_complexity,
                        "medium_complexity_nodes": medium_complexity,
                        "high_complexity_nodes": high_complexity
                    }
                },
                "documentation": {
                    "score": round(documentation_score, 2),
                    "description": "Code documentation coverage",
                    "details": {
                        "documented_functions": documented_functions,
                        "total_functions": len(function_nodes),
                        "documentation_ratio": round(documentation_ratio, 3)
                    }
                },
                "structure": {
                    "score": round(structure_score, 2),
                    "description": "Code organization and structure",
                    "details": {
                        "total_files": file_count,
                        "total_functions": len(function_nodes),
                        "total_classes": len(class_nodes),
                        "avg_functions_per_file": round(avg_functions_per_file, 2)
                    }
                }
            },
            "recommendations": recommendations,
            "summary": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        logger.error(f"Error analyzing code quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def clone_repository(git_url: str, target_dir: str) -> bool:
    """
    Clone a git repository asynchronously.
    
    Args:
        git_url: Git repository URL
        target_dir: Target directory for cloning
        
    Returns:
        True if cloning succeeded, False otherwise
    """
    try:
        # Use subprocess to clone repository
        process = await asyncio.create_subprocess_exec(
            'git', 'clone', git_url, target_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Successfully cloned {git_url} to {target_dir}")
            return True
        else:
            logger.error(f"Failed to clone {git_url}: {stderr.decode()}")
            return False
            
    except Exception as e:
        logger.error(f"Error cloning repository {git_url}: {e}")
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 