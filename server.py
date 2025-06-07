import os
import logging
import asyncio
import uuid
import shutil
import sys
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path as PathLib

# Try importing the MCP SDK with error handling
try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:
    print(f"Error importing MCP SDK: {e}", file=sys.stderr)
    print("Please install the MCP SDK: pip install mcp", file=sys.stderr)
    exit(1)

# Try importing our core components with error handling
try:
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
except ImportError as e:
    print(f"Error importing project modules: {e}", file=sys.stderr)
    print("Please ensure all project dependencies are installed and modules are available", file=sys.stderr)
    exit(1)

# For NetworkX operations
try:
    import networkx as nx
except ImportError as e:
    print(f"Error importing NetworkX: {e}", file=sys.stderr)
    print("Please install NetworkX: pip install networkx", file=sys.stderr)
    exit(1)

# Configure logging to use stderr - use INFO level to see progress
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("KnowledgeGraphMCP")

# Initialize core components with error handling
try:
    storage = StorageManager()
    graph_manager = GraphManager(storage)
    analyzer = CodeAnalyzer(storage, graph_manager)
    search_engine = SearchEngine(storage, graph_manager)
    context_manager = ContextManager(storage, graph_manager)
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    print(f"Error initializing components: {e}", file=sys.stderr)
    exit(1)

# Repository Lifecycle Tools



@mcp.tool()
async def create_repository_and_build_knowledge_graph(repo_path: str, repo_id: str = "") -> dict:
    """Create or clone a repository and analyze it to build a knowledge graph"""
    try:
        # Generate repo_id if not provided - with better validation
        if not repo_id or repo_id.strip() == "":
            repo_id = str(uuid.uuid4())
            logger.info(f"Generated new repo_id: {repo_id}")
        else:
            logger.info(f"Using provided repo_id: {repo_id}")
        
        # Validate repo_id
        if not repo_id or not isinstance(repo_id, str) or len(repo_id.strip()) == 0:
            raise Exception("Invalid repo_id generated or provided")
        
        logger.info(f"Starting repository creation for {repo_path} with repo_id {repo_id}")
        
        # Check if repo already exists
        if await storage.repo_exists(repo_id):
            raise Exception(f"Repository {repo_id} already exists")
        
        # Handle local path vs git URL
        if repo_path.startswith(('http://', 'https://', 'git@')):
            logger.info(f"Cloning repository from {repo_path}")
            # Clone repository with timeout
            temp_dir = f"temp_{repo_id}"
            
            try:
                # Add timeout for cloning (5 minutes)
                clone_task = asyncio.create_task(clone_repository(repo_path, temp_dir))
                if not await asyncio.wait_for(clone_task, timeout=300):  # 5 minutes
                    raise Exception("Failed to clone repository")
                logger.info(f"Successfully cloned repository to {temp_dir}")
            except asyncio.TimeoutError:
                raise Exception("Repository cloning timed out after 5 minutes")
            
            actual_repo_path = temp_dir
        else:
            # Use local path
            if not os.path.exists(repo_path):
                raise Exception("Repository path does not exist")
            actual_repo_path = repo_path
            logger.info(f"Using local repository path: {actual_repo_path}")
        
        logger.info("Starting repository analysis...")
        # Analyze repository with timeout
        try:
            # Add timeout for analysis (10 minutes)
            analysis_task = asyncio.create_task(analyzer.analyze_repository(actual_repo_path, repo_id))
            analysis_result = await asyncio.wait_for(analysis_task, timeout=600)  # 10 minutes
            logger.info(f"Analysis completed: {analysis_result['nodes_created']} nodes, {analysis_result['edges_created']} edges")
        except asyncio.TimeoutError:
            raise Exception("Repository analysis timed out after 10 minutes")
        
        # Clean up temp directory if created
        if repo_path.startswith(('http://', 'https://', 'git@')):
            try:
                shutil.rmtree(actual_repo_path, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory {actual_repo_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")
        
        # Log repository creation
        await storage.append_audit_event(repo_id, AuditEvent(
            user="system",
            action="create_repo",
            entity_type="repo",
            entity_uuid=repo_id,
            details={
                "repo_path": repo_path,
                "analysis_result": analysis_result
            }
        ))
        
        return {
            "repo_id": repo_id,
            "message": f"Repository analyzed successfully. Created {analysis_result['nodes_created']} nodes and {analysis_result['edges_created']} edges.",
            "analysis_result": analysis_result
        }
        
    except Exception as e:
        logger.error(f"Error creating repository: {e}")
        raise Exception(str(e))

@mcp.tool()
async def delete_repository(repo_id: str) -> dict:
    """Delete all data for a repository"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        success = await storage.delete_repo(repo_id)
        if not success:
            raise Exception("Failed to delete repository")
        
        return {"message": f"Repository {repo_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting repository: {e}")
        raise Exception(str(e))

@mcp.tool()
async def create_entity(repo_id: str, file_path: str, entity_type: str, name: str, 
                      start_line: int, end_line: Optional[int] = None, 
                      docstring: Optional[str] = None, tags: Optional[List[str]] = None,
                      created_by: str = "user") -> dict:
    """Create a new entity manually"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        # Create new node
        node = GraphNode(
            repo_id=repo_id,
            file_path=file_path,
            type=entity_type,
            name=name,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            tags=tags or [],
            created_by=created_by,
            manual=True
        )
        
        # Add to graph
        await graph_manager.add_node(repo_id, node)
        
        # Update index
        try:
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
        except Exception as idx_error:
            logger.warning(f"Failed to update index: {idx_error}")
        
        # Log creation
        await storage.append_audit_event(repo_id, AuditEvent(
            user=created_by,
            action="create_entity",
            entity_type="node",
            entity_uuid=node.uuid,
            details={"name": node.name, "type": node.type, "manual": True}
        ))
        
        return node.model_dump()
        
    except Exception as e:
        logger.error(f"Error creating entity: {e}")
        raise Exception(str(e))

@mcp.tool()
async def update_entity(repo_id: str, entity_uuid: str, **update_data) -> dict:
    """Update entity fields"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        # Get existing entity
        existing_node = await graph_manager.get_node(repo_id, entity_uuid)
        if not existing_node:
            raise Exception("Entity not found")
        
        # Update fields
        for field, value in update_data.items():
            if hasattr(existing_node, field) and value is not None:
                setattr(existing_node, field, value)
        
        # Add to history
        existing_node.history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "update",
            "changes": update_data
        })
        
        # Update in graph
        await graph_manager.update_node(repo_id, existing_node)
        
        # Log update
        await storage.append_audit_event(repo_id, AuditEvent(
            user="user",
            action="update_entity",
            entity_type="node",
            entity_uuid=entity_uuid,
            details=update_data
        ))
        
        return existing_node.model_dump()
        
    except Exception as e:
        logger.error(f"Error updating entity: {e}")
        raise Exception(str(e))

@mcp.tool()
async def delete_entity(repo_id: str, entity_uuid: str) -> dict:
    """Delete an entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        # Check if entity exists
        entity = await graph_manager.get_node(repo_id, entity_uuid)
        if not entity:
            raise Exception("Entity not found")
        
        # Remove from graph
        success = await graph_manager.remove_node(repo_id, entity_uuid)
        if not success:
            raise Exception("Failed to delete entity")
        
        return {"message": "Entity deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting entity: {e}")
        raise Exception(str(e))

@mcp.tool()
async def create_relation(repo_id: str, from_uuid: str, to_uuid: str, relation_type: str, created_by: str = "user") -> dict:
    """Create a new relationship manually"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        # Verify both entities exist
        from_node = await graph_manager.get_node(repo_id, from_uuid)
        to_node = await graph_manager.get_node(repo_id, to_uuid)
        
        if not from_node:
            raise Exception("Source entity not found")
        if not to_node:
            raise Exception("Target entity not found")
        
        # Create edge
        edge = GraphEdge(
            from_uuid=from_uuid,
            to_uuid=to_uuid,
            type=relation_type,
            created_by=created_by,
            manual=True
        )
        
        # Add to graph
        await graph_manager.add_edge(repo_id, edge)
        
        # Log creation
        await storage.append_audit_event(repo_id, AuditEvent(
            user=created_by,
            action="create_relation",
            entity_type="edge",
            entity_uuid=edge.uuid,
            details={"type": edge.type, "from_uuid": edge.from_uuid, "to_uuid": edge.to_uuid}
        ))
        
        return edge.model_dump()
        
    except Exception as e:
        logger.error(f"Error creating relation: {e}")
        raise Exception(str(e))

@mcp.tool()
async def retrieve_multiple_entity_code_snippets(repo_id: str, uuids: List[str]) -> dict:
    """Get code and metadata for multiple entities"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        contexts = await context_manager.get_bulk_context(repo_id, uuids)
        return {"contexts": contexts}
        
    except Exception as e:
        logger.error(f"Error getting bulk code: {e}")
        raise Exception(str(e))

@mcp.tool()
async def find_entities_by_probable_names(repo_id: str, probable_names: List[str]) -> dict:
    """Get entities by probable names"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        entities = await search_engine.find_by_probable_names(repo_id, probable_names)
        return {"entities": entities}
        
    except Exception as e:
        logger.error(f"Error getting probable code: {e}")
        raise Exception(str(e))

@mcp.tool()
async def debug_test(message: str = "test") -> dict:
    """Simple debug test function"""
    logger.info(f"Debug test called with message: {message}")
    return {
        "status": "ok",
        "message": f"Debug test successful: {message}",
        "timestamp": datetime.now().isoformat(),
        "server_info": "Knowledge Graph MCP Server v2"
    }

@mcp.tool()
async def ping() -> dict:
    """Simple ping tool to test MCP connectivity"""
    return {
        "status": "ok",
        "message": "MCP server is responding",
        "timestamp": datetime.now().isoformat()
    }

# Repository tools

@mcp.tool("repos://list")
async def list_repositories() -> dict:
    """List all available repositories"""
    try:
        repos = await storage.list_repos()
        return {"repositories": repos}
    except Exception as e:
        logger.error(f"Error listing repositories: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/entities")
async def retrieve_repository_entity_index(repo_id: str) -> dict:
    """Get flat index of all entities with filtering and pagination"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        params = SearchParams(
            name=None,
            type=None,
            file_path=None,
            tags=[],
            limit=50,
            offset=0
        )
        
        entities = await search_engine.search_entities(repo_id, params)
        return {"entities": entities, "count": len(entities)}
        
    except Exception as e:
        logger.error(f"Error getting entities index: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/entity/{entity_uuid}")
async def retrieve_entity_details(repo_id: str, entity_uuid: str) -> dict:
    """Get entity details by UUID"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        entity = await graph_manager.get_node(repo_id, entity_uuid)
        if not entity:
            raise Exception("Entity not found")
        
        return entity.model_dump()
        
    except Exception as e:
        logger.error(f"Error getting entity: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/entity/{entity_uuid}/context")
async def retrieve_entity_full_context(repo_id: str, entity_uuid: str) -> dict:
    """Get full context for an entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        context = await context_manager.get_entity_context(repo_id, entity_uuid)
        if not context:
            raise Exception("Entity not found")
        
        return context.model_dump() if hasattr(context, 'model_dump') else context
        
    except Exception as e:
        logger.error(f"Error getting entity context: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/statistics")
async def calculate_repository_statistics(repo_id: str) -> dict:
    """Get comprehensive repository statistics and metrics"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
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
        
        # Calculate density safely
        density = 0
        if node_count > 1:
            possible_edges = node_count * (node_count - 1)
            if possible_edges > 0:
                density = edge_count / possible_edges
        
        return {
            "basic_metrics": {
                "total_nodes": node_count,
                "total_edges": edge_count,
                "graph_density": density
            },
            "node_distribution": node_types,
            "edge_distribution": edge_types
        }
        
    except Exception as e:
        logger.error(f"Error getting repository statistics: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/entity/{entity_uuid}/neighbors")
async def find_connected_entity_neighbors(repo_id: str, entity_uuid: str) -> dict:
    """Get neighbor entities"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        neighbor_uuids = await graph_manager.get_neighbors(repo_id, entity_uuid, "both")
        neighbors = []
        
        for uuid in neighbor_uuids[:10]:  # Limit to 10
            neighbor = await graph_manager.get_node(repo_id, uuid)
            if neighbor:
                neighbors.append(neighbor.model_dump())
        
        return {"neighbors": neighbors, "total": len(neighbor_uuids)}
        
    except Exception as e:
        logger.error(f"Error getting entity neighbors: {e}")
        raise Exception(str(e))

# Analytics tools

@mcp.tool("repo://{repo_id}/complexity-analysis")
async def measure_repository_code_complexity(repo_id: str) -> dict:
    """Analyze repository complexity and patterns"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
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
        
        return {
            "most_complex_functions": most_complex,
            "complexity_distribution": {
                "low_complexity": len([x for x in complexity_scores.values() if x["complexity_score"] <= 2]),
                "medium_complexity": len([x for x in complexity_scores.values() if 2 < x["complexity_score"] <= 5]),
                "high_complexity": len([x for x in complexity_scores.values() if x["complexity_score"] > 5])
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing complexity: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/hotspots")
async def identify_critical_code_hotspots(repo_id: str) -> dict:
    """Find code hotspots - most important/central code elements"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
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
            "hotspots": hotspots[:10],  # Top 10
            "summary": {
                "total_analyzed": len(hotspots),
                "average_hotspot_score": sum(h["hotspot_score"] for h in hotspots) / len(hotspots) if hotspots else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error finding hotspots: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/dependencies")
async def map_repository_dependencies(repo_id: str) -> dict:
    """Analyze dependency relationships and patterns"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
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
        
        return {
            "summary": {
                "total_nodes": dep_graph.number_of_nodes(),
                "total_import_relationships": dep_graph.number_of_edges(),
                "circular_dependencies_count": len(circular_deps)
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
            ]
        }
        
    except Exception as e:
        logger.error(f"Error analyzing dependencies: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/impact-analysis/{node_id}")
async def analyze_node_change_impact(repo_id: str, node_id: str) -> dict:
    """Analyze the impact of changes to a specific node"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        graph = await graph_manager.get_graph(repo_id)
        
        if not graph.has_node(node_id):
            raise Exception("Node not found")
        
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
                "impact_score": impact_score,
                "risk_level": risk_level
            },
            "direct_dependents": [
                {
                    "node": node,
                    **graph.nodes.get(node, {})
                } for node in list(direct_dependents)[:10]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error analyzing impact: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/language-analysis")
async def detect_repository_programming_languages(repo_id: str) -> dict:
    """Analyze programming languages used in the repository"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
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
        
        return {
            "primary_language": primary_language,
            "language_distribution": language_percentages,
            "file_counts": language_counts,
            "total_files": len(file_paths),
            "total_nodes": total_nodes,
            "file_extensions": file_extensions,
            "analysis_summary": {
                "is_multilingual": len([l for l in language_percentages.values() if l > 10]) > 1,
                "dominant_language_percentage": max(language_percentages.values()) if language_percentages else 0,
                "language_diversity_score": len(language_percentages),
                "total_languages": len(language_counts)
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing languages: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/code-quality-metrics")
async def evaluate_repository_code_quality(repo_id: str) -> dict:
    """Analyze overall code quality metrics"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
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
        
        # Calculate component scores (0-100)
        maintainability_score = min(100, max(0, 100 - (avg_degree * 10)))
        complexity_score = min(100, max(0, 100 - (high_complexity / total_nodes * 100) if total_nodes > 0 else 100))
        documentation_score = documentation_ratio * 100
        
        # Overall score (weighted average)
        overall_score = (
            maintainability_score * 0.4 +
            complexity_score * 0.4 +
            documentation_score * 0.2
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
        
        return {
            "overall_score": round(overall_score, 2),
            "quality_grade": quality_grade,
            "metrics": {
                "maintainability": round(maintainability_score, 2),
                "complexity": round(complexity_score, 2),
                "documentation": round(documentation_score, 2)
            },
            "component_details": {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "function_count": len(function_nodes),
                "class_count": len(class_nodes),
                "average_degree": round(avg_degree, 2),
                "max_degree": max_degree,
                "documentation_ratio": round(documentation_ratio, 3)
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing code quality: {e}")
        raise Exception(str(e))

# Search tools

@mcp.tool("repo://{repo_id}/search")
async def search_entities(repo_id: str) -> dict:
    """Advanced search with fuzzy matching and filtering"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        params = SearchParams(
            name=None,
            type=None,
            file_path=None,
            tags=[],
            limit=50,
            offset=0
        )
        
        results = await search_engine.search_entities(repo_id, params)
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/relations")
async def list_relations(repo_id: str) -> dict:
    """List all relationships with filtering"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        graph = await graph_manager.get_graph(repo_id)
        edges = []
        
        for from_id, to_id, edge_data in graph.edges(data=True):
            edge_info = {
                "from_uuid": from_id,
                "to_uuid": to_id,
                **edge_data
            }
            edges.append(edge_info)
        
        # Apply pagination
        return {"relations": edges[:50], "total": len(edges)}
        
    except Exception as e:
        logger.error(f"Error listing relations: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/entity/{entity_uuid}/subgraph")
async def extract_entity_surrounding_subgraph(repo_id: str, entity_uuid: str) -> dict:
    """Get subgraph around an entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        subgraph = await graph_manager.get_subgraph(repo_id, entity_uuid, 2)
        
        # Convert to serializable format
        nodes = []
        edges = []
        
        for node_id, node_data in subgraph.nodes(data=True):
            nodes.append({
                "uuid": node_id,
                **node_data
            })
        
        for from_id, to_id, edge_data in subgraph.edges(data=True):
            edges.append({
                "from_uuid": from_id,
                "to_uuid": to_id,
                **edge_data
            })
        
        return {"nodes": nodes, "edges": edges}
        
    except Exception as e:
        logger.error(f"Error getting subgraph: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/entity/{entity_uuid}/tree")
async def get_entity_call_inheritance_tree(repo_id: str, entity_uuid: str) -> dict:
    """Get call or inheritance tree for an entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        tree = await context_manager.get_call_tree(repo_id, entity_uuid, "out", 3)
        return {"tree": tree}
        
    except Exception as e:
        logger.error(f"Error getting entity tree: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/audit")
async def retrieve_repository_audit_log(repo_id: str) -> dict:
    """Get audit log for repository"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        events = await storage.load_audit_events(repo_id, 100, 0)
        return {"events": [event.model_dump() if hasattr(event, 'model_dump') else event for event in events]}
        
    except Exception as e:
        logger.error(f"Error getting audit log: {e}")
        raise Exception(str(e))

@mcp.tool("repo://{repo_id}/entity/{entity_uuid}/audit")
async def retrieve_entity_audit_history(repo_id: str, entity_uuid: str) -> dict:
    """Get audit events for a specific entity"""
    try:
        if not await storage.repo_exists(repo_id):
            raise Exception("Repository not found")
        
        all_events = await storage.load_audit_events(repo_id, limit=10000)
        entity_events = [event for event in all_events if getattr(event, 'entity_uuid', None) == entity_uuid]
        
        return {"events": [event.model_dump() if hasattr(event, 'model_dump') else event for event in entity_events]}
        
    except Exception as e:
        logger.error(f"Error getting entity audit: {e}")
        raise Exception(str(e))

async def clone_repository(git_url: str, target_dir: str) -> bool:
    """Clone a git repository asynchronously"""
    try:
        logger.info(f"Starting git clone of {git_url} to {target_dir}")
        
        process = await asyncio.create_subprocess_exec(
            'git', 'clone', '--depth', '1', git_url, target_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        logger.info("Git clone process started, waiting for completion...")
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
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        print(f"Failed to start MCP server: {e}", file=sys.stderr)
        exit(1)
