import os
import json
import pickle
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
import networkx as nx
from datetime import datetime
import aiofiles

from models import GraphNode, GraphEdge, AuditEvent, EntityIndexEntry


class StorageManager:
    """Manages all file-based storage operations for the knowledge graph"""
    
    def __init__(self, base_data_dir: str = "data"):
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(exist_ok=True)
    
    def get_repo_dir(self, repo_id: str) -> Path:
        """Get the directory path for a repository"""
        return self.base_data_dir / repo_id
    
    def get_repo_path(self, repo_id: str) -> Path:
        """Get the repository path (same as get_repo_dir for compatibility)"""
        return self.get_repo_dir(repo_id)
    
    def ensure_repo_dir(self, repo_id: str) -> Path:
        """Ensure repository directory exists"""
        repo_dir = self.get_repo_dir(repo_id)
        repo_dir.mkdir(exist_ok=True)
        return repo_dir
    
    def get_graph_path(self, repo_id: str) -> Path:
        """Get path to graph pickle file"""
        return self.get_repo_dir(repo_id) / "graph.gpickle"
    
    def get_index_path(self, repo_id: str) -> Path:
        """Get path to entities index file"""
        return self.get_repo_dir(repo_id) / "entities_index.json"
    
    def get_audit_path(self, repo_id: str) -> Path:
        """Get path to audit log file"""
        return self.get_repo_dir(repo_id) / "audit.jsonl"
    
    def get_tags_path(self, repo_id: str) -> Path:
        """Get path to tags file"""
        return self.get_repo_dir(repo_id) / "tags.json"
    
    async def load_graph(self, repo_id: str) -> nx.DiGraph:
        """Load graph from pickle file"""
        graph_path = self.get_graph_path(repo_id)
        if not graph_path.exists():
            return nx.DiGraph()
        
        def _load():
            with open(graph_path, 'rb') as f:
                return pickle.load(f)
        
        return await asyncio.get_event_loop().run_in_executor(None, _load)
    
    async def save_graph(self, repo_id: str, graph: nx.DiGraph) -> None:
        """Save graph to pickle file"""
        self.ensure_repo_dir(repo_id)
        graph_path = self.get_graph_path(repo_id)
        
        def _save():
            with open(graph_path, 'wb') as f:
                pickle.dump(graph, f)
        
        await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def load_index(self, repo_id: str) -> List[EntityIndexEntry]:
        """Load entities index from JSON file"""
        index_path = self.get_index_path(repo_id)
        if not index_path.exists():
            return []
        
        def _load():
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [EntityIndexEntry(**item) for item in data]
        
        return await asyncio.get_event_loop().run_in_executor(None, _load)
    
    async def save_index(self, repo_id: str, entities: List[EntityIndexEntry]) -> None:
        """Save entities index to file"""
        index_path = self.get_index_path(repo_id)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [entity.model_dump() for entity in entities]
        async with aiofiles.open(index_path, 'w') as f:
            await f.write(json.dumps(data, indent=2, default=str))
    
    async def add_to_entities_index(self, repo_id: str, entity: EntityIndexEntry) -> None:
        """Add a single entity to the index"""
        # Load existing index
        existing_entities = await self.load_index(repo_id)
        
        # Add new entity (or update if exists)
        entity_exists = False
        for i, existing_entity in enumerate(existing_entities):
            if existing_entity.uuid == entity.uuid:
                existing_entities[i] = entity
                entity_exists = True
                break
        
        if not entity_exists:
            existing_entities.append(entity)
        
        # Save updated index
        await self.save_index(repo_id, existing_entities)
    
    async def append_audit_event(self, repo_id: str, event: AuditEvent) -> None:
        """Append audit event to JSONL file"""
        self.ensure_repo_dir(repo_id)
        audit_path = self.get_audit_path(repo_id)
        
        def _append():
            with open(audit_path, 'a', encoding='utf-8') as f:
                f.write(event.model_dump_json() + '\n')
        
        await asyncio.get_event_loop().run_in_executor(None, _append)
    
    async def load_audit_events(self, repo_id: str, limit: int = 100, offset: int = 0) -> List[AuditEvent]:
        """Load audit events from JSONL file"""
        audit_path = self.get_audit_path(repo_id)
        if not audit_path.exists():
            return []
        
        def _load():
            events = []
            with open(audit_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num < offset:
                        continue
                    if len(events) >= limit:
                        break
                    try:
                        event_data = json.loads(line.strip())
                        events.append(AuditEvent(**event_data))
                    except json.JSONDecodeError:
                        continue
            return events
        
        return await asyncio.get_event_loop().run_in_executor(None, _load)
    
    async def load_tags(self, repo_id: str) -> Dict[str, Any]:
        """Load tags configuration"""
        tags_path = self.get_tags_path(repo_id)
        if not tags_path.exists():
            return {"auto_tags": [], "manual_tags": {}}
        
        def _load():
            with open(tags_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return await asyncio.get_event_loop().run_in_executor(None, _load)
    
    async def save_tags(self, repo_id: str, tags_data: Dict[str, Any]) -> None:
        """Save tags configuration"""
        self.ensure_repo_dir(repo_id)
        tags_path = self.get_tags_path(repo_id)
        
        def _save():
            with open(tags_path, 'w', encoding='utf-8') as f:
                json.dump(tags_data, f, indent=2, ensure_ascii=False)
        
        await asyncio.get_event_loop().run_in_executor(None, _save)
    
    async def delete_repo(self, repo_id: str) -> bool:
        """Delete all repository data"""
        repo_dir = self.get_repo_dir(repo_id)
        if not repo_dir.exists():
            return False
        
        def _delete():
            import shutil
            shutil.rmtree(repo_dir)
            return True
        
        return await asyncio.get_event_loop().run_in_executor(None, _delete)
    
    async def repo_exists(self, repo_id: str) -> bool:
        """Check if repository exists and has been properly initialized"""
        repo_dir = self.get_repo_dir(repo_id)
        if not repo_dir.exists():
            return False
        
        # Check if essential files exist (at least the graph file should exist for a valid repo)
        graph_path = self.get_graph_path(repo_id)
        return graph_path.exists()
    
    async def list_repos(self) -> List[str]:
        """List all available repositories"""
        def _list():
            return [d.name for d in self.base_data_dir.iterdir() if d.is_dir()]
        
        return await asyncio.get_event_loop().run_in_executor(None, _list)


class GraphManager:
    """Manages the NetworkX graph and provides high-level operations"""
    
    def __init__(self, storage: StorageManager):
        self.storage = storage
        self._graphs: Dict[str, nx.DiGraph] = {}
    
    async def get_graph(self, repo_id: str) -> nx.DiGraph:
        """Get graph for repository, loading if necessary"""
        if repo_id not in self._graphs:
            self._graphs[repo_id] = await self.storage.load_graph(repo_id)
        return self._graphs[repo_id]
    
    async def add_node(self, repo_id: str, node: GraphNode) -> None:
        """Add node to graph"""
        graph = await self.get_graph(repo_id)
        graph.add_node(node.uuid, **node.model_dump())
        await self.storage.save_graph(repo_id, graph)
    
    async def update_node(self, repo_id: str, node: GraphNode) -> None:
        """Update node in graph"""
        graph = await self.get_graph(repo_id)
        if node.uuid in graph.nodes:
            graph.nodes[node.uuid].update(node.model_dump())
            await self.storage.save_graph(repo_id, graph)
    
    async def remove_node(self, repo_id: str, node_uuid: str) -> bool:
        """Remove node from graph"""
        graph = await self.get_graph(repo_id)
        if node_uuid in graph.nodes:
            graph.remove_node(node_uuid)
            await self.storage.save_graph(repo_id, graph)
            return True
        return False
    
    async def add_edge(self, repo_id: str, edge: GraphEdge) -> None:
        """Add edge to graph"""
        graph = await self.get_graph(repo_id)
        graph.add_edge(edge.from_uuid, edge.to_uuid, **edge.model_dump())
        await self.storage.save_graph(repo_id, graph)
    
    async def update_edge(self, repo_id: str, edge: GraphEdge) -> None:
        """Update edge in graph"""
        graph = await self.get_graph(repo_id)
        if graph.has_edge(edge.from_uuid, edge.to_uuid):
            graph.edges[edge.from_uuid, edge.to_uuid].update(edge.model_dump())
            await self.storage.save_graph(repo_id, graph)
    
    async def remove_edge(self, repo_id: str, from_uuid: str, to_uuid: str) -> bool:
        """Remove edge from graph"""
        graph = await self.get_graph(repo_id)
        if graph.has_edge(from_uuid, to_uuid):
            graph.remove_edge(from_uuid, to_uuid)
            await self.storage.save_graph(repo_id, graph)
            return True
        return False
    
    async def get_node(self, repo_id: str, node_uuid: str) -> Optional[GraphNode]:
        """Get node by UUID"""
        graph = await self.get_graph(repo_id)
        if node_uuid in graph.nodes:
            node_data = graph.nodes[node_uuid]
            return GraphNode(**node_data)
        return None
    
    async def get_neighbors(self, repo_id: str, node_uuid: str, direction: str = "both") -> List[str]:
        """Get neighbor node UUIDs"""
        graph = await self.get_graph(repo_id)
        if node_uuid not in graph.nodes:
            return []
        
        if direction == "in":
            return list(graph.predecessors(node_uuid))
        elif direction == "out":
            return list(graph.successors(node_uuid))
        else:  # both
            return list(set(graph.predecessors(node_uuid)) | set(graph.successors(node_uuid)))
    
    async def get_edges_for_node(self, repo_id: str, node_uuid: str, direction: str = "both") -> List[GraphEdge]:
        """Get edges connected to a node"""
        graph = await self.get_graph(repo_id)
        edges = []
        
        if direction in ["out", "both"]:
            for _, to_uuid, edge_data in graph.out_edges(node_uuid, data=True):
                edges.append(GraphEdge(**edge_data))
        
        if direction in ["in", "both"]:
            for from_uuid, _, edge_data in graph.in_edges(node_uuid, data=True):
                edges.append(GraphEdge(**edge_data))
        
        return edges
    
    async def get_subgraph(self, repo_id: str, node_uuid: str, depth: int = 1) -> nx.DiGraph:
        """Get subgraph around a node up to specified depth"""
        graph = await self.get_graph(repo_id)
        if node_uuid not in graph.nodes:
            return nx.DiGraph()
        
        # BFS to get nodes within depth
        visited = set()
        queue = [(node_uuid, 0)]
        subgraph_nodes = set()
        
        while queue:
            current_uuid, current_depth = queue.pop(0)
            if current_uuid in visited or current_depth > depth:
                continue
            
            visited.add(current_uuid)
            subgraph_nodes.add(current_uuid)
            
            if current_depth < depth:
                for neighbor in graph.neighbors(current_uuid):
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))
        
        return graph.subgraph(subgraph_nodes).copy()
    
    async def get_nodes(self, repo_id: str) -> List[GraphNode]:
        """Get all nodes in the repository"""
        graph = await self.get_graph(repo_id)
        nodes = []
        for node_uuid, node_data in graph.nodes(data=True):
            try:
                nodes.append(GraphNode(**node_data))
            except Exception as e:
                # Skip invalid nodes
                continue
        return nodes
    
    async def get_edges(self, repo_id: str) -> List[GraphEdge]:
        """Get all edges in the repository"""
        graph = await self.get_graph(repo_id)
        edges = []
        for from_uuid, to_uuid, edge_data in graph.edges(data=True):
            try:
                edges.append(GraphEdge(**edge_data))
            except Exception as e:
                # Skip invalid edges
                continue
        return edges 