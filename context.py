import os
import asyncio
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
from models import GraphNode, GraphEdge, EntityContext
from storage import StorageManager, GraphManager


class ContextManager:
    """Manages context extraction and enrichment for entities"""
    
    def __init__(self, storage: StorageManager, graph_manager: GraphManager):
        self.storage = storage
        self.graph_manager = graph_manager
    
    async def get_entity_context(self, repo_id: str, entity_uuid: str) -> Optional[EntityContext]:
        """Get full context for an entity including code, neighbors, and relationships"""
        node = await self.graph_manager.get_node(repo_id, entity_uuid)
        if not node:
            return None
        
        # Get code snippet
        code_snippet = await self._extract_code_snippet(repo_id, node)
        
        # Get neighbors
        neighbors = await self._get_neighbor_nodes(repo_id, entity_uuid)
        
        # Get relationships
        relationships = await self.graph_manager.get_edges_for_node(repo_id, entity_uuid)
        
        return EntityContext(
            entity=node,
            code_snippet=code_snippet,
            neighbors=neighbors,
            relationships=relationships
        )
    
    async def get_bulk_context(self, repo_id: str, entity_uuids: List[str]) -> List[EntityContext]:
        """Get context for multiple entities efficiently"""
        contexts = []
        
        # Process in parallel for efficiency
        tasks = [self.get_entity_context(repo_id, uuid) for uuid in entity_uuids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, EntityContext):
                contexts.append(result)
        
        return contexts
    
    async def get_code_only(self, repo_id: str, entity_uuids: List[str]) -> Dict[str, str]:
        """Get just the code snippets for multiple entities"""
        result = {}
        
        for uuid in entity_uuids:
            node = await self.graph_manager.get_node(repo_id, uuid)
            if node:
                code = await self._extract_code_snippet(repo_id, node)
                result[uuid] = code or ""
        
        return result
    
    async def _extract_code_snippet(self, repo_id: str, node: GraphNode) -> Optional[str]:
        """Extract code snippet for a node"""
        if not node.start_line or not node.end_line:
            return None
        
        # Construct file path
        repo_dir = self.storage.get_repo_dir(repo_id)
        file_path = repo_dir.parent / node.file_path  # Assume repo files are stored alongside data
        
        # If file doesn't exist in expected location, try to find it
        if not file_path.exists():
            # Try to find the file in the repository
            possible_paths = [
                repo_dir / "source" / node.file_path,
                repo_dir / "repo" / node.file_path,
                Path(node.file_path)
            ]
            
            for path in possible_paths:
                if path.exists():
                    file_path = path
                    break
            else:
                return None
        
        def _read_lines():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    start_idx = max(0, node.start_line - 1)
                    end_idx = min(len(lines), node.end_line)
                    return ''.join(lines[start_idx:end_idx])
            except Exception:
                return None
        
        return await asyncio.get_event_loop().run_in_executor(None, _read_lines)
    
    async def _get_neighbor_nodes(self, repo_id: str, entity_uuid: str, limit: int = 10) -> List[GraphNode]:
        """Get neighbor nodes for an entity"""
        neighbor_uuids = await self.graph_manager.get_neighbors(repo_id, entity_uuid)
        neighbors = []
        
        for uuid in neighbor_uuids[:limit]:  # Limit to avoid too much data
            node = await self.graph_manager.get_node(repo_id, uuid)
            if node:
                neighbors.append(node)
        
        return neighbors
    
    async def get_call_tree(self, repo_id: str, entity_uuid: str, direction: str = "out", max_depth: int = 3) -> Dict[str, Any]:
        """Get call tree for an entity"""
        graph = await self.graph_manager.get_graph(repo_id)
        
        if entity_uuid not in graph.nodes:
            return {}
        
        visited = set()
        tree = await self._build_tree_recursive(repo_id, graph, entity_uuid, direction, max_depth, 0, visited)
        
        return tree
    
    async def _build_tree_recursive(self, repo_id: str, graph, node_uuid: str, direction: str, max_depth: int, current_depth: int, visited: Set[str]) -> Dict[str, Any]:
        """Recursively build call tree"""
        if current_depth >= max_depth or node_uuid in visited:
            return {}
        
        visited.add(node_uuid)
        node_data = graph.nodes.get(node_uuid, {})
        
        tree = {
            'uuid': node_uuid,
            'name': node_data.get('name', 'unknown'),
            'type': node_data.get('type', 'unknown'),
            'file_path': node_data.get('file_path', ''),
            'children': []
        }
        
        # Get connected nodes based on direction
        if direction == "out":
            connected = list(graph.successors(node_uuid))
        elif direction == "in":
            connected = list(graph.predecessors(node_uuid))
        else:  # both
            connected = list(set(graph.successors(node_uuid)) | set(graph.predecessors(node_uuid)))
        
        # Filter by edge type (calls, inherits, etc.)
        call_related_edges = ['calls', 'inherits', 'uses']
        filtered_connected = []
        
        for connected_uuid in connected:
            if direction == "out" and graph.has_edge(node_uuid, connected_uuid):
                edge_data = graph.edges[node_uuid, connected_uuid]
            elif direction == "in" and graph.has_edge(connected_uuid, node_uuid):
                edge_data = graph.edges[connected_uuid, node_uuid]
            else:
                continue
            
            if edge_data.get('type') in call_related_edges:
                filtered_connected.append(connected_uuid)
        
        # Recursively build children
        for child_uuid in filtered_connected[:5]:  # Limit children to avoid explosion
            child_tree = await self._build_tree_recursive(repo_id, graph, child_uuid, direction, max_depth, current_depth + 1, visited.copy())
            if child_tree:
                tree['children'].append(child_tree)
        
        return tree
    
    async def get_dependencies(self, repo_id: str, entity_uuid: str) -> Dict[str, List[GraphNode]]:
        """Get dependencies (imports, calls) for an entity"""
        graph = await self.graph_manager.get_graph(repo_id)
        dependencies = {
            'imports': [],
            'calls': [],
            'uses': [],
            'inherits': []
        }
        
        if entity_uuid not in graph.nodes:
            return dependencies
        
        # Get outgoing edges
        for _, to_uuid, edge_data in graph.out_edges(entity_uuid, data=True):
            edge_type = edge_data.get('type', 'unknown')
            target_node = await self.graph_manager.get_node(repo_id, to_uuid)
            
            if target_node and edge_type in dependencies:
                dependencies[edge_type].append(target_node)
        
        return dependencies
    
    async def get_dependents(self, repo_id: str, entity_uuid: str) -> Dict[str, List[GraphNode]]:
        """Get entities that depend on this entity"""
        graph = await self.graph_manager.get_graph(repo_id)
        dependents = {
            'imported_by': [],
            'called_by': [],
            'used_by': [],
            'inherited_by': []
        }
        
        if entity_uuid not in graph.nodes:
            return dependents
        
        # Get incoming edges
        for from_uuid, _, edge_data in graph.in_edges(entity_uuid, data=True):
            edge_type = edge_data.get('type', 'unknown')
            source_node = await self.graph_manager.get_node(repo_id, from_uuid)
            
            if source_node:
                if edge_type == 'imports':
                    dependents['imported_by'].append(source_node)
                elif edge_type == 'calls':
                    dependents['called_by'].append(source_node)
                elif edge_type == 'uses':
                    dependents['used_by'].append(source_node)
                elif edge_type == 'inherits':
                    dependents['inherited_by'].append(source_node)
        
        return dependents


class CodeExtractor:
    """Utilities for extracting and enriching code snippets"""
    
    @staticmethod
    async def extract_function_signature(file_path: str, start_line: int) -> Optional[str]:
        """Extract just the function signature"""
        def _extract():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if start_line <= len(lines):
                        signature_line = lines[start_line - 1].strip()
                        
                        # Handle multi-line signatures
                        if not signature_line.endswith(':'):
                            for i in range(start_line, min(start_line + 5, len(lines))):
                                next_line = lines[i].strip()
                                signature_line += ' ' + next_line
                                if next_line.endswith(':'):
                                    break
                        
                        return signature_line
            except Exception:
                return None
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract)
    
    @staticmethod
    async def extract_with_context(file_path: str, start_line: int, end_line: int, context_lines: int = 3) -> Optional[Dict[str, str]]:
        """Extract code with surrounding context"""
        def _extract():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    # Calculate bounds with context
                    context_start = max(0, start_line - 1 - context_lines)
                    context_end = min(len(lines), end_line + context_lines)
                    
                    before_context = ''.join(lines[context_start:start_line - 1])
                    main_code = ''.join(lines[start_line - 1:end_line])
                    after_context = ''.join(lines[end_line:context_end])
                    
                    return {
                        'before': before_context,
                        'main': main_code,
                        'after': after_context,
                        'full': before_context + main_code + after_context
                    }
            except Exception:
                return None
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract)
    
    @staticmethod
    def extract_imports_from_code(code: str) -> List[str]:
        """Extract import statements from code"""
        imports = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        
        return imports 