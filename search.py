import re
from typing import List, Dict, Optional, Set, Any
from difflib import SequenceMatcher
from models import GraphNode, EntityIndexEntry, SearchParams
from storage import StorageManager, GraphManager


class SearchEngine:
    """Advanced search functionality for the knowledge graph"""
    
    def __init__(self, storage: StorageManager, graph_manager: GraphManager):
        self.storage = storage
        self.graph_manager = graph_manager
    
    async def search_entities(self, repo_id: str, params: SearchParams) -> List[EntityIndexEntry]:
        """Advanced entity search with filtering and fuzzy matching"""
        # Load the index
        index = await self.storage.load_index(repo_id)
        
        # Apply filters
        filtered_entities = self._apply_filters(index, params)
        
        # Apply fuzzy matching if name search is specified
        if params.name:
            filtered_entities = self._fuzzy_search_by_name(filtered_entities, params.name)
        
        # Sort by relevance
        filtered_entities = self._sort_by_relevance(filtered_entities, params)
        
        # Apply pagination
        start = params.offset
        end = start + params.limit
        return filtered_entities[start:end]
    
    async def autocomplete(self, repo_id: str, query: str, limit: int = 10) -> List[str]:
        """Provide autocomplete suggestions for entity names"""
        index = await self.storage.load_index(repo_id)
        
        suggestions = []
        query_lower = query.lower()
        
        for entity in index:
            if query_lower in entity.name.lower():
                suggestions.append(entity.name)
            
            if len(suggestions) >= limit:
                break
        
        return sorted(suggestions)
    
    async def find_by_probable_names(self, repo_id: str, probable_names: List[str]) -> List[GraphNode]:
        """Find entities by probable names like 'src/file.py:function_name'"""
        results = []
        graph = await self.graph_manager.get_graph(repo_id)
        
        for probable_name in probable_names:
            matches = await self._find_by_probable_name(repo_id, probable_name, graph)
            results.extend(matches)
        
        return results
    
    def _apply_filters(self, entities: List[EntityIndexEntry], params: SearchParams) -> List[EntityIndexEntry]:
        """Apply type, file_path, and tag filters"""
        filtered = entities
        
        # Filter by type
        if params.type:
            filtered = [e for e in filtered if e.type == params.type]
        
        # Filter by file path
        if params.file_path:
            filtered = [e for e in filtered if params.file_path in e.file_path]
        
        # Filter by tags
        if params.tags:
            filtered = [e for e in filtered if any(tag in e.tags for tag in params.tags)]
        
        return filtered
    
    def _fuzzy_search_by_name(self, entities: List[EntityIndexEntry], query: str) -> List[EntityIndexEntry]:
        """Perform fuzzy name matching"""
        query_lower = query.lower()
        scored_entities = []
        
        for entity in entities:
            name_lower = entity.name.lower()
            
            # Exact match gets highest score
            if query_lower == name_lower:
                score = 1.0
            # Starts with query gets high score
            elif name_lower.startswith(query_lower):
                score = 0.9
            # Contains query gets medium score
            elif query_lower in name_lower:
                score = 0.7
            # Fuzzy match using SequenceMatcher
            else:
                ratio = SequenceMatcher(None, query_lower, name_lower).ratio()
                if ratio >= 0.6:  # Threshold for fuzzy match
                    score = ratio * 0.5  # Lower score for fuzzy matches
                else:
                    continue
            
            scored_entities.append((entity, score))
        
        # Sort by score descending
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, _ in scored_entities]
    
    def _sort_by_relevance(self, entities: List[EntityIndexEntry], params: SearchParams) -> List[EntityIndexEntry]:
        """Sort entities by relevance based on search parameters"""
        def relevance_score(entity: EntityIndexEntry) -> float:
            score = 0.0
            
            # Prefer certain types
            type_scores = {
                'function': 1.0,
                'class': 0.9,
                'async_function': 1.0,
                'import': 0.3,
                'variable': 0.5
            }
            score += type_scores.get(entity.type, 0.5)
            
            # Prefer entities with docstrings
            if entity.short_doc:
                score += 0.2
            
            # Prefer entities with useful tags
            useful_tags = {'API', 'SERVICE', 'DATABASE', 'UTILITY'}
            if any(tag in useful_tags for tag in entity.tags):
                score += 0.3
            
            # Prefer shorter file paths (closer to root)
            path_depth = entity.file_path.count('/')
            score += max(0, (5 - path_depth) * 0.1)
            
            return score
        
        return sorted(entities, key=relevance_score, reverse=True)
    
    async def _find_by_probable_name(self, repo_id: str, probable_name: str, graph) -> List[GraphNode]:
        """Find entity by probable name format like 'file.py:function' or 'module.function'"""
        results = []
        
        # Parse probable name
        if ':' in probable_name:
            file_part, entity_part = probable_name.split(':', 1)
        else:
            file_part = ""
            entity_part = probable_name
        
        # Search in graph nodes
        for node_id, node_data in graph.nodes(data=True):
            node = GraphNode(**node_data)
            
            # Check if entity name matches
            name_match = (entity_part == node.name or 
                         entity_part in node.name or 
                         self._fuzzy_match(entity_part, node.name))
            
            if not name_match:
                continue
            
            # Check file path if specified
            if file_part:
                file_match = (file_part in node.file_path or 
                             node.file_path.endswith(file_part) or
                             self._fuzzy_match(file_part, node.file_path))
                if not file_match:
                    continue
            
            results.append(node)
        
        return results
    
    def _fuzzy_match(self, query: str, target: str, threshold: float = 0.6) -> bool:
        """Check if two strings match with fuzzy logic"""
        ratio = SequenceMatcher(None, query.lower(), target.lower()).ratio()
        return ratio >= threshold 