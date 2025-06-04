"""
Knowledge Graph v2 - Core Analysis Module

This module contains the main CodeAnalyzer orchestrator that coordinates
repository analysis using multiple specialized analyzers.
"""

import ast
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from datetime import datetime
import networkx as nx
from collections import defaultdict

from models import GraphNode, GraphEdge, AuditEvent, EntityIndexEntry
from storage import StorageManager, GraphManager

# Configure logging
logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Main orchestrator for code analysis and knowledge graph construction.
    
    This class coordinates multiple specialized analyzers to process repositories
    and build comprehensive knowledge graphs with entities and relationships.
    """
    
    def __init__(self, storage: StorageManager, graph_manager: GraphManager):
        """Initialize the CodeAnalyzer with storage and graph management."""
        self.storage = storage
        self.graph_manager = graph_manager
        
        # Tag patterns for entity classification
        self.tag_patterns = {
            'API': ['api', 'endpoint', 'route', 'handler', 'view'],
            'DATABASE': ['db', 'database', 'model', 'table', 'sql', 'query'],
            'UTILITY': ['util', 'helper', 'common', 'shared'],
            'CONFIG': ['config', 'setting', 'env', 'constant'],
            'TEST': ['test', 'mock', 'fixture'],
            'SERVICE': ['service', 'client', 'manager'],
            'EXCEPTION': ['error', 'exception', 'raise'],
            'ASYNC': ['async', 'await', 'coroutine']
        }
        
        # Global registry for cross-file analysis
        self.global_entities: Dict[str, GraphNode] = {}
        self.global_imports: Dict[str, Dict[str, str]] = {}
        self.call_relationships: List[Tuple[str, str, str, int]] = []
        
        # Advanced analysis structures
        self.defines: Dict[str, Set[str]] = defaultdict(set)
        self.references: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.nx_graph: nx.MultiDiGraph = nx.MultiDiGraph()
    
    async def analyze_repository(self, repo_path: str, repo_id: str) -> Dict[str, Any]:
        """
        Analyze entire repository and build knowledge graph.
        
        Args:
            repo_path: Path to the repository to analyze
            repo_id: Unique identifier for the repository
            
        Returns:
            Dictionary containing analysis results and statistics
        """
        logger.info(f"Starting repository analysis for {repo_id} at {repo_path}")
        
        results = {
            'repo_id': repo_id,
            'nodes_created': 0,
            'edges_created': 0,
            'files_analyzed': 0,
            'errors': []
        }
        
        try:
            # Clear global state for fresh analysis
            self._clear_global_state()
            logger.info("Cleared global state")
            
            # Ensure repository directory exists
            self.storage.ensure_repo_dir(repo_id)
            logger.info(f"Ensured repository directory for {repo_id}")
            
            # Get all source files (Python, JavaScript, etc.)
            logger.info(f"Finding source files in {repo_path}")
            source_files = await self._find_source_files(repo_path)
            logger.info(f"Found {len(source_files)} source files to analyze")
            
            # Dynamic batch sizing based on repository size
            file_count = len(source_files)
            if file_count > 500:
                logger.info(f"Large repository detected ({file_count} files), using optimized settings")
                batch_size = 20  # Larger batches for big repos
            elif file_count > 200:
                logger.info(f"Medium-large repository ({file_count} files), adjusting batch size")
                batch_size = 15
            elif file_count > 100:
                logger.info(f"Medium repository ({file_count} files), standard processing")
                batch_size = 10
            else:
                logger.info(f"Small repository ({file_count} files), fast processing")
                batch_size = 5
            
            # First pass: collect all nodes and imports with parallel processing
            all_nodes = []
            all_edges = []
            
            logger.info(f"Starting parallel analysis of {len(source_files)} files with batch size {batch_size}")
            
            # Process files in parallel batches (batch_size determined above)
            for i in range(0, len(source_files), batch_size):
                batch = source_files[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(source_files) + batch_size - 1)//batch_size}: files {i+1}-{min(i+batch_size, len(source_files))}")
                
                # Create tasks for parallel processing
                tasks = []
                for file_path in batch:
                    relative_path = self._get_relative_path(repo_path, file_path)
                    task = self._analyze_file(file_path, relative_path, repo_id)
                    tasks.append(task)
                
                # Execute batch in parallel
                try:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            file_path = batch[j]
                            logger.error(f"Error analyzing {file_path}: {result}")
                            results['errors'].append(f"Error analyzing {file_path}: {str(result)}")
                        else:
                            nodes, edges = result
                            all_nodes.extend(nodes)
                            all_edges.extend(edges)
                            results['files_analyzed'] += 1
                            
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    results['errors'].append(f"Error processing batch: {str(e)}")
            
            logger.info(f"File analysis complete. Found {len(all_nodes)} nodes and {len(all_edges)} edges")
            
            # Debug information about global state
            logger.info(f"Global state after file analysis:")
            logger.info(f"  - Global entities: {len(self.global_entities)}")
            logger.info(f"  - Global imports: {len(self.global_imports)}")
            logger.info(f"  - Call relationships: {len(self.call_relationships)}")
            
            if self.global_entities:
                logger.info("Sample global entities:")
                for i, (key, entity) in enumerate(list(self.global_entities.items())[:5]):
                    logger.info(f"  {i+1}. {key} -> {entity.name} ({entity.type}) in {entity.file_path}")
            else:
                logger.warning("No global entities registered! This will cause 0 cross-file relationships.")
            
            if self.call_relationships:
                logger.info("Sample call relationships:")
                for i, (caller_file, callee_name, context, line) in enumerate(self.call_relationships[:5]):
                    logger.info(f"  {i+1}. {caller_file} -> {callee_name} (context: {context})")
            else:
                logger.warning("No call relationships found!")
            
            # Build NetworkX graph for advanced analysis
            logger.info("Building NetworkX graph for advanced analysis")
            self._build_networkx_graph(all_nodes, all_edges)
            
            # Apply PageRank ranking
            logger.info("Applying PageRank ranking")
            ranked_entities = self._apply_pagerank_ranking()
            
            # Second pass: create cross-file relationships with ranking
            logger.info("Creating cross-file relationships")
            cross_file_edges = self._create_cross_file_edges_with_ranking(ranked_entities)
            all_edges.extend(cross_file_edges)
            
            logger.info(f"Total edges after cross-file analysis: {len(all_edges)}")
            
            # Add nodes and edges to graph
            logger.info("Persisting analysis results to storage")
            await self._persist_analysis_results(repo_id, all_nodes, all_edges, results)
            
            # Build and save index
            logger.info("Building entities index")
            await self._build_entities_index(repo_id, all_nodes)
            
        except Exception as e:
            logger.error(f"Repository analysis error: {e}")
            results['errors'].append(f"Repository analysis error: {str(e)}")
        
        return results
    
    def _clear_global_state(self) -> None:
        """Clear all global analysis state for fresh repository analysis."""
        self.global_entities.clear()
        self.global_imports.clear()
        self.call_relationships.clear()
        self.defines.clear()
        self.references.clear()
        self.nx_graph.clear()
    
    async def _find_source_files(self, repo_path: str) -> List[str]:
        """Find all source files in repository (Python, JavaScript, etc.)."""
        def _find():
            source_files = []
            # Supported file extensions
            extensions = {'.py', '.js', '.jsx', '.ts', '.tsx'}
            
            # Directories to skip completely
            skip_dirs = {
                '__pycache__', 'venv', 'env', 'node_modules', '.git', 
                '.pytest_cache', '.tox', 'dist', 'build', '.egg-info',
                'htmlcov', '.coverage', 'docs', 'documentation', 'examples'
            }
            
            # File patterns to skip
            skip_files = {
                'test_', 'tests_', '_test.py', '_tests.py', 'conftest.py',
                'setup.py', '__init__.py'  # Skip empty __init__.py files
            }
            
            for root, dirs, files in os.walk(repo_path):
                # Skip common directories to ignore
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in skip_dirs]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file)
                    
                    if ext.lower() in extensions:
                        # Skip test files and other non-essential files
                        should_skip = False
                        for pattern in skip_files:
                            if pattern in file.lower():
                                should_skip = True
                                break
                        
                        # Skip __init__.py if it's very small (likely empty)
                        if file == '__init__.py':
                            try:
                                if os.path.getsize(file_path) < 100:  # Less than 100 bytes
                                    should_skip = True
                            except:
                                pass
                        
                        if not should_skip:
                            source_files.append(file_path)
            
            # For very large repositories, prioritize core files
            if len(source_files) > 1000:
                logger.info(f"Very large repository ({len(source_files)} files), applying smart prioritization")
                source_files = self._prioritize_files(source_files)
            
            return source_files
        
        return await asyncio.get_event_loop().run_in_executor(None, _find)
    
    def _get_relative_path(self, repo_path: str, file_path: str) -> str:
        """Get relative path from repository root."""
        return os.path.relpath(file_path, repo_path).replace('\\', '/')
    
    async def _analyze_file(self, file_path: str, relative_path: str, repo_id: str) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Analyze a single source file using appropriate parsers.
        """
        def _parse_file():
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Determine file type
                _, ext = os.path.splitext(file_path)
                
                if ext.lower() == '.py':
                    # Use Hybrid parser for Python files (AST + Tree-sitter)
                    from .syntax_engine import HybridParser
                    parser = HybridParser(
                        file_path=relative_path,
                        repo_id=repo_id,
                        content=content,
                        tag_patterns=self.tag_patterns,
                        global_entities=self.global_entities,
                        global_imports=self.global_imports,
                        call_relationships=self.call_relationships
                    )
                    
                    # Parse and extract entities
                    result = parser.parse()
                    
                    # Update global call relationships
                    self.call_relationships.extend(result.calls)
                    
                    # CRITICAL: Update global state from parser
                    # The parser modifies its own copies, we need to sync back
                    self.global_entities.update(parser.global_entities)
                    self.global_imports.update(parser.global_imports)
                    
                    return result.nodes, result.edges
                
                else:
                    # Use MultiLanguageAnalyzer for non-Python files
                    from .language_analyzer import MultiLanguageAnalyzer
                    
                    analyzer = MultiLanguageAnalyzer()
                    result = analyzer.analyze_file(relative_path, content, repo_id)
                    
                    # Extract nodes from result
                    nodes = []
                    if result and "parse_result" in result and result["parse_result"]:
                        parse_result = result["parse_result"]
                        if hasattr(parse_result, 'nodes'):
                            nodes = parse_result.nodes
                        elif isinstance(parse_result, dict) and 'nodes' in parse_result:
                            nodes = parse_result['nodes']
                    
                    # No edges for now from multi-language analyzer
                    return nodes, []
                
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                return [], []
        
        return await asyncio.get_event_loop().run_in_executor(None, _parse_file)
    
    async def _persist_analysis_results(
        self, 
        repo_id: str, 
        all_nodes: List[GraphNode], 
        all_edges: List[GraphEdge], 
        results: Dict[str, Any]
    ) -> None:
        """Persist analysis results to storage with batch processing."""
        
        # Dynamic batch sizing based on total entities
        total_entities = len(all_nodes) + len(all_edges)
        if total_entities > 5000:
            node_batch_size = 100
            edge_batch_size = 100
        elif total_entities > 2000:
            node_batch_size = 75
            edge_batch_size = 75
        else:
            node_batch_size = 50
            edge_batch_size = 50
            
        logger.info(f"Using storage batch sizes: nodes={node_batch_size}, edges={edge_batch_size}")
        
        # Process nodes in batches
        for i in range(0, len(all_nodes), node_batch_size):
            batch = all_nodes[i:i + node_batch_size]
            
            # Create tasks for parallel node adding
            node_tasks = []
            audit_tasks = []
            
            for node in batch:
                node_tasks.append(self.graph_manager.add_node(repo_id, node))
                audit_tasks.append(self.storage.append_audit_event(repo_id, AuditEvent(
                    user="analyzer",
                    action="add_node",
                    entity_type="node",
                    entity_uuid=node.uuid,
                    details={"file_path": node.file_path, "type": node.type, "name": node.name}
                )))
            
            # Execute batch in parallel
            await asyncio.gather(*node_tasks, return_exceptions=True)
            await asyncio.gather(*audit_tasks, return_exceptions=True)
            results['nodes_created'] += len(batch)
        
        # Process edges in batches
        for i in range(0, len(all_edges), edge_batch_size):
            batch = all_edges[i:i + edge_batch_size]
            
            # Create tasks for parallel edge adding
            edge_tasks = []
            audit_tasks = []
            
            for edge in batch:
                edge_tasks.append(self.graph_manager.add_edge(repo_id, edge))
                audit_tasks.append(self.storage.append_audit_event(repo_id, AuditEvent(
                    user="analyzer",
                    action="add_edge",
                    entity_type="edge",
                    entity_uuid=edge.uuid,
                    details={"type": edge.type, "from_uuid": edge.from_uuid, "to_uuid": edge.to_uuid}
                )))
            
            # Execute batch in parallel
            await asyncio.gather(*edge_tasks, return_exceptions=True)
            await asyncio.gather(*audit_tasks, return_exceptions=True)
            results['edges_created'] += len(batch)
    
    async def _build_entities_index(self, repo_id: str, nodes: List[GraphNode]) -> None:
        """Build searchable index of entities."""
        for node in nodes:
            index_entry = EntityIndexEntry(
                uuid=node.uuid,
                name=node.name,
                type=node.type,
                file_path=node.file_path,
                start_line=node.start_line,
                tags=node.tags,
                short_doc=node.docstring[:100] if node.docstring else None
            )
            await self.storage.add_to_entities_index(repo_id, index_entry)
    
    def _build_networkx_graph(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> None:
        """Build NetworkX graph for advanced analysis algorithms."""
        # Add nodes to NetworkX graph
        for node in nodes:
            node_attrs = {
                'name': node.name,
                'type': node.type,
                'file_path': node.file_path,
                'tags': node.tags,
                'complexity': node.metadata.get('complexity', 1)
            }
            self.nx_graph.add_node(node.uuid, **node_attrs)
        
        # Add edges to NetworkX graph
        for edge in edges:
            edge_attrs = {
                'type': edge.type,
                'weight': self._calculate_edge_weight(edge, None, None)
            }
            self.nx_graph.add_edge(edge.from_uuid, edge.to_uuid, **edge_attrs)
    
    def _calculate_edge_weight(self, edge: GraphEdge, source: Optional[GraphNode], target: Optional[GraphNode]) -> float:
        """Calculate edge weight for graph algorithms."""
        base_weight = 1.0
        
        # Weight by relationship type
        type_weights = {
            'calls': 1.0,
            'imports': 0.8,
            'inherits': 1.2,
            'defines': 0.6,
            'uses': 0.4
        }
        
        weight = base_weight * type_weights.get(edge.type, 1.0)
        
        # Additional weight factors could be added here
        return weight
    
    def _apply_pagerank_ranking(self) -> Dict[str, float]:
        """Apply fast importance ranking using simple heuristics instead of PageRank."""
        if len(self.nx_graph.nodes) == 0:
            return {}
        
        node_count = len(self.nx_graph.nodes)
        logger.info(f"Applying fast importance ranking for {node_count} entities")
        
        # Always use fast heuristic ranking - much faster than PageRank
        return self._fast_heuristic_ranking()
    
    def _fast_heuristic_ranking(self) -> Dict[str, float]:
        """Ultra-fast importance ranking using call frequency and simple heuristics."""
        scores = {}
        
        # Count how many times each entity is called
        call_counts = defaultdict(int)
        
        # Count calls from call_relationships
        for caller_file, callee_name, context, line in self.call_relationships:
            # Skip "CONTAINS" relationships as they're not actual calls
            if not callee_name.startswith("CONTAINS:"):
                call_counts[callee_name] += 1
        
        # Score each node in the NetworkX graph
        for node_uuid in self.nx_graph.nodes:
            node_data = self.nx_graph.nodes[node_uuid]
            name = node_data.get('name', '')
            entity_type = node_data.get('type', '')
            file_path = node_data.get('file_path', '')
            
            # Base score starts at 0.1
            score = 0.1
            
            # 1. Call frequency bonus (most important factor)
            call_count = call_counts.get(name, 0)
            score += call_count * 0.2  # Each call adds 0.2 to score
            
            # 2. Entity type bonus
            type_scores = {
                'function': 0.3,    # Functions are usually more important
                'class': 0.4,       # Classes often central to design
                'module': 0.2,      # Modules are structural
                'import': 0.1       # Imports are supportive
            }
            score += type_scores.get(entity_type, 0.1)
            
            # 3. File position bonus (entities in main files more important)
            file_name = os.path.basename(file_path).lower()
            if any(keyword in file_name for keyword in ['main', 'app', 'core', 'manager']):
                score += 0.3
            elif any(keyword in file_name for keyword in ['api', 'server', 'client']):
                score += 0.2
            elif any(keyword in file_name for keyword in ['util', 'helper', 'common']):
                score += 0.1
            
            # 4. Name-based heuristics
            if name.lower() in ['main', 'init', 'setup', 'start', 'run']:
                score += 0.2  # Entry points are important
            elif name.startswith('_') and not name.startswith('__'):
                score -= 0.1  # Private functions slightly less important
            elif name.startswith('test_'):
                score -= 0.2  # Test functions less important for core logic
            
            # 5. Graph connectivity bonus (in-degree from NetworkX)
            in_degree = self.nx_graph.in_degree(node_uuid)
            score += min(in_degree * 0.1, 0.5)  # Cap the bonus at 0.5
            
            scores[node_uuid] = score
        
        # Normalize scores to 0-1 range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {uuid: score / max_score for uuid, score in scores.items()}
        
        # Log top entities
        sorted_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Fast heuristic ranking completed. Top entities:")
        for uuid, score in sorted_entities[:5]:
            node_data = self.nx_graph.nodes.get(uuid, {})
            name = node_data.get('name', 'Unknown')
            entity_type = node_data.get('type', 'unknown')
            call_count = call_counts.get(name, 0)
            logger.info(f"  {name} ({entity_type}): {score:.4f} (calls: {call_count})")
        
        return scores
    
    def _create_cross_file_edges_with_ranking(self, ranked_entities: Dict[str, float]) -> List[GraphEdge]:
        """Create cross-file relationships using PageRank scores."""
        edges = []
        
        logger.info(f"Starting cross-file analysis with {len(self.call_relationships)} call relationships")
        logger.info(f"Global entities count: {len(self.global_entities)}")
        
        # Debug: Log first few call relationships
        if self.call_relationships:
            logger.info("Sample call relationships:")
            for i, (caller_file, callee_name, context, line) in enumerate(self.call_relationships[:5]):
                logger.info(f"  {i+1}. {caller_file} -> {callee_name} (context: {context}, line: {line})")
        else:
            logger.warning("No call relationships found!")
            return edges
        
        # PERFORMANCE FIX: Limit the number of call relationships we process
        # For very large repositories, processing all relationships can be extremely slow
        max_relationships = 10000  # Process maximum 10,000 relationships
        
        call_relationships_to_process = self.call_relationships
        if len(self.call_relationships) > max_relationships:
            logger.info(f"Large number of call relationships ({len(self.call_relationships)}), limiting to {max_relationships} for performance")
            
            # Prioritize non-CONTAINS relationships
            non_contains = [rel for rel in self.call_relationships if not rel[1].startswith("CONTAINS:")]
            contains = [rel for rel in self.call_relationships if rel[1].startswith("CONTAINS:")]
            
            # Take up to max_relationships, prioritizing actual calls over contains
            if len(non_contains) >= max_relationships:
                call_relationships_to_process = non_contains[:max_relationships]
            else:
                remaining = max_relationships - len(non_contains)
                call_relationships_to_process = non_contains + contains[:remaining]
            
            logger.info(f"Processing {len(call_relationships_to_process)} prioritized relationships ({len([r for r in call_relationships_to_process if not r[1].startswith('CONTAINS:')])} actual calls)")
        
        successful_matches = 0
        failed_caller_lookups = 0
        failed_callee_lookups = 0
        
        for caller_file, callee_name, context, line in call_relationships_to_process:
            # Find caller node
            caller_node = self._find_caller_node(caller_file, context, line)
            if not caller_node:
                failed_caller_lookups += 1
                logger.debug(f"Could not find caller node: {caller_file}:{context}")
                continue
            
            # Find callee node using multiple strategies
            callee_node = self._find_entity_by_name_advanced(callee_name)
            if not callee_node:
                # Try resolving through imports
                callee_node = self._resolve_through_imports(caller_file, callee_name)
            
            if not callee_node:
                failed_callee_lookups += 1
                logger.debug(f"Could not find callee node: {callee_name}")
                continue
            
            if callee_node and caller_node.uuid != callee_node.uuid:
                # Create cross-file relationship
                edge = GraphEdge(
                    from_uuid=caller_node.uuid,
                    to_uuid=callee_node.uuid,
                    type="calls",
                    properties={
                        "line": line,
                        "context": context,
                        "cross_file": True,
                        "caller_rank": ranked_entities.get(caller_node.uuid, 0.0),
                        "callee_rank": ranked_entities.get(callee_node.uuid, 0.0)
                    },
                    created_by="system",
                    created_at=datetime.now()
                )
                edges.append(edge)
                successful_matches += 1
        
        logger.info(f"Cross-file analysis complete:")
        logger.info(f"  - Successful matches: {successful_matches}")
        logger.info(f"  - Failed caller lookups: {failed_caller_lookups}")
        logger.info(f"  - Failed callee lookups: {failed_callee_lookups}")
        logger.info(f"  - Created {len(edges)} cross-file relationships")
        
        return edges
    
    def _find_caller_node(self, file_path: str, context: str, line_num: int) -> Optional[GraphNode]:
        """Find the caller node within a specific file context with improved resolution."""
        # Strategy 1: Exact match with file prefix
        caller_key = f"{file_path}:{context}"
        if caller_key in self.global_entities:
            return self.global_entities[caller_key]
        
        # Strategy 2: Try context name directly (for functions/classes in same file)
        if context in self.global_entities:
            node = self.global_entities[context]
            if node.file_path == file_path:
                return node
        
        # Strategy 3: Module-qualified context
        module_name = file_path.replace('.py', '').replace('/', '.').replace('\\', '.')
        module_qualified = f"{module_name}.{context}"
        if module_qualified in self.global_entities:
            return self.global_entities[module_qualified]
        
        # Strategy 4: Try to find any node in this file with this name
        for key, entity in self.global_entities.items():
            if entity.file_path == file_path and entity.name == context:
                return entity
        
        # Strategy 5: Fall back to module node
        module_key = f"{file_path}:{module_name}"
        if module_key in self.global_entities:
            return self.global_entities[module_key]
        
        # Strategy 6: Try module name directly  
        if module_name in self.global_entities:
            return self.global_entities[module_name]
        
        return None
    
    def _find_entity_by_name_advanced(self, name: str) -> Optional[GraphNode]:
        """Find entity using advanced name resolution strategies with ranking."""
        # Strategy 1: Direct match
        if name in self.global_entities:
            return self.global_entities[name]
        
        # Strategy 2: Try partial matches (for method calls like obj.method)
        if '.' in name:
            method_name = name.split('.')[-1]
            if method_name in self.global_entities:
                return self.global_entities[method_name]
            
            # Try the full qualified name
            for key in self.global_entities:
                if key.endswith(name) or key.endswith(f".{name}"):
                    return self.global_entities[key]
        
        # Strategy 3: Pattern matching with priorities
        candidates = []
        for full_name, entity in self.global_entities.items():
            score = 0
            
            # Exact name match gets highest score
            if entity.name == name:
                score = 100
            # File prefix match
            elif full_name.endswith(f":{name}"):
                score = 90
            # Module qualified match
            elif full_name.endswith(f".{name}"):
                score = 80
            # Substring match in name
            elif name in entity.name:
                score = 60
            # Substring match in full key
            elif name in full_name:
                score = 40
            
            if score > 0:
                candidates.append((score, entity))
        
        # Return highest scoring candidate
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        return None
    
    def _resolve_through_imports(self, caller_file: str, callee_name: str) -> Optional[GraphNode]:
        """Resolve function calls through import statements."""
        if caller_file not in self.global_imports:
            return None
        
        imports = self.global_imports[caller_file]
        
        # Direct import resolution
        if callee_name in imports:
            full_name = imports[callee_name]
            if full_name in self.global_entities:
                return self.global_entities[full_name]
        
        # Method call resolution (obj.method)
        if '.' in callee_name:
            obj_name, method_name = callee_name.split('.', 1)
            if obj_name in imports:
                module_name = imports[obj_name]
                qualified_name = f"{module_name}.{method_name}"
                if qualified_name in self.global_entities:
                    return self.global_entities[qualified_name]
        
        return None

    def _prioritize_files(self, source_files: List[str]) -> List[str]:
        """Smart prioritization for large repositories - analyze important files first"""
        
        # Priority scoring system
        priority_files = []
        
        for file_path in source_files:
            file_name = os.path.basename(file_path)
            dir_path = os.path.dirname(file_path)
            
            priority = 0
            
            # Core application files get highest priority
            if any(keyword in file_name.lower() for keyword in ['main', 'app', 'core', 'base', 'manager']):
                priority += 100
            
            # API and interface files
            if any(keyword in file_name.lower() for keyword in ['api', 'client', 'server', 'handler', 'controller']):
                priority += 80
            
            # Model and data files
            if any(keyword in file_name.lower() for keyword in ['model', 'schema', 'entity', 'data']):
                priority += 70
            
            # Service and business logic
            if any(keyword in file_name.lower() for keyword in ['service', 'logic', 'processor', 'engine']):
                priority += 60
            
            # Utility files get medium priority
            if any(keyword in file_name.lower() for keyword in ['util', 'helper', 'common', 'shared']):
                priority += 40
            
            # Files in root or src directories get bonus
            if dir_path.count(os.sep) <= 2:  # Root level or one level deep
                priority += 30
            
            # Penalize deeply nested files
            if dir_path.count(os.sep) > 5:
                priority -= 20
            
            # Larger files likely more important (but cap it)
            try:
                file_size = os.path.getsize(file_path)
                size_bonus = min(20, file_size // 1000)  # 1 point per KB, max 20
                priority += size_bonus
            except:
                pass
            
            priority_files.append((file_path, priority))
        
        # Sort by priority (highest first) and return first 80% of files
        priority_files.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 80% of files, but ensure we don't go below 100 files
        target_count = max(100, int(len(priority_files) * 0.8))
        selected_files = [fp for fp, _ in priority_files[:target_count]]
        
        logger.info(f"Prioritized {len(selected_files)} files out of {len(source_files)} "
                   f"(top {len(selected_files)/len(source_files)*100:.1f}%)")
        
        return selected_files


# Make CodeAnalyzer available at module level
__all__ = ['CodeAnalyzer'] 