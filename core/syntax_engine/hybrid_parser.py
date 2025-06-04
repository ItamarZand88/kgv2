"""
Hybrid Parser combining AST and Tree-sitter Analysis

This module provides hybrid parsing that combines AST and tree-sitter
analysis for comprehensive code understanding (potpie approach).
"""

import ast
import logging
from typing import List, Dict, Optional, Tuple, Any

from models import GraphNode, GraphEdge
from .base_parser import BaseParser, ParseResult

logger = logging.getLogger(__name__)


class HybridParser(BaseParser, ast.NodeVisitor):
    """
    Enhanced AST parser with advanced relationship detection.
    
    This parser provides sophisticated Python code analysis using 
    improved AST traversal with better cross-file relationship tracking.
    """
    
    def __init__(
        self, 
        file_path: str, 
        repo_id: str, 
        content: str,
        tag_patterns: Optional[Dict[str, List[str]]] = None,
        global_entities: Optional[Dict[str, GraphNode]] = None,
        global_imports: Optional[Dict[str, Dict[str, str]]] = None,
        call_relationships: Optional[List[Tuple[str, str, str, int]]] = None
    ):
        """Initialize hybrid parser with both AST and tree-sitter capabilities."""
        super().__init__(file_path, repo_id, content)
        
        self.lines = content.split('\n')
        self.tag_patterns = tag_patterns or self._default_tag_patterns()
        
        # Global registries (shared across all files)
        self.global_entities = global_entities or {}
        self.global_imports = global_imports or {}
        self.call_relationships = call_relationships or []
        
        # Local state for this file
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.call_graph: List[Tuple[str, str, int]] = []  # (caller_context, callee, line_num)
        
        # Note: This parser now focuses on enhanced AST analysis
        # Tree-sitter integration can be added later if needed
    
    def supports_language(self, language: str) -> bool:
        """Check if this parser supports the given language."""
        return language == 'python'  # Currently supports Python primarily
    
    def parse(self) -> ParseResult:
        """Parse using hybrid approach: tree-sitter + AST fallback."""
        try:
            tree = ast.parse(self.content)
            return self._analyze_hybrid(tree)
        except SyntaxError as e:
            logger.error(f"Syntax error in {self.file_path}: {e}")
            return ParseResult([], [], {}, [])
        except Exception as e:
            logger.error(f"Error parsing {self.file_path}: {e}")
            return ParseResult([], [], {}, [])
    
    def _analyze_hybrid(self, tree: ast.AST) -> ParseResult:
        """Enhanced AST analysis with improved relationship detection."""
        logger.info(f"Enhanced AST analysis: {self.file_path}")
        
        # Run comprehensive AST analysis
        self.visit(tree)
        
        # Create module node
        module_node = self._create_node(
            name=self.file_path.replace('.py', '').replace('/', '.').replace('\\', '.'),
            node_type="module",
            start_line=1,
            end_line=len(self.lines),
            tags=["MODULE", "HYBRID"],
            metadata={
                "lines_count": len(self.lines),
                "parser": "hybrid"
            }
        )
        self.nodes.append(module_node)
        
        # Register imports and nodes globally - DO THIS AFTER ALL NODES ARE CREATED
        self.global_imports[self.file_path] = self.imports.copy()
        self._register_nodes_globally()
        
        # Debug: Log what we registered
        logger.info(f"Registered {len(self.nodes)} nodes from {self.file_path} to global registry")
        logger.info(f"Global entities now has {len(self.global_entities)} total entities")
        
        # Create local call edges
        self._create_call_edges()
        
        logger.info(f"Enhanced AST analysis complete: {len(self.nodes)} nodes, {len(self.edges)} edges")
        
        return ParseResult(
            nodes=self.nodes,
            edges=self.edges,
            imports=self.imports,
            calls=self.call_relationships
        )
    
    def visit_Import(self, node: ast.Import) -> None:
        """Handle import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = alias.name
            
            # Create import node
            import_node = self._create_node(
                name=alias.name,
                node_type="import",
                start_line=node.lineno,
                end_line=node.lineno,
                tags=self._generate_tags(alias.name, "import")
            )
            self.nodes.append(import_node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle from ... import statements."""
        module = node.module or ""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            full_name = f"{module}.{alias.name}" if module else alias.name
            self.imports[name] = full_name
            
            # Also register the simple name for cross-file resolution
            if module:
                self.imports[alias.name] = full_name
            
            # Create import node
            import_node = self._create_node(
                name=full_name,
                node_type="import",
                start_line=node.lineno,
                end_line=node.lineno,
                tags=self._generate_tags(full_name, "import")
            )
            self.nodes.append(import_node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle function definitions."""
        old_function = self.current_function
        self.current_function = node.name
        
        func_node = self._create_function_node(node)
        self.nodes.append(func_node)
        
        # Continue visiting child nodes
        self.generic_visit(node)
        
        self.current_function = old_function
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions."""
        old_function = self.current_function
        self.current_function = node.name
        
        func_node = self._create_function_node(node, is_async=True)
        self.nodes.append(func_node)
        
        # Continue visiting child nodes
        self.generic_visit(node)
        
        self.current_function = old_function
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        
        class_node = self._create_node(
            name=node.name,
            node_type="class",
            start_line=node.lineno,
            end_line=self._get_end_line(node),
            tags=self._generate_tags(node.name, "class"),
            metadata={
                "docstring": ast.get_docstring(node),
                "base_classes": [self._extract_base_class_name(base) for base in node.bases]
            }
        )
        self.nodes.append(class_node)
        
        # Add inheritance relationships
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.call_relationships.append((self.file_path, base.id, "inheritance", node.lineno))
        
        # Continue visiting child nodes
        self.generic_visit(node)
        
        self.current_class = old_class
    
    def visit_Call(self, node: ast.Call) -> None:
        """Handle function calls with enhanced detection."""
        caller_context = self.current_function or self.current_class or "module"
        
        # Filter out irrelevant built-in and external calls
        builtin_functions = {
            'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 
            'tuple', 'bool', 'range', 'enumerate', 'zip', 'min', 'max', 
            'sum', 'abs', 'round'
        }
        external_modules = {
            'random', 'os', 'sys', 'logging', 'json', 'requests', 
            'flask', 'django'
        }
        
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Skip built-in functions
            if func_name not in builtin_functions:
                self.call_graph.append((caller_context, func_name, node.lineno))
                # Register for cross-file analysis
                self.call_relationships.append((self.file_path, func_name, caller_context, node.lineno))
                
        elif isinstance(node.func, ast.Attribute):
            # Handle method calls like obj.method() or module.function()
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr
                full_call = f"{obj_name}.{method_name}"
                
                # Skip external module calls unless they might be internal
                if obj_name not in external_modules or obj_name in self.imports:
                    self.call_graph.append((caller_context, full_call, node.lineno))
                    # Register for cross-file analysis - try both full and method name
                    self.call_relationships.append((self.file_path, full_call, caller_context, node.lineno))
                    self.call_relationships.append((self.file_path, method_name, caller_context, node.lineno))
                    
                    # Enhanced detection (potpie-style)
                    if obj_name in self.imports:
                        imported_module = self.imports[obj_name]
                        qualified_call = f"{imported_module}.{method_name}"
                        self.call_relationships.append((self.file_path, qualified_call, caller_context, node.lineno))
            
            # Handle nested attributes like handlers["start"](args)
            elif isinstance(node.func.value, ast.Subscript) and isinstance(node.func.value.value, ast.Name):
                obj_name = node.func.value.value.id
                if isinstance(node.func.value.slice, ast.Constant):
                    key = node.func.value.slice.value
                    if isinstance(key, str):
                        self.call_relationships.append((self.file_path, key, caller_context, node.lineno))
        
        # Special handling for dictionary calls
        if hasattr(node, 'args'):
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    for key, value in zip(arg.keys, arg.values):
                        if (isinstance(key, ast.Constant) and isinstance(key.value, str) and
                            isinstance(value, ast.Name)):
                            function_name = value.id
                            self.call_relationships.append((self.file_path, function_name, caller_context, node.lineno))
        
        self.generic_visit(node)
    
    def _create_function_node(self, node, is_async: bool = False) -> GraphNode:
        """Create a GraphNode for a function with potpie-style enhancements."""
        func_type = "function"
        tags = self._generate_tags(node.name, func_type)
        
        if is_async:
            tags.append("ASYNC")
        
        metadata = {
            "docstring": ast.get_docstring(node),
            "is_async": is_async,
            "parameters": [arg.arg for arg in node.args.args],
            "parser": "hybrid"
        }
        
        func_node = self._create_node(
            name=node.name,
            node_type=func_type,
            start_line=node.lineno,
            end_line=self._get_end_line(node),
            tags=tags,
            metadata=metadata
        )
        
        # Store containment relationship for later processing
        if self.current_class:
            parent_context = "class"
        else:
            parent_context = "module"
        
        self.call_relationships.append((self.file_path, f"CONTAINS:{node.name}", parent_context, node.lineno))
        
        return func_node
    
    def _generate_tags(self, name: str, entity_type: str) -> List[str]:
        """Generate tags based on naming patterns and context."""
        tags = [entity_type.upper()]
        name_lower = name.lower()
        
        # Auto-tag based on patterns
        for tag, patterns in self.tag_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                tags.append(tag)
        
        # Special rules based on entity type
        if entity_type == "function":
            if name.startswith('test_'):
                tags.append("TEST")
            elif name.startswith('_'):
                tags.append("PRIVATE")
            elif name.startswith('__') and name.endswith('__'):
                tags.append("MAGIC")
        
        elif entity_type == "class":
            if name.endswith('Error') or name.endswith('Exception'):
                tags.append("EXCEPTION")
            elif name.endswith('Test'):
                tags.append("TEST")
        
        return tags
    
    def _get_end_line(self, node: ast.AST) -> int:
        """Get the end line of an AST node."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
        
        # Fallback: estimate based on content
        start = node.lineno
        lines_to_check = self.lines[start-1:start+50]  # Check next 50 lines max
        
        for i, line in enumerate(lines_to_check):
            if i > 0 and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                return start + i - 1
        
        return start
    
    def _extract_base_class_name(self, base: ast.AST) -> str:
        """Extract base class name from AST node."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return ast.unparse(base) if hasattr(ast, 'unparse') else str(base)
        else:
            return str(base)
    
    def _register_nodes_globally(self) -> None:
        """Register all nodes in global entity registry with multiple keys for better resolution."""
        # First, build a map of class contexts for methods
        class_contexts = {}
        for node in self.nodes:
            if node.type == 'class':
                class_contexts[node.name] = node
        
        for node in self.nodes:
            # Primary registration with simple name
            self.global_entities[node.name] = node
            
            # Register with file prefix for disambiguation
            self.global_entities[f"{self.file_path}:{node.name}"] = node
            
            # Register with module context if this is a class/function
            if node.type in ['class', 'function']:
                module_name = self.file_path.replace('.py', '').replace('/', '.').replace('\\', '.')
                self.global_entities[f"{module_name}.{node.name}"] = node
                
                # For methods, find their class context from metadata or position
                if node.type == 'function':
                    # Check if this is a method by looking at all classes in this file
                    for class_name, class_node in class_contexts.items():
                        # Simple heuristic: if function is defined after class and has reasonable line distance
                        if (node.start_line > class_node.start_line and 
                            node.start_line < class_node.end_line):
                            # This function is likely in this class
                            self.global_entities[f"{class_name}.{node.name}"] = node
                            self.global_entities[f"{module_name}.{class_name}.{node.name}"] = node
                            break
    
    def _create_call_edges(self) -> None:
        """Create edges for function calls within this file."""
        for caller_context, callee, line_num in self.call_graph:
            # Find caller node
            caller_node = self._find_local_node(caller_context)
            callee_node = self._find_local_node(callee)
            
            if caller_node and callee_node and caller_node.uuid != callee_node.uuid:
                call_edge = self._create_edge(
                    from_uuid=caller_node.uuid,
                    to_uuid=callee_node.uuid,
                    edge_type="calls",
                    metadata={"line": line_num, "parser": "hybrid"}
                )
                self.edges.append(call_edge)
    
    def _find_local_node(self, name: str) -> Optional[GraphNode]:
        """Find a node by name within this file's nodes."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def _default_tag_patterns(self) -> Dict[str, List[str]]:
        """Default tag patterns for entity classification."""
        return {
            'API': ['api', 'endpoint', 'route', 'handler', 'view'],
            'DATABASE': ['db', 'database', 'model', 'table', 'sql', 'query'],
            'UTILITY': ['util', 'helper', 'common', 'shared'],
            'CONFIG': ['config', 'setting', 'env', 'constant'],
            'TEST': ['test', 'mock', 'fixture'],
            'SERVICE': ['service', 'client', 'manager'],
            'EXCEPTION': ['error', 'exception', 'raise'],
            'ASYNC': ['async', 'await', 'coroutine']
        } 