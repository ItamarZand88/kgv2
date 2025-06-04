"""
Relationship Builder - Advanced Relationship Detection and Construction

This module provides sophisticated relationship building capabilities
for creating complex relationships between code entities.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

from models import GraphNode, GraphEdge

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Extended relationship types for sophisticated analysis."""
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    CONTAINS = "contains"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    DECORATES = "decorates"
    HANDLES = "handles"
    CONFIGURES = "configures"
    TESTS = "tests"
    MOCKS = "mocks"
    OVERRIDES = "overrides"
    AGGREGATES = "aggregates"
    COMPOSES = "composes"


@dataclass
class RelationshipPattern:
    """Pattern for detecting specific relationship types."""
    name: str
    source_pattern: str
    target_pattern: str
    relationship_type: RelationshipType
    confidence: float
    bidirectional: bool = False


class RelationshipBuilder:
    """
    Advanced relationship builder for sophisticated code analysis.
    
    This builder creates complex relationships between entities
    based on code patterns, semantic analysis, and contextual understanding.
    """
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.relationship_analyzers = self._initialize_analyzers()
    
    def build_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str, 
        file_path: str,
        cross_file_data: Optional[Dict[str, Any]] = None
    ) -> List[GraphEdge]:
        """
        Build sophisticated relationships between entities.
        
        Args:
            nodes: List of entities to analyze
            content: Source code content
            file_path: Path to the source file
            cross_file_data: Additional data for cross-file analysis
            
        Returns:
            List of relationship edges
        """
        relationships = []
        
        # Pattern-based relationship detection
        pattern_relationships = self._detect_pattern_relationships(nodes, content, file_path)
        relationships.extend(pattern_relationships)
        
        # Semantic relationship analysis
        semantic_relationships = self._analyze_semantic_relationships(nodes, content)
        relationships.extend(semantic_relationships)
        
        # Decorator relationships
        decorator_relationships = self._detect_decorator_relationships(nodes, content)
        relationships.extend(decorator_relationships)
        
        # Handler relationships
        handler_relationships = self._detect_handler_relationships(nodes, content)
        relationships.extend(handler_relationships)
        
        # Test relationships
        test_relationships = self._detect_test_relationships(nodes, content)
        relationships.extend(test_relationships)
        
        # Configuration relationships
        config_relationships = self._detect_configuration_relationships(nodes, content)
        relationships.extend(config_relationships)
        
        # Cross-file relationships
        if cross_file_data:
            cross_relationships = self._build_cross_file_relationships(nodes, cross_file_data)
            relationships.extend(cross_relationships)
        
        logger.info(f"Built {len(relationships)} relationships for {file_path}")
        return relationships
    
    def _initialize_patterns(self) -> List[RelationshipPattern]:
        """Initialize relationship detection patterns."""
        return [
            # Inheritance patterns
            RelationshipPattern(
                name="class_inheritance",
                source_pattern=r"class\s+(\w+)\s*\(\s*(\w+)",
                target_pattern=r"\2",
                relationship_type=RelationshipType.INHERITS,
                confidence=0.9
            ),
            
            # Decorator patterns
            RelationshipPattern(
                name="decorator_usage",
                source_pattern=r"@(\w+)\s*def\s+(\w+)",
                target_pattern=r"\1",
                relationship_type=RelationshipType.DECORATES,
                confidence=0.8
            ),
            
            # Import patterns
            RelationshipPattern(
                name="direct_import",
                source_pattern=r"from\s+(\w+)\s+import\s+(\w+)",
                target_pattern=r"\1",
                relationship_type=RelationshipType.IMPORTS,
                confidence=0.9
            ),
            
            # Composition patterns
            RelationshipPattern(
                name="class_composition",
                source_pattern=r"self\.(\w+)\s*=\s*(\w+)\(",
                target_pattern=r"\2",
                relationship_type=RelationshipType.COMPOSES,
                confidence=0.7
            )
        ]
    
    def _initialize_analyzers(self) -> Dict[str, callable]:
        """Initialize relationship analyzers."""
        return {
            "api_handler": self._analyze_api_handler_relationships,
            "test_subject": self._analyze_test_subject_relationships,
            "config_usage": self._analyze_config_usage_relationships,
            "dependency_injection": self._analyze_dependency_injection
        }
    
    def _detect_pattern_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str, 
        file_path: str
    ) -> List[GraphEdge]:
        """Detect relationships using regex patterns."""
        relationships = []
        
        # Create node lookup for fast access
        node_lookup = {node.name: node for node in nodes}
        
        for pattern in self.patterns:
            import re
            regex = re.compile(pattern.source_pattern, re.MULTILINE)
            
            for match in regex.finditer(content):
                source_name = match.group(2) if match.groups() and len(match.groups()) >= 2 else match.group(1)
                target_name = match.group(1) if pattern.target_pattern == r"\1" else match.group(2)
                
                # Find corresponding nodes
                source_node = node_lookup.get(source_name)
                target_node = node_lookup.get(target_name)
                
                if source_node and target_node and source_node != target_node:
                    relationship = self._create_relationship(
                        source_node=source_node,
                        target_node=target_node,
                        relationship_type=pattern.relationship_type,
                        confidence=pattern.confidence,
                        metadata={
                            "pattern": pattern.name,
                            "line": content[:match.start()].count('\n') + 1
                        }
                    )
                    relationships.append(relationship)
                    
                    # Add bidirectional relationship if specified
                    if pattern.bidirectional:
                        reverse_relationship = self._create_relationship(
                            source_node=target_node,
                            target_node=source_node,
                            relationship_type=pattern.relationship_type,
                            confidence=pattern.confidence,
                            metadata={
                                "pattern": f"{pattern.name}_reverse",
                                "line": content[:match.start()].count('\n') + 1
                            }
                        )
                        relationships.append(reverse_relationship)
        
        return relationships
    
    def _analyze_semantic_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str
    ) -> List[GraphEdge]:
        """Analyze semantic relationships between entities."""
        relationships = []
        
        # Apply each semantic analyzer
        for analyzer_name, analyzer_func in self.relationship_analyzers.items():
            analyzer_relationships = analyzer_func(nodes, content)
            relationships.extend(analyzer_relationships)
        
        return relationships
    
    def _detect_decorator_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str
    ) -> List[GraphEdge]:
        """Detect decorator-function relationships."""
        relationships = []
        lines = content.split('\n')
        
        # Find decorator patterns
        import re
        decorator_pattern = re.compile(r'@(\w+)')
        
        for i, line in enumerate(lines):
            decorator_match = decorator_pattern.search(line)
            if decorator_match:
                decorator_name = decorator_match.group(1)
                
                # Look for function definition in next few lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip().startswith('def ') or lines[j].strip().startswith('async def '):
                        func_match = re.search(r'def\s+(\w+)', lines[j])
                        if func_match:
                            func_name = func_match.group(1)
                            
                            # Find corresponding nodes
                            decorator_node = self._find_node_by_name(nodes, decorator_name)
                            function_node = self._find_node_by_name(nodes, func_name)
                            
                            if function_node:  # Decorator might be external
                                relationship = self._create_relationship(
                                    source_node=function_node,
                                    target_node=decorator_node if decorator_node else self._create_external_node(decorator_name, i + 1),
                                    relationship_type=RelationshipType.DECORATES,
                                    confidence=0.9,
                                    metadata={
                                        "decorator": decorator_name,
                                        "line": i + 1
                                    }
                                )
                                relationships.append(relationship)
                        break
        
        return relationships
    
    def _detect_handler_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str
    ) -> List[GraphEdge]:
        """Detect handler-function relationships."""
        relationships = []
        
        # Find handler dictionary patterns
        import re
        handler_pattern = re.compile(r'(\w+)\s*=\s*\{[^}]*["\'](\w+)["\']\s*:\s*(\w+)', re.MULTILINE | re.DOTALL)
        
        for match in handler_pattern.finditer(content):
            handler_dict = match.group(1)
            handler_key = match.group(2)
            handler_func = match.group(3)
            
            # Find corresponding nodes
            dict_node = self._find_node_by_name(nodes, handler_dict)
            func_node = self._find_node_by_name(nodes, handler_func)
            
            if dict_node and func_node:
                relationship = self._create_relationship(
                    source_node=dict_node,
                    target_node=func_node,
                    relationship_type=RelationshipType.HANDLES,
                    confidence=0.8,
                    metadata={
                        "handler_key": handler_key,
                        "line": content[:match.start()].count('\n') + 1
                    }
                )
                relationships.append(relationship)
        
        return relationships
    
    def _detect_test_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str
    ) -> List[GraphEdge]:
        """Detect test-subject relationships."""
        relationships = []
        
        # Find test functions and their subjects
        test_functions = [node for node in nodes if node.name.startswith('test_') or 'TEST' in node.tags]
        
        for test_func in test_functions:
            # Extract potential subject from test name
            subject_name = self._extract_test_subject(test_func.name)
            if subject_name:
                subject_node = self._find_node_by_name(nodes, subject_name)
                if subject_node:
                    relationship = self._create_relationship(
                        source_node=test_func,
                        target_node=subject_node,
                        relationship_type=RelationshipType.TESTS,
                        confidence=0.7,
                        metadata={"test_pattern": "name_based"}
                    )
                    relationships.append(relationship)
            
            # Analyze test content for imported subjects
            test_context = self._extract_function_context(test_func, content)
            if test_context:
                imported_subjects = self._find_imported_test_subjects(test_context, nodes)
                for subject in imported_subjects:
                    relationship = self._create_relationship(
                        source_node=test_func,
                        target_node=subject,
                        relationship_type=RelationshipType.TESTS,
                        confidence=0.6,
                        metadata={"test_pattern": "import_based"}
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _detect_configuration_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str
    ) -> List[GraphEdge]:
        """Detect configuration-usage relationships."""
        relationships = []
        
        # Find configuration constants and their usage
        config_nodes = [node for node in nodes if 'CONFIG' in node.tags or node.name.isupper()]
        
        for config_node in config_nodes:
            # Find usage of this configuration
            import re
            usage_pattern = re.compile(rf'\b{re.escape(config_node.name)}\b')
            
            for match in usage_pattern.finditer(content):
                line_number = content[:match.start()].count('\n') + 1
                
                # Find which function/class uses this config
                using_entity = self._find_entity_at_line(nodes, line_number)
                if using_entity and using_entity != config_node:
                    relationship = self._create_relationship(
                        source_node=using_entity,
                        target_node=config_node,
                        relationship_type=RelationshipType.CONFIGURES,
                        confidence=0.6,
                        metadata={"line": line_number}
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _build_cross_file_relationships(
        self, 
        nodes: List[GraphNode], 
        cross_file_data: Dict[str, Any]
    ) -> List[GraphEdge]:
        """Build relationships across file boundaries."""
        relationships = []
        
        # This would use the global entities registry and import information
        # Implementation depends on the cross_file_data structure
        
        return relationships
    
    def _analyze_api_handler_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str
    ) -> List[GraphEdge]:
        """Analyze API handler relationships."""
        relationships = []
        
        # Find API endpoints and their handlers
        api_nodes = [node for node in nodes if 'API' in node.tags or 'ENDPOINT' in node.tags]
        
        for api_node in api_nodes:
            # Find the function that handles this endpoint
            handler_func = self._find_api_handler_function(api_node, nodes, content)
            if handler_func:
                relationship = self._create_relationship(
                    source_node=api_node,
                    target_node=handler_func,
                    relationship_type=RelationshipType.HANDLES,
                    confidence=0.8,
                    metadata={"relationship_type": "api_handler"}
                )
                relationships.append(relationship)
        
        return relationships
    
    def _analyze_test_subject_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str
    ) -> List[GraphEdge]:
        """Analyze test-subject relationships."""
        relationships = []
        
        # Implementation for sophisticated test-subject analysis
        # This could analyze mock usage, assertion patterns, etc.
        
        return relationships
    
    def _analyze_config_usage_relationships(
        self, 
        nodes: List[GraphNode], 
        content: str
    ) -> List[GraphEdge]:
        """Analyze configuration usage relationships."""
        relationships = []
        
        # Implementation for configuration analysis
        # This could track environment variables, settings, etc.
        
        return relationships
    
    def _analyze_dependency_injection(
        self, 
        nodes: List[GraphNode], 
        content: str
    ) -> List[GraphEdge]:
        """Analyze dependency injection patterns."""
        relationships = []
        
        # Implementation for DI pattern analysis
        # This could detect constructor injection, service locator patterns, etc.
        
        return relationships
    
    def _create_relationship(
        self,
        source_node: GraphNode,
        target_node: GraphNode,
        relationship_type: RelationshipType,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphEdge:
        """Create a relationship edge."""
        import uuid
        from datetime import datetime
        
        base_metadata = {
            "confidence": confidence,
            "builder": "RelationshipBuilder",
            "relationship_type": relationship_type.value
        }
        
        if metadata:
            base_metadata.update(metadata)
        
        return GraphEdge(
            uuid=str(uuid.uuid4()),
            from_uuid=source_node.uuid,
            to_uuid=target_node.uuid,
            type=relationship_type.value,
            properties=base_metadata,
            created_by="RelationshipBuilder",
            created_at=datetime.now()
        )
    
    def _find_node_by_name(self, nodes: List[GraphNode], name: str) -> Optional[GraphNode]:
        """Find a node by name."""
        for node in nodes:
            if node.name == name:
                return node
        return None
    
    def _create_external_node(self, name: str, line: int, repo_id: str = "external") -> GraphNode:
        """Create a placeholder node for external entities."""
        import uuid
        from datetime import datetime
        
        return GraphNode(
            uuid=str(uuid.uuid4()),
            repo_id=repo_id,
            name=name,
            type="external",
            file_path="external",
            start_line=line,
            end_line=line,
            tags=["EXTERNAL"],
            metadata={"source": "relationship_builder"},
            created_by="RelationshipBuilder",
            created_at=datetime.now()
        )
    
    def _extract_test_subject(self, test_name: str) -> Optional[str]:
        """Extract subject name from test function name."""
        # Remove 'test_' prefix and extract likely subject
        if test_name.startswith('test_'):
            subject_part = test_name[5:]  # Remove 'test_'
            # Take first part before underscore as likely subject
            parts = subject_part.split('_')
            if parts:
                return parts[0]
        return None
    
    def _extract_function_context(self, node: GraphNode, content: str) -> Optional[str]:
        """Extract the content of a function."""
        lines = content.split('\n')
        if node.start_line <= len(lines):
            start_idx = node.start_line - 1
            end_idx = min(node.end_line, len(lines))
            return '\n'.join(lines[start_idx:end_idx])
        return None
    
    def _find_imported_test_subjects(self, test_content: str, nodes: List[GraphNode]) -> List[GraphNode]:
        """Find subjects imported/used in test content."""
        subjects = []
        
        # Look for function calls that might be test subjects
        import re
        call_pattern = re.compile(r'(\w+)\s*\(')
        
        for match in call_pattern.finditer(test_content):
            func_name = match.group(1)
            # Skip common test utilities
            if func_name not in ['assert', 'assertEqual', 'assertTrue', 'mock', 'patch']:
                subject_node = self._find_node_by_name(nodes, func_name)
                if subject_node and subject_node not in subjects:
                    subjects.append(subject_node)
        
        return subjects
    
    def _find_entity_at_line(self, nodes: List[GraphNode], line_number: int) -> Optional[GraphNode]:
        """Find the entity that contains a specific line."""
        candidates = []
        
        for node in nodes:
            if node.start_line <= line_number <= node.end_line:
                candidates.append(node)
        
        # Return the most specific (smallest range) entity
        if candidates:
            return min(candidates, key=lambda n: n.end_line - n.start_line)
        
        return None
    
    def _find_api_handler_function(
        self, 
        api_node: GraphNode, 
        nodes: List[GraphNode], 
        content: str
    ) -> Optional[GraphNode]:
        """Find the function that handles an API endpoint."""
        # Look for function definition near the API decorator
        lines = content.split('\n')
        
        # Search around the API node's line
        start_search = max(0, api_node.start_line - 2)
        end_search = min(len(lines), api_node.start_line + 5)
        
        import re
        for i in range(start_search, end_search):
            if i < len(lines):
                func_match = re.search(r'def\s+(\w+)', lines[i])
                if func_match:
                    func_name = func_match.group(1)
                    handler_func = self._find_node_by_name(nodes, func_name)
                    if handler_func:
                        return handler_func
        
        return None 