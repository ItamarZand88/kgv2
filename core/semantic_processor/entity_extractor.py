"""
Entity Extractor - Advanced Entity Recognition and Classification

This module provides sophisticated entity extraction capabilities
for identifying and classifying code entities beyond basic AST parsing.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from models import GraphNode

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Extended entity types for sophisticated classification."""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    IMPORT = "import"
    VARIABLE = "variable"
    CONSTANT = "constant"
    DECORATOR = "decorator"
    HANDLER = "handler"
    ENDPOINT = "endpoint"
    MODEL = "model"
    UTILITY = "utility"
    TEST = "test"
    EXCEPTION = "exception"


@dataclass
class EntityPattern:
    """Pattern for identifying specific entity types."""
    name: str
    pattern: str
    entity_type: EntityType
    confidence: float
    tags: List[str]


class EntityExtractor:
    """
    Advanced entity extractor for sophisticated code analysis.
    
    This extractor goes beyond basic AST parsing to identify
    semantic patterns and classify entities based on their context,
    naming patterns, and usage patterns.
    """
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.context_analyzers = self._initialize_context_analyzers()
    
    def extract_entities(
        self, 
        nodes: List[GraphNode], 
        content: str, 
        file_path: str,
        repo_id: str = "test_repo"
    ) -> List[GraphNode]:
        """
        Extract additional entities from code analysis.
        
        Args:
            nodes: Existing nodes from basic parsing
            content: Source code content
            file_path: Path to the source file
            repo_id: Repository identifier
            
        Returns:
            List of additional entities found through semantic analysis
        """
        additional_entities = []
        
        # Pattern-based entity extraction
        pattern_entities = self._extract_pattern_based_entities(content, file_path, repo_id)
        additional_entities.extend(pattern_entities)
        
        # Context-based entity classification enhancement
        enhanced_entities = self._enhance_entity_classification(nodes, content, repo_id)
        additional_entities.extend(enhanced_entities)
        
        # Variable and constant detection
        variable_entities = self._extract_variables_and_constants(content, file_path, repo_id)
        additional_entities.extend(variable_entities)
        
        # Handler and endpoint detection
        handler_entities = self._extract_handlers_and_endpoints(content, file_path, nodes, repo_id)
        additional_entities.extend(handler_entities)
        
        logger.info(f"Extracted {len(additional_entities)} additional entities from {file_path}")
        return additional_entities
    
    def _initialize_patterns(self) -> List[EntityPattern]:
        """Initialize entity recognition patterns."""
        return [
            # API Endpoints
            EntityPattern(
                name="flask_route",
                pattern=r'@app\.route\(["\']([^"\']+)["\']',
                entity_type=EntityType.ENDPOINT,
                confidence=0.9,
                tags=["API", "FLASK", "ENDPOINT"]
            ),
            EntityPattern(
                name="fastapi_endpoint",
                pattern=r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                entity_type=EntityType.ENDPOINT,
                confidence=0.9,
                tags=["API", "FASTAPI", "ENDPOINT"]
            ),
            
            # Decorators
            EntityPattern(
                name="decorator",
                pattern=r'@(\w+)',
                entity_type=EntityType.DECORATOR,
                confidence=0.8,
                tags=["DECORATOR"]
            ),
            
            # Constants
            EntityPattern(
                name="constants",
                pattern=r'^([A-Z][A-Z0-9_]*)\s*=',
                entity_type=EntityType.CONSTANT,
                confidence=0.7,
                tags=["CONSTANT"]
            ),
            
            # Exception classes
            EntityPattern(
                name="exception_class",
                pattern=r'class\s+(\w*(?:Error|Exception))\s*\(',
                entity_type=EntityType.EXCEPTION,
                confidence=0.9,
                tags=["EXCEPTION", "CLASS"]
            ),
            
            # Test functions
            EntityPattern(
                name="test_function",
                pattern=r'def\s+(test_\w+)\s*\(',
                entity_type=EntityType.TEST,
                confidence=0.9,
                tags=["TEST", "FUNCTION"]
            ),
            
            # Model classes (common patterns)
            EntityPattern(
                name="model_class",
                pattern=r'class\s+(\w+)(?:Model|Schema|Entity)\s*\(',
                entity_type=EntityType.MODEL,
                confidence=0.8,
                tags=["MODEL", "CLASS"]
            ),
            
            # Utility functions
            EntityPattern(
                name="utility_function",
                pattern=r'def\s+(_?\w*(?:util|helper|tool)\w*)\s*\(',
                entity_type=EntityType.UTILITY,
                confidence=0.7,
                tags=["UTILITY", "FUNCTION"]
            )
        ]
    
    def _initialize_context_analyzers(self) -> Dict[str, callable]:
        """Initialize context-based analyzers."""
        return {
            "api_context": self._analyze_api_context,
            "test_context": self._analyze_test_context,
            "model_context": self._analyze_model_context,
            "util_context": self._analyze_util_context
        }
    
    def _extract_pattern_based_entities(self, content: str, file_path: str, repo_id: str) -> List[GraphNode]:
        """Extract entities based on regex patterns."""
        entities = []
        lines = content.split('\n')
        
        for pattern in self.patterns:
            regex = re.compile(pattern.pattern, re.MULTILINE)
            for match in regex.finditer(content):
                # Find line number
                line_number = content[:match.start()].count('\n') + 1
                
                # Extract entity name (first capture group)
                entity_name = match.group(1) if match.groups() else match.group(0)
                
                # Create entity node
                entity = self._create_semantic_entity(
                    name=entity_name,
                    entity_type=pattern.entity_type,
                    file_path=file_path,
                    line_number=line_number,
                    confidence=pattern.confidence,
                    tags=pattern.tags.copy(),
                    pattern_name=pattern.name,
                    repo_id=repo_id
                )
                entities.append(entity)
        
        return entities
    
    def _enhance_entity_classification(self, nodes: List[GraphNode], content: str, repo_id: str) -> List[GraphNode]:
        """Enhance classification of existing entities."""
        enhanced = []
        
        for node in nodes:
            # Analyze context around the entity
            context = self._extract_entity_context(node, content)
            
            # Apply context analyzers
            for analyzer_name, analyzer_func in self.context_analyzers.items():
                enhancement = analyzer_func(node, context, repo_id)
                if enhancement:
                    enhanced.append(enhancement)
        
        return enhanced
    
    def _extract_variables_and_constants(self, content: str, file_path: str, repo_id: str) -> List[GraphNode]:
        """Extract significant variables and constants."""
        entities = []
        lines = content.split('\n')
        
        # Global variable assignments
        global_var_pattern = re.compile(r'^(\w+)\s*=\s*(.+)$', re.MULTILINE)
        
        for match in global_var_pattern.finditer(content):
            var_name = match.group(1)
            var_value = match.group(2).strip()
            line_number = content[:match.start()].count('\n') + 1
            
            # Skip imports and function definitions
            if 'import' in var_value or 'def ' in var_value or 'class ' in var_value:
                continue
            
            # Determine if it's a constant or variable
            if var_name.isupper():
                entity_type = EntityType.CONSTANT
                tags = ["CONSTANT", "GLOBAL"]
            else:
                entity_type = EntityType.VARIABLE
                tags = ["VARIABLE", "GLOBAL"]
            
            entity = self._create_semantic_entity(
                name=var_name,
                entity_type=entity_type,
                file_path=file_path,
                line_number=line_number,
                confidence=0.6,
                tags=tags,
                metadata={"value": var_value[:50]},  # Store first 50 chars of value
                repo_id=repo_id
            )
            entities.append(entity)
        
        return entities
    
    def _extract_handlers_and_endpoints(
        self, 
        content: str, 
        file_path: str, 
        existing_nodes: List[GraphNode],
        repo_id: str
    ) -> List[GraphNode]:
        """Extract handler functions and API endpoints."""
        entities = []
        
        # Find handler dictionaries (common in web frameworks)
        handler_pattern = re.compile(r'(\w+)\s*=\s*\{[^}]*["\'](\w+)["\']\s*:\s*(\w+)', re.MULTILINE | re.DOTALL)
        
        for match in handler_pattern.finditer(content):
            handler_dict = match.group(1)
            handler_key = match.group(2)
            handler_func = match.group(3)
            line_number = content[:match.start()].count('\n') + 1
            
            # Create handler entity
            handler_entity = self._create_semantic_entity(
                name=f"{handler_dict}[{handler_key}]",
                entity_type=EntityType.HANDLER,
                file_path=file_path,
                line_number=line_number,
                confidence=0.8,
                tags=["HANDLER", "MAPPING"],
                metadata={
                    "handler_dict": handler_dict,
                    "handler_key": handler_key,
                    "handler_function": handler_func
                },
                repo_id=repo_id
            )
            entities.append(handler_entity)
        
        return entities
    
    def _create_semantic_entity(
        self,
        name: str,
        entity_type: EntityType,
        file_path: str,
        line_number: int,
        confidence: float,
        tags: List[str],
        pattern_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        repo_id: str = "test_repo"
    ) -> GraphNode:
        """Create a semantic entity node."""
        import uuid
        from datetime import datetime
        
        base_metadata = {
            "confidence": confidence,
            "extraction_method": "semantic_pattern",
            "extractor": "EntityExtractor"
        }
        
        if pattern_name:
            base_metadata["pattern"] = pattern_name
        
        if metadata:
            base_metadata.update(metadata)
        
        return GraphNode(
            uuid=str(uuid.uuid4()),
            repo_id=repo_id,
            name=name,
            type=entity_type.value,
            file_path=file_path,
            start_line=line_number,
            end_line=line_number,
            tags=tags,
            metadata=base_metadata,
            created_by="EntityExtractor",
            created_at=datetime.now()
        )
    
    def _extract_entity_context(self, node: GraphNode, content: str) -> Dict[str, Any]:
        """Extract context around an entity."""
        lines = content.split('\n')
        start_line = max(0, node.start_line - 3)
        end_line = min(len(lines), node.end_line + 3)
        
        context_lines = lines[start_line:end_line]
        
        return {
            "surrounding_lines": context_lines,
            "file_content": content,
            "line_range": (start_line, end_line)
        }
    
    def _analyze_api_context(self, node: GraphNode, context: Dict[str, Any], repo_id: str) -> Optional[GraphNode]:
        """Analyze if entity is in API context."""
        context_text = '\n'.join(context["surrounding_lines"])
        
        api_indicators = [
            '@app.route', '@app.get', '@app.post', 
            'request', 'response', 'HTTP', 'API'
        ]
        
        if any(indicator in context_text for indicator in api_indicators):
            if node.type == "function":
                # Create enhanced API function entity
                return self._create_semantic_entity(
                    name=f"api_{node.name}",
                    entity_type=EntityType.ENDPOINT,
                    file_path=node.file_path,
                    line_number=node.start_line,
                    confidence=0.8,
                    tags=["API", "ENDPOINT", "ENHANCED"],
                    metadata={"base_entity": node.uuid},
                    repo_id=repo_id
                )
        
        return None
    
    def _analyze_test_context(self, node: GraphNode, context: Dict[str, Any], repo_id: str) -> Optional[GraphNode]:
        """Analyze if entity is in test context."""
        context_text = '\n'.join(context["surrounding_lines"])
        
        test_indicators = [
            'assert', 'pytest', 'unittest', 'mock', 'test_', '@pytest'
        ]
        
        if any(indicator in context_text for indicator in test_indicators):
            if node.type == "function" and not node.name.startswith('test_'):
                # Create enhanced test utility entity
                return self._create_semantic_entity(
                    name=f"test_util_{node.name}",
                    entity_type=EntityType.TEST,
                    file_path=node.file_path,
                    line_number=node.start_line,
                    confidence=0.7,
                    tags=["TEST", "UTILITY", "ENHANCED"],
                    metadata={"base_entity": node.uuid},
                    repo_id=repo_id
                )
        
        return None
    
    def _analyze_model_context(self, node: GraphNode, context: Dict[str, Any], repo_id: str) -> Optional[GraphNode]:
        """Analyze if entity is in model/data context."""
        context_text = '\n'.join(context["surrounding_lines"])
        
        model_indicators = [
            'pydantic', 'BaseModel', 'dataclass', 'SQLAlchemy', 
            'Table', 'Column', 'relationship'
        ]
        
        if any(indicator in context_text for indicator in model_indicators):
            if node.type == "class":
                # Create enhanced model entity
                return self._create_semantic_entity(
                    name=f"model_{node.name}",
                    entity_type=EntityType.MODEL,
                    file_path=node.file_path,
                    line_number=node.start_line,
                    confidence=0.8,
                    tags=["MODEL", "DATA", "ENHANCED"],
                    metadata={"base_entity": node.uuid},
                    repo_id=repo_id
                )
        
        return None
    
    def _analyze_util_context(self, node: GraphNode, context: Dict[str, Any], repo_id: str) -> Optional[GraphNode]:
        """Analyze if entity is utility function."""
        if node.type != "function":
            return None
        
        util_indicators = [
            'helper', 'util', 'tool', 'common', 'shared'
        ]
        
        # Check function name and file path
        name_lower = node.name.lower()
        file_lower = node.file_path.lower()
        
        if (any(indicator in name_lower for indicator in util_indicators) or
            any(indicator in file_lower for indicator in util_indicators)):
            
            return self._create_semantic_entity(
                name=f"util_{node.name}",
                entity_type=EntityType.UTILITY,
                file_path=node.file_path,
                line_number=node.start_line,
                confidence=0.7,
                tags=["UTILITY", "HELPER", "ENHANCED"],
                metadata={"base_entity": node.uuid},
                repo_id=repo_id
            )
        
        return None
    
    def extract_entities_from_content(
        self, 
        content: str, 
        file_path: str,
        repo_id: str = "test_repo"
    ) -> List[GraphNode]:
        """
        Extract entities directly from content without existing nodes.
        
        Args:
            content: Source code content
            file_path: Path to the source file
            repo_id: Repository identifier
            
        Returns:
            List of entities found through semantic analysis
        """
        # Start with empty node list
        return self.extract_entities([], content, file_path, repo_id) 