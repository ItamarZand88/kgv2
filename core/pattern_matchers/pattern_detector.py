"""
Pattern Detector - Advanced Code Pattern Recognition

This module provides sophisticated pattern detection capabilities
for identifying architectural patterns, design patterns, and code idioms.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from models import GraphNode, GraphEdge

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns that can be detected."""
    DESIGN_PATTERN = "design_pattern"
    ARCHITECTURAL_PATTERN = "architectural_pattern"
    IDIOM = "idiom"
    ANTI_PATTERN = "anti_pattern"
    CODE_SMELL = "code_smell"
    REFACTORING_OPPORTUNITY = "refactoring_opportunity"


@dataclass
class PatternMatch:
    """Represents a detected pattern."""
    pattern_id: str
    pattern_name: str
    pattern_type: PatternType
    confidence: float
    file_path: str
    start_line: int
    end_line: int
    entities: List[GraphNode]
    relationships: List[GraphEdge]
    evidence: Dict[str, Any]
    description: str
    suggestions: List[str] = None


class PatternDetector:
    """
    Advanced pattern detector for identifying code patterns.
    
    This detector analyzes code structure, relationships, and syntax
    to identify common patterns, anti-patterns, and refactoring opportunities.
    """
    
    def __init__(self):
        self.pattern_signatures = self._initialize_pattern_signatures()
        self.anti_pattern_signatures = self._initialize_anti_pattern_signatures()
        self.idiom_patterns = self._initialize_idiom_patterns()
    
    def detect_patterns(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str, 
        file_path: str
    ) -> List[PatternMatch]:
        """
        Detect patterns in the analyzed code.
        
        Args:
            nodes: Code entities (functions, classes, etc.)
            edges: Relationships between entities
            content: Source code content
            file_path: Path to the source file
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        # Detect design patterns
        design_patterns = self._detect_design_patterns(nodes, edges, content)
        patterns.extend(design_patterns)
        
        # Detect architectural patterns
        arch_patterns = self._detect_architectural_patterns(nodes, edges, content)
        patterns.extend(arch_patterns)
        
        # Detect anti-patterns
        anti_patterns = self._detect_anti_patterns(nodes, edges, content, file_path)
        patterns.extend(anti_patterns)
        
        # Detect code idioms
        idioms = self._detect_idioms(content, file_path)
        patterns.extend(idioms)
        
        # Detect refactoring opportunities
        refactoring_ops = self._detect_refactoring_opportunities(nodes, edges, content)
        patterns.extend(refactoring_ops)
        
        return patterns
    
    def _detect_design_patterns(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str
    ) -> List[PatternMatch]:
        """Detect common design patterns."""
        patterns = []
        
        # Singleton pattern
        singleton = self._detect_singleton_pattern(nodes, content)
        if singleton:
            patterns.append(singleton)
        
        # Factory pattern
        factory = self._detect_factory_pattern(nodes, edges, content)
        if factory:
            patterns.append(factory)
        
        # Observer pattern
        observer = self._detect_observer_pattern(nodes, edges, content)
        if observer:
            patterns.append(observer)
        
        # Decorator pattern
        decorator = self._detect_decorator_pattern(nodes, edges, content)
        if decorator:
            patterns.append(decorator)
        
        # Strategy pattern
        strategy = self._detect_strategy_pattern(nodes, edges, content)
        if strategy:
            patterns.append(strategy)
        
        return patterns
    
    def _detect_architectural_patterns(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str
    ) -> List[PatternMatch]:
        """Detect architectural patterns."""
        patterns = []
        
        # MVC pattern
        mvc = self._detect_mvc_pattern(nodes, edges)
        if mvc:
            patterns.append(mvc)
        
        # Repository pattern
        repository = self._detect_repository_pattern(nodes, edges, content)
        if repository:
            patterns.append(repository)
        
        # Service layer pattern
        service_layer = self._detect_service_layer_pattern(nodes, content)
        if service_layer:
            patterns.append(service_layer)
        
        return patterns
    
    def _detect_anti_patterns(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str, 
        file_path: str
    ) -> List[PatternMatch]:
        """Detect anti-patterns and code smells."""
        patterns = []
        
        # God class
        god_class = self._detect_god_class(nodes, content)
        if god_class:
            patterns.append(god_class)
        
        # Long parameter list
        long_params = self._detect_long_parameter_list(nodes, content)
        patterns.extend(long_params)
        
        # Duplicate code
        duplicates = self._detect_duplicate_code(content, file_path)
        patterns.extend(duplicates)
        
        # Magic numbers
        magic_numbers = self._detect_magic_numbers(content, file_path)
        patterns.extend(magic_numbers)
        
        return patterns
    
    def _detect_idioms(self, content: str, file_path: str) -> List[PatternMatch]:
        """Detect language-specific idioms."""
        patterns = []
        
        # Python idioms
        if file_path.endswith('.py'):
            python_idioms = self._detect_python_idioms(content, file_path)
            patterns.extend(python_idioms)
        
        # JavaScript idioms
        elif file_path.endswith(('.js', '.ts')):
            js_idioms = self._detect_javascript_idioms(content, file_path)
            patterns.extend(js_idioms)
        
        return patterns
    
    def _detect_refactoring_opportunities(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str
    ) -> List[PatternMatch]:
        """Detect refactoring opportunities."""
        patterns = []
        
        # Extract method opportunities
        extract_method = self._detect_extract_method_opportunities(nodes, content)
        patterns.extend(extract_method)
        
        # Extract class opportunities
        extract_class = self._detect_extract_class_opportunities(nodes, edges)
        if extract_class:
            patterns.append(extract_class)
        
        return patterns
    
    # Design Pattern Detection Methods
    
    def _detect_singleton_pattern(self, nodes: List[GraphNode], content: str) -> Optional[PatternMatch]:
        """Detect Singleton pattern implementation."""
        # Look for classes with private constructor and getInstance method
        for node in nodes:
            if node.type == "class":
                class_content = self._extract_class_content(node, content)
                
                # Check for singleton indicators
                has_private_constructor = "_instance" in class_content or "__new__" in class_content
                has_get_instance = "getInstance" in class_content or "get_instance" in class_content
                
                if has_private_constructor and has_get_instance:
                    return PatternMatch(
                        pattern_id="singleton",
                        pattern_name="Singleton Pattern",
                        pattern_type=PatternType.DESIGN_PATTERN,
                        confidence=0.8,
                        file_path=node.file_path,
                        start_line=node.start_line,
                        end_line=node.end_line,
                        entities=[node],
                        relationships=[],
                        evidence={"private_constructor": has_private_constructor, "get_instance": has_get_instance},
                        description="Class implements Singleton pattern with private constructor and getInstance method"
                    )
        
        return None
    
    def _detect_factory_pattern(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str
    ) -> Optional[PatternMatch]:
        """Detect Factory pattern implementation."""
        # Look for classes/functions that create other objects
        for node in nodes:
            if node.type in ["class", "function"]:
                node_content = self._extract_node_content(node, content)
                
                # Check for factory indicators
                creates_objects = bool(re.search(r'return\s+\w+\s*\(', node_content))
                has_factory_name = any(keyword in node.name.lower() 
                                     for keyword in ["factory", "create", "build", "make"])
                
                if creates_objects and has_factory_name:
                    return PatternMatch(
                        pattern_id="factory",
                        pattern_name="Factory Pattern",
                        pattern_type=PatternType.DESIGN_PATTERN,
                        confidence=0.7,
                        file_path=node.file_path,
                        start_line=node.start_line,
                        end_line=node.end_line,
                        entities=[node],
                        relationships=[],
                        evidence={"creates_objects": creates_objects, "factory_name": has_factory_name},
                        description="Factory pattern detected - creates objects based on parameters"
                    )
        
        return None
    
    def _detect_observer_pattern(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str
    ) -> Optional[PatternMatch]:
        """Detect Observer pattern implementation."""
        # Look for subject-observer relationships
        subjects = []
        observers = []
        
        for node in nodes:
            if node.type == "class":
                node_content = self._extract_class_content(node, content)
                
                # Check for subject characteristics
                has_observers_list = "observers" in node_content or "listeners" in node_content
                has_notify_method = "notify" in node_content or "update" in node_content
                
                if has_observers_list and has_notify_method:
                    subjects.append(node)
                
                # Check for observer characteristics
                has_update_method = "update" in node_content or "on_notify" in node_content
                if has_update_method:
                    observers.append(node)
        
        if subjects and observers:
            all_entities = subjects + observers
            return PatternMatch(
                pattern_id="observer",
                pattern_name="Observer Pattern",
                pattern_type=PatternType.DESIGN_PATTERN,
                confidence=0.8,
                file_path=subjects[0].file_path,
                start_line=min(node.start_line for node in all_entities),
                end_line=max(node.end_line for node in all_entities),
                entities=all_entities,
                relationships=[],
                evidence={"subjects": len(subjects), "observers": len(observers)},
                description="Observer pattern detected with subject-observer relationships"
            )
        
        return None
    
    def _detect_decorator_pattern(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str
    ) -> Optional[PatternMatch]:
        """Detect Decorator pattern (not Python decorators)."""
        # Look for decorator pattern structure
        decorators = []
        
        for node in nodes:
            if node.type == "class":
                node_content = self._extract_class_content(node, content)
                
                # Check for decorator characteristics
                has_component_ref = "component" in node_content.lower()
                has_delegate_method = any(keyword in node_content 
                                        for keyword in ["delegate", "forward", "proxy"])
                
                if has_component_ref and has_delegate_method:
                    decorators.append(node)
        
        if decorators:
            return PatternMatch(
                pattern_id="decorator_pattern",
                pattern_name="Decorator Pattern",
                pattern_type=PatternType.DESIGN_PATTERN,
                confidence=0.7,
                file_path=decorators[0].file_path,
                start_line=min(node.start_line for node in decorators),
                end_line=max(node.end_line for node in decorators),
                entities=decorators,
                relationships=[],
                evidence={"decorators_count": len(decorators)},
                description="Decorator pattern detected - objects wrap other objects to add behavior"
            )
        
        return None
    
    def _detect_strategy_pattern(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str
    ) -> Optional[PatternMatch]:
        """Detect Strategy pattern implementation."""
        strategies = []
        contexts = []
        
        for node in nodes:
            if node.type == "class":
                node_content = self._extract_class_content(node, content)
                
                # Check for strategy characteristics
                has_strategy_interface = any(keyword in node.name.lower() 
                                           for keyword in ["strategy", "algorithm", "policy"])
                has_execute_method = any(keyword in node_content 
                                       for keyword in ["execute", "apply", "perform"])
                
                if has_strategy_interface and has_execute_method:
                    strategies.append(node)
                
                # Check for context characteristics
                has_strategy_reference = "strategy" in node_content.lower()
                has_set_strategy = "set_strategy" in node_content or "setStrategy" in node_content
                
                if has_strategy_reference and has_set_strategy:
                    contexts.append(node)
        
        if strategies and contexts:
            all_entities = strategies + contexts
            return PatternMatch(
                pattern_id="strategy",
                pattern_name="Strategy Pattern",
                pattern_type=PatternType.DESIGN_PATTERN,
                confidence=0.8,
                file_path=all_entities[0].file_path,
                start_line=min(node.start_line for node in all_entities),
                end_line=max(node.end_line for node in all_entities),
                entities=all_entities,
                relationships=[],
                evidence={"strategies": len(strategies), "contexts": len(contexts)},
                description="Strategy pattern detected with interchangeable algorithms"
            )
        
        return None
    
    # Architectural Pattern Detection
    
    def _detect_mvc_pattern(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> Optional[PatternMatch]:
        """Detect MVC (Model-View-Controller) pattern."""
        models = []
        views = []
        controllers = []
        
        for node in nodes:
            if node.type == "class":
                name_lower = node.name.lower()
                
                if any(keyword in name_lower for keyword in ["model", "entity", "data"]):
                    models.append(node)
                elif any(keyword in name_lower for keyword in ["view", "template", "ui"]):
                    views.append(node)
                elif any(keyword in name_lower for keyword in ["controller", "handler", "action"]):
                    controllers.append(node)
        
        # Need at least one of each to consider it MVC
        if models and views and controllers:
            all_entities = models + views + controllers
            return PatternMatch(
                pattern_id="mvc",
                pattern_name="MVC Pattern",
                pattern_type=PatternType.ARCHITECTURAL_PATTERN,
                confidence=0.7,
                file_path=all_entities[0].file_path,
                start_line=min(node.start_line for node in all_entities),
                end_line=max(node.end_line for node in all_entities),
                entities=all_entities,
                relationships=[],
                evidence={"models": len(models), "views": len(views), "controllers": len(controllers)},
                description="MVC architectural pattern detected"
            )
        
        return None
    
    def _detect_repository_pattern(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        content: str
    ) -> Optional[PatternMatch]:
        """Detect Repository pattern implementation."""
        repositories = []
        
        for node in nodes:
            if node.type == "class" and "repository" in node.name.lower():
                node_content = self._extract_class_content(node, content)
                
                # Check for repository methods
                has_crud_methods = any(method in node_content.lower() 
                                     for method in ["find", "save", "delete", "create", "update"])
                
                if has_crud_methods:
                    repositories.append(node)
        
        if repositories:
            return PatternMatch(
                pattern_id="repository",
                pattern_name="Repository Pattern",
                pattern_type=PatternType.ARCHITECTURAL_PATTERN,
                confidence=0.8,
                file_path=repositories[0].file_path,
                start_line=min(node.start_line for node in repositories),
                end_line=max(node.end_line for node in repositories),
                entities=repositories,
                relationships=[],
                evidence={"repositories_count": len(repositories)},
                description="Repository pattern detected - encapsulates data access logic"
            )
        
        return None
    
    def _detect_service_layer_pattern(self, nodes: List[GraphNode], content: str) -> Optional[PatternMatch]:
        """Detect Service Layer pattern."""
        services = []
        
        for node in nodes:
            if node.type == "class" and "service" in node.name.lower():
                node_content = self._extract_class_content(node, content)
                
                # Check for service characteristics
                has_business_logic = any(keyword in node_content.lower() 
                                       for keyword in ["process", "handle", "execute", "perform"])
                
                if has_business_logic:
                    services.append(node)
        
        if services:
            return PatternMatch(
                pattern_id="service_layer",
                pattern_name="Service Layer Pattern",
                pattern_type=PatternType.ARCHITECTURAL_PATTERN,
                confidence=0.7,
                file_path=services[0].file_path,
                start_line=min(node.start_line for node in services),
                end_line=max(node.end_line for node in services),
                entities=services,
                relationships=[],
                evidence={"services_count": len(services)},
                description="Service layer pattern detected - encapsulates business logic"
            )
        
        return None
    
    # Anti-Pattern Detection
    
    def _detect_god_class(self, nodes: List[GraphNode], content: str) -> Optional[PatternMatch]:
        """Detect God Class anti-pattern."""
        for node in nodes:
            if node.type == "class":
                node_content = self._extract_class_content(node, content)
                
                # Simple heuristics for god class
                lines_count = len(node_content.split('\n'))
                methods_count = len(re.findall(r'def\s+\w+', node_content))
                
                # Thresholds for god class
                if lines_count > 200 or methods_count > 20:
                    return PatternMatch(
                        pattern_id="god_class",
                        pattern_name="God Class",
                        pattern_type=PatternType.ANTI_PATTERN,
                        confidence=0.8,
                        file_path=node.file_path,
                        start_line=node.start_line,
                        end_line=node.end_line,
                        entities=[node],
                        relationships=[],
                        evidence={"lines_count": lines_count, "methods_count": methods_count},
                        description="God class detected - class is too large and has too many responsibilities",
                        suggestions=["Consider breaking this class into smaller, more focused classes"]
                    )
        
        return None
    
    def _detect_long_parameter_list(self, nodes: List[GraphNode], content: str) -> List[PatternMatch]:
        """Detect Long Parameter List anti-pattern."""
        patterns = []
        
        for node in nodes:
            if node.type == "function":
                # Extract function signature
                func_content = self._extract_node_content(node, content)
                func_def_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', func_content)
                
                if func_def_match:
                    params = func_def_match.group(1)
                    param_count = len([p.strip() for p in params.split(',') if p.strip()])
                    
                    if param_count > 5:  # Threshold for too many parameters
                        patterns.append(PatternMatch(
                            pattern_id="long_parameter_list",
                            pattern_name="Long Parameter List",
                            pattern_type=PatternType.ANTI_PATTERN,
                            confidence=0.7,
                            file_path=node.file_path,
                            start_line=node.start_line,
                            end_line=node.end_line,
                            entities=[node],
                            relationships=[],
                            evidence={"parameter_count": param_count},
                            description=f"Function has {param_count} parameters, which is too many",
                            suggestions=["Consider using a parameter object or breaking the function into smaller parts"]
                        ))
        
        return patterns
    
    def _detect_duplicate_code(self, content: str, file_path: str) -> List[PatternMatch]:
        """Detect duplicate code blocks."""
        patterns = []
        lines = content.split('\n')
        
        # Simple duplicate detection - look for repeated blocks
        for i in range(len(lines) - 5):
            block = '\n'.join(lines[i:i+5])
            if block.strip():
                # Check if this block appears elsewhere
                remaining_content = '\n'.join(lines[i+5:])
                if block in remaining_content:
                    patterns.append(PatternMatch(
                        pattern_id="duplicate_code",
                        pattern_name="Duplicate Code",
                        pattern_type=PatternType.CODE_SMELL,
                        confidence=0.6,
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 5,
                        entities=[],
                        relationships=[],
                        evidence={"duplicate_block": block[:100]},
                        description="Duplicate code block detected",
                        suggestions=["Extract this code into a reusable function or method"]
                    ))
                    break  # Only report first occurrence
        
        return patterns
    
    def _detect_magic_numbers(self, content: str, file_path: str) -> List[PatternMatch]:
        """Detect magic numbers in code."""
        patterns = []
        lines = content.split('\n')
        
        # Look for numeric literals (excluding common ones like 0, 1, -1)
        magic_number_pattern = re.compile(r'\b(?!0\b|1\b|-1\b)\d{2,}\b')
        
        for i, line in enumerate(lines, 1):
            matches = magic_number_pattern.finditer(line)
            for match in matches:
                patterns.append(PatternMatch(
                    pattern_id="magic_number",
                    pattern_name="Magic Number",
                    pattern_type=PatternType.CODE_SMELL,
                    confidence=0.6,
                    file_path=file_path,
                    start_line=i,
                    end_line=i,
                    entities=[],
                    relationships=[],
                    evidence={"magic_number": match.group()},
                    description=f"Magic number '{match.group()}' should be replaced with a named constant",
                    suggestions=["Replace this magic number with a well-named constant"]
                ))
        
        return patterns
    
    # Language-specific Idiom Detection
    
    def _detect_python_idioms(self, content: str, file_path: str) -> List[PatternMatch]:
        """Detect Python-specific idioms and patterns."""
        patterns = []
        
        # Context manager usage
        if "with " in content:
            patterns.append(PatternMatch(
                pattern_id="python_context_manager",
                pattern_name="Context Manager Usage",
                pattern_type=PatternType.IDIOM,
                confidence=0.9,
                file_path=file_path,
                start_line=1,
                end_line=len(content.split('\n')),
                entities=[],
                relationships=[],
                evidence={"uses_with_statement": True},
                description="Good use of Python context managers",
            ))
        
        # List comprehensions
        if re.search(r'\[.+for\s+.+in\s+.+\]', content):
            patterns.append(PatternMatch(
                pattern_id="python_list_comprehension",
                pattern_name="List Comprehension",
                pattern_type=PatternType.IDIOM,
                confidence=0.8,
                file_path=file_path,
                start_line=1,
                end_line=len(content.split('\n')),
                entities=[],
                relationships=[],
                evidence={"uses_list_comprehension": True},
                description="Good use of Pythonic list comprehensions"
            ))
        
        return patterns
    
    def _detect_javascript_idioms(self, content: str, file_path: str) -> List[PatternMatch]:
        """Detect JavaScript-specific idioms and patterns."""
        patterns = []
        
        # Arrow functions
        if "=>" in content:
            patterns.append(PatternMatch(
                pattern_id="js_arrow_functions",
                pattern_name="Arrow Functions",
                pattern_type=PatternType.IDIOM,
                confidence=0.8,
                file_path=file_path,
                start_line=1,
                end_line=len(content.split('\n')),
                entities=[],
                relationships=[],
                evidence={"uses_arrow_functions": True},
                description="Good use of modern JavaScript arrow functions"
            ))
        
        # Destructuring
        if re.search(r'(?:const|let|var)\s*{\s*\w+.*}\s*=', content):
            patterns.append(PatternMatch(
                pattern_id="js_destructuring",
                pattern_name="Destructuring Assignment",
                pattern_type=PatternType.IDIOM,
                confidence=0.8,
                file_path=file_path,
                start_line=1,
                end_line=len(content.split('\n')),
                entities=[],
                relationships=[],
                evidence={"uses_destructuring": True},
                description="Good use of ES6 destructuring"
            ))
        
        return patterns
    
    # Refactoring Opportunity Detection
    
    def _detect_extract_method_opportunities(self, nodes: List[GraphNode], content: str) -> List[PatternMatch]:
        """Detect opportunities to extract methods."""
        patterns = []
        
        for node in nodes:
            if node.type == "function":
                node_content = self._extract_node_content(node, content)
                lines_count = len(node_content.split('\n'))
                
                # Large function that could be broken down
                if lines_count > 30:
                    patterns.append(PatternMatch(
                        pattern_id="extract_method",
                        pattern_name="Extract Method Opportunity",
                        pattern_type=PatternType.REFACTORING_OPPORTUNITY,
                        confidence=0.7,
                        file_path=node.file_path,
                        start_line=node.start_line,
                        end_line=node.end_line,
                        entities=[node],
                        relationships=[],
                        evidence={"lines_count": lines_count},
                        description=f"Large function ({lines_count} lines) could benefit from method extraction",
                        suggestions=["Consider breaking this function into smaller, more focused methods"]
                    ))
        
        return patterns
    
    def _detect_extract_class_opportunities(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> Optional[PatternMatch]:
        """Detect opportunities to extract classes."""
        # Look for functions that could be grouped into a class
        functions = [node for node in nodes if node.type == "function"]
        
        if len(functions) > 10:  # Many functions might benefit from being grouped
            return PatternMatch(
                pattern_id="extract_class",
                pattern_name="Extract Class Opportunity",
                pattern_type=PatternType.REFACTORING_OPPORTUNITY,
                confidence=0.6,
                file_path=functions[0].file_path,
                start_line=min(node.start_line for node in functions),
                end_line=max(node.end_line for node in functions),
                entities=functions,
                relationships=[],
                evidence={"functions_count": len(functions)},
                description=f"File has {len(functions)} functions that might benefit from being organized into classes",
                suggestions=["Consider grouping related functions into classes"]
            )
        
        return None
    
    # Helper Methods
    
    def _extract_class_content(self, class_node: GraphNode, content: str) -> str:
        """Extract the content of a class."""
        lines = content.split('\n')
        start_idx = max(0, class_node.start_line - 1)
        end_idx = min(len(lines), class_node.end_line)
        return '\n'.join(lines[start_idx:end_idx])
    
    def _extract_node_content(self, node: GraphNode, content: str) -> str:
        """Extract the content of any node."""
        lines = content.split('\n')
        start_idx = max(0, node.start_line - 1)
        end_idx = min(len(lines), node.end_line)
        return '\n'.join(lines[start_idx:end_idx])
    
    def _initialize_pattern_signatures(self) -> Dict[str, Any]:
        """Initialize pattern signatures for detection."""
        return {
            "singleton": {
                "indicators": ["_instance", "getInstance", "__new__"],
                "confidence_threshold": 0.7
            },
            "factory": {
                "indicators": ["create", "make", "build", "factory"],
                "confidence_threshold": 0.6
            },
            "observer": {
                "indicators": ["observer", "listener", "notify", "update"],
                "confidence_threshold": 0.7
            }
        }
    
    def _initialize_anti_pattern_signatures(self) -> Dict[str, Any]:
        """Initialize anti-pattern signatures."""
        return {
            "god_class": {
                "max_lines": 200,
                "max_methods": 20
            },
            "long_parameter_list": {
                "max_parameters": 5
            }
        }
    
    def _initialize_idiom_patterns(self) -> Dict[str, Dict]:
        """Initialize language idiom patterns."""
        return {
            "python": {
                "context_manager": r"with\s+\w+",
                "list_comprehension": r"\[.+for\s+.+in\s+.+\]",
                "generator_expression": r"\(.+for\s+.+in\s+.+\)"
            },
            "javascript": {
                "arrow_function": r"=>\s*",
                "destructuring": r"(?:const|let|var)\s*{\s*\w+.*}\s*=",
                "template_literal": r"`[^`]*\$\{[^}]*\}[^`]*`"
            }
        } 