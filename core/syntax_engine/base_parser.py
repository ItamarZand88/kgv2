"""
Base Parser Interface for Syntax Engine

This module defines the abstract base class that all parsers must implement
to ensure consistent interface across different parsing strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional, Tuple, Any, Protocol
from pathlib import Path

from models import GraphNode, GraphEdge


class ParseResult:
    """Container for parsing results."""
    
    def __init__(
        self, 
        nodes: List[GraphNode], 
        edges: List[GraphEdge], 
        imports: Dict[str, str],
        calls: List[Tuple[str, str, str, int]]
    ):
        self.nodes = nodes
        self.edges = edges
        self.imports = imports  # alias -> full_name mapping
        self.calls = calls  # (caller_file, callee_name, context, line)


class BaseParser(ABC):
    """
    Abstract base class for all code parsers.
    
    This interface ensures that all parsing implementations provide
    consistent methods for code analysis and entity extraction.
    """
    
    def __init__(self, file_path: str, repo_id: str, content: str):
        """
        Initialize parser with file information.
        
        Args:
            file_path: Path to the file being analyzed
            repo_id: Repository identifier
            content: File content as string
        """
        self.file_path = file_path
        self.repo_id = repo_id
        self.content = content
        
        # Analysis state
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        self.imports: Dict[str, str] = {}
        self.calls: List[Tuple[str, str, str, int]] = []
    
    @abstractmethod
    def parse(self) -> ParseResult:
        """
        Parse the file content and extract entities and relationships.
        
        Returns:
            ParseResult containing nodes, edges, imports, and calls
        """
        pass
    
    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """
        Check if this parser supports the given language.
        
        Args:
            language: Programming language identifier
            
        Returns:
            True if the parser supports this language
        """
        pass
    
    def get_language_from_file(self) -> Optional[str]:
        """
        Detect programming language from file extension.
        
        Returns:
            Language identifier or None if not detected
        """
        file_path = Path(self.file_path)
        extension = file_path.suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cxx': 'cpp',
            '.cc': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.cs': 'csharp',
            '.kt': 'kotlin',
            '.swift': 'swift'
        }
        
        return language_map.get(extension)
    
    def _create_node(
        self,
        name: str,
        node_type: str,
        start_line: int,
        end_line: int,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphNode:
        """
        Create a GraphNode with standardized format.
        
        Args:
            name: Entity name
            node_type: Type of entity (function, class, etc.)
            start_line: Starting line number
            end_line: Ending line number
            tags: Optional list of tags
            metadata: Optional metadata dictionary
            
        Returns:
            Created GraphNode instance
        """
        import uuid
        from datetime import datetime
        
        return GraphNode(
            uuid=str(uuid.uuid4()),
            repo_id=self.repo_id,
            name=name,
            type=node_type,
            file_path=self.file_path,
            start_line=start_line,
            end_line=end_line,
            tags=tags or [],
            created_by="system",
            created_at=datetime.now(),
            manual=False
        )
    
    def _create_edge(
        self,
        from_uuid: str,
        to_uuid: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphEdge:
        """
        Create a GraphEdge with standardized format.
        
        Args:
            from_uuid: Source node UUID
            to_uuid: Target node UUID
            edge_type: Type of relationship
            metadata: Optional metadata dictionary
            
        Returns:
            Created GraphEdge instance
        """
        import uuid
        from datetime import datetime
        
        return GraphEdge(
            uuid=str(uuid.uuid4()),
            from_uuid=from_uuid,
            to_uuid=to_uuid,
            type=edge_type,
            properties=metadata or {},
            created_by="system",
            created_at=datetime.now(),
            manual=False
        ) 