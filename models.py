from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class GraphNode(BaseModel):
    """A node in the knowledge graph representing a code entity"""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    repo_id: str
    file_path: str  # e.g. "src/foo/bar.py"
    type: str  # function, class, module, variable, file, etc.
    name: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    docstring: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional entity metadata
    created_by: str
    created_at: datetime = Field(default_factory=datetime.now)
    manual: bool = False
    history: List[Dict[str, Any]] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """An edge in the knowledge graph representing a relationship between entities"""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_uuid: str
    to_uuid: str
    type: str  # calls, inherits, imports, uses, etc.
    properties: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata
    created_by: str
    created_at: datetime = Field(default_factory=datetime.now)
    manual: bool = False
    history: List[Dict[str, Any]] = Field(default_factory=list)


class AuditEvent(BaseModel):
    """Audit event for tracking all changes in the system"""
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    user: str
    action: str  # add_node, update_edge, tag_entity, etc.
    entity_type: str  # node, edge, repo
    entity_uuid: str
    details: Dict[str, Any] = Field(default_factory=dict)


# Request/Response Models for API endpoints

class CreateRepoRequest(BaseModel):
    repo_path: str
    repo_id: Optional[str] = None


class CreateRepoResponse(BaseModel):
    repo_id: str
    message: str


class CreateNodeRequest(BaseModel):
    file_path: str
    type: str
    name: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    docstring: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_by: str = "user"


class UpdateNodeRequest(BaseModel):
    file_path: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    docstring: Optional[str] = None
    tags: Optional[List[str]] = None
    manual: Optional[bool] = None


class CreateEdgeRequest(BaseModel):
    from_uuid: str
    to_uuid: str
    type: str
    created_by: str = "user"


class UpdateEdgeRequest(BaseModel):
    type: Optional[str] = None
    manual: Optional[bool] = None


class BulkCodeRequest(BaseModel):
    uuids: List[str]


class ProbableCodeRequest(BaseModel):
    probable_names: List[str]


class EntityContext(BaseModel):
    """Full context for an entity including code, metadata, and relationships"""
    entity: GraphNode
    code_snippet: Optional[str] = None
    neighbors: List[GraphNode] = Field(default_factory=list)
    relationships: List[GraphEdge] = Field(default_factory=list)


class SearchParams(BaseModel):
    """Parameters for entity search"""
    name: Optional[str] = None
    type: Optional[str] = None
    file_path: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 50
    offset: int = 0


class EntityIndexEntry(BaseModel):
    """Flattened entity for index and search"""
    uuid: str
    name: str
    type: str
    file_path: str
    start_line: Optional[int]
    tags: List[str]
    short_doc: Optional[str] = None 