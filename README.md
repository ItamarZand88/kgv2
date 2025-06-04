# Knowledge Graph API

A modular, maintainable, and extensible API for parsing Python repositories into a rich knowledge graph, supporting powerful search, tagging, context extraction, code retrieval, metadata overlays, and complete auditability.

## Features

- **Repository Analysis**: Parse Python repositories using AST to extract functions, classes, imports, and relationships
- **Knowledge Graph**: Build and maintain a NetworkX-based graph of code entities and their relationships
- **Advanced Search**: Fuzzy search, tag-based filtering, and probable name matching
- **Context Extraction**: Get full context including code snippets, neighbors, and relationship trees
- **Audit Logging**: Complete audit trail of all changes and operations
- **Tagging System**: Automatic and manual tagging with extensible tag categories
- **RESTful API**: Fully documented FastAPI endpoints with OpenAPI specifications

## Architecture

- **FastAPI**: Asynchronous web framework with automatic OpenAPI documentation
- **NetworkX**: In-memory graph engine with pickle persistence
- **Pydantic**: Strict schema validation and serialization
- **Local Storage**: File-based storage in `data/{repo_id}/` directories
- **Modular Design**: Clear separation between API, analysis, storage, and search layers

## Installation

1. **Clone this repository**:

```bash
git clone <repository-url>
cd kgv2
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Create data directory** (if not exists):

```bash
mkdir data
```

## Quick Start

1. **Start the API server**:

```bash
python api.py
```

The server will start on `http://localhost:8000`

2. **View API documentation**:
   Open `http://localhost:8000/docs` in your browser to see the interactive OpenAPI documentation.

3. **Analyze a repository**:

```bash
curl -X POST "http://localhost:8000/repo" \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/your/python/project"}'
```

## API Endpoints Overview

### Repository Management

- `POST /repo` - Create and analyze a repository
- `DELETE /repo/{repo_id}` - Delete repository data
- `GET /repos` - List all repositories

### Search & Discovery

- `GET /repo/{repo_id}/entities-index` - Get flat entity index with filtering
- `GET /repo/{repo_id}/search` - Advanced fuzzy search
- `POST /repo/{repo_id}/probable-code` - Find entities by probable names

### Entity Management

- `GET /repo/{repo_id}/entities` - List entities
- `GET /repo/{repo_id}/entity/{uuid}` - Get entity details
- `POST /repo/{repo_id}/entity` - Create entity manually
- `PATCH /repo/{repo_id}/entity/{uuid}` - Update entity
- `DELETE /repo/{repo_id}/entity/{uuid}` - Delete entity

### Relationships

- `GET /repo/{repo_id}/relations` - List relationships
- `POST /repo/{repo_id}/relation` - Create relationship

### Context & Code

- `GET /repo/{repo_id}/entity/{uuid}/context` - Get full entity context
- `POST /repo/{repo_id}/bulk-code` - Get bulk entity code
- `GET /repo/{repo_id}/entity/{uuid}/neighbors` - Get entity neighbors
- `GET /repo/{repo_id}/entity/{uuid}/subgraph` - Get entity subgraph
- `GET /repo/{repo_id}/entity/{uuid}/tree` - Get call/inheritance tree

### Audit

- `GET /repo/{repo_id}/audit` - Get audit log
- `GET /repo/{repo_id}/entity/{uuid}/audit` - Get entity audit history

## Usage Examples

### 1. Analyze a Local Repository

```python
import requests

# Analyze a local Python project
response = requests.post("http://localhost:8000/repo", json={
    "repo_path": "/path/to/python/project",
    "repo_id": "my-project"
})

print(response.json())
# {"repo_id": "my-project", "message": "Repository analyzed successfully..."}
```

### 2. Search for Entities

```python
# Search for functions containing "main"
response = requests.get("http://localhost:8000/repo/my-project/search", params={
    "name": "main",
    "type": "function"
})

entities = response.json()["results"]
for entity in entities:
    print(f"{entity['name']} in {entity['file_path']}")
```

### 3. Get Entity Context

```python
# Get full context for an entity
entity_uuid = "some-uuid-here"
response = requests.get(f"http://localhost:8000/repo/my-project/entity/{entity_uuid}/context")

context = response.json()
print("Code:", context["code_snippet"])
print("Neighbors:", [n["name"] for n in context["neighbors"]])
```

### 4. Find by Probable Names

```python
# Find entities by probable names
response = requests.post("http://localhost:8000/repo/my-project/probable-code", json={
    "probable_names": ["src/main.py:main", "utils.helper", "api.routes.user_handler"]
})

entities = response.json()["entities"]
```

### 5. Get Call Tree

```python
# Get call tree for a function
response = requests.get(f"http://localhost:8000/repo/my-project/entity/{entity_uuid}/tree", params={
    "direction": "out",
    "max_depth": 3
})

tree = response.json()["tree"]
```

## Data Model

### GraphNode (Entity)

```python
{
    "uuid": "550e8400-e29b-41d4-a716-446655440000",
    "repo_id": "my-project",
    "file_path": "src/main.py",
    "type": "function",  # function, class, import, variable, etc.
    "name": "main",
    "start_line": 10,
    "end_line": 25,
    "docstring": "Main entry point...",
    "tags": ["API", "ENTRY_POINT"],
    "created_by": "analyzer",
    "created_at": "2023-01-01T12:00:00Z",
    "manual": false,
    "history": []
}
```

### GraphEdge (Relationship)

```python
{
    "uuid": "660e8400-e29b-41d4-a716-446655440000",
    "from_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "to_uuid": "770e8400-e29b-41d4-a716-446655440000",
    "type": "calls",  # calls, inherits, imports, uses, etc.
    "created_by": "analyzer",
    "created_at": "2023-01-01T12:00:00Z",
    "manual": false,
    "history": []
}
```

## Auto-Tagging System

The analyzer automatically applies tags based on naming patterns and code analysis:

- **API**: Functions/classes with "api", "endpoint", "route", "handler", "view"
- **DATABASE**: Items with "db", "database", "model", "table", "sql", "query"
- **UTILITY**: Items with "util", "helper", "common", "shared"
- **CONFIG**: Items with "config", "setting", "env", "constant"
- **TEST**: Items with "test", "mock", "fixture" or starting with "test\_"
- **SERVICE**: Items with "service", "client", "manager"
- **EXCEPTION**: Classes ending in "Error" or "Exception"
- **ASYNC**: Async functions and coroutines
- **PRIVATE**: Functions starting with "\_"
- **MAGIC**: Python magic methods ("**method**")

## File Structure

```
data/
└── {repo_id}/
    ├── graph.gpickle        # NetworkX graph (binary)
    ├── entities_index.json  # Flat entity index for search
    ├── audit.jsonl         # Append-only audit log
    └── tags.json           # Tag configuration
```

## Development

### Adding New Entity Types

1. Update the analyzer in `analyzer.py` to recognize new AST node types
2. Add appropriate tagging rules in the `tag_patterns` dictionary
3. Update the data models in `models.py` if needed

### Adding New Relationship Types

1. Extend the `FileAnalyzer` class to detect new relationship patterns
2. Update edge creation logic in `analyzer.py`
3. Add filtering support in search endpoints

### Adding New Search Features

1. Extend the `SearchEngine` class in `search.py`
2. Add new endpoint in `api.py`
3. Update the OpenAPI documentation

## Configuration

The system uses sensible defaults but can be configured:

- **Data Directory**: Change `base_data_dir` in `StorageManager`
- **Tag Patterns**: Modify `tag_patterns` in `CodeAnalyzer`
- **Search Thresholds**: Adjust fuzzy matching thresholds in `SearchEngine`
- **API Limits**: Modify query parameter limits in API endpoints

## Error Handling

All endpoints return consistent error responses:

```json
{ "error": "Description of error" }
```

Common HTTP status codes:

- `400`: Bad request (invalid input)
- `404`: Resource not found
- `500`: Internal server error

## Performance Considerations

- **Indexing**: Entity index provides fast search without loading full graph
- **Async Operations**: All file I/O is async for better concurrency
- **Pagination**: All list endpoints support limit/offset pagination
- **Graph Caching**: Graphs are cached in memory after first load
- **Parallel Processing**: Bulk operations use async concurrency

## Contributing

1. Follow the modular architecture
2. Add appropriate tests for new features
3. Update this README for new functionality
4. Ensure all endpoints have proper OpenAPI documentation
5. Maintain backward compatibility in API changes

## License

MIT License - see LICENSE file for details.
