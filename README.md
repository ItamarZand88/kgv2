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