# Knowledge Graph MCP Server

This project provides a unified Model Context Protocol (MCP) server for advanced code analysis and knowledge graph functionality. The codebase has been converted to use the MCP Python SDK, allowing AI tools and LLM clients to access powerful code analysis capabilities through a single, comprehensive server.

## 🚀 Overview

The Knowledge Graph MCP implementation provides a **unified server** with all functionality consolidated into a single `server.py` file:

- **Repository Management** - Core functionality for repository analysis and entity CRUD operations
- **Advanced Analytics** - Complexity analysis, hotspots, dependencies, and code quality metrics
- **Search & Context** - Search, context retrieval, audit functionality, and graph operations

All tools and resources are available through one server instead of multiple separate servers, making deployment and management much simpler.

## 📋 Prerequisites

- Python 3.8+
- MCP Python SDK
- Required dependencies (see `requirements.txt` or `pyproject.toml`)

## 🛠️ Installation

### Option 1: Automatic Installation

Run the installation script:

```bash
python install_mcp.py
```

This will:

- Check dependencies
- Install the unified MCP server
- Register it with your local MCP registry

### Option 2: Manual Installation

1. Install the MCP CLI:

```bash
pip install mcp
```

2. Install the unified server:

```bash
mcp install server.py --name knowledge-graph-unified
```

## 🔧 Available Tools and Resources

### Tools (Actions/Mutations):

- `create_repository(repo_path, repo_id)` - Analyze and create a repository
- `delete_repository(repo_id)` - Delete repository data
- `create_entity(...)` - Create a new entity manually
- `update_entity(repo_id, entity_uuid, **update_data)` - Update entity fields
- `delete_entity(repo_id, entity_uuid)` - Delete an entity
- `create_relation(...)` - Create relationships between entities
- `get_bulk_code(repo_id, uuids)` - Get code for multiple entities
- `get_probable_code(repo_id, probable_names)` - Find entities by probable names

### Resources (Data Retrieval):

#### Repository Management:

- `repos://list` - List all repositories
- `repo://{repo_id}/entities` - Get entities index with filtering
- `repo://{repo_id}/entity/{entity_uuid}` - Get specific entity details
- `repo://{repo_id}/entity/{entity_uuid}/context` - Get full entity context
- `repo://{repo_id}/statistics` - Repository statistics and metrics
- `repo://{repo_id}/entity/{entity_uuid}/neighbors` - Get neighbor entities

#### Analytics:

- `repo://{repo_id}/complexity-analysis` - Analyze code complexity patterns
- `repo://{repo_id}/hotspots` - Find most important/central code elements
- `repo://{repo_id}/dependencies` - Analyze dependency relationships
- `repo://{repo_id}/impact-analysis/{node_id}` - Analyze impact of changes
- `repo://{repo_id}/language-analysis` - Programming language analysis
- `repo://{repo_id}/code-quality-metrics` - Overall code quality assessment

#### Search & Graph Operations:

- `repo://{repo_id}/search` - Advanced entity search with filtering
- `repo://{repo_id}/relations` - List relationships with filtering
- `repo://{repo_id}/entity/{entity_uuid}/subgraph` - Get entity subgraph
- `repo://{repo_id}/entity/{entity_uuid}/tree` - Get call/inheritance trees
- `repo://{repo_id}/audit` - Repository audit log
- `repo://{repo_id}/entity/{entity_uuid}/audit` - Entity-specific audit events

## 🎯 Usage Examples

### With MCP Inspector

```bash
# Inspect available tools and resources
mcp inspect knowledge-graph-unified

# Test a specific tool
mcp call knowledge-graph-unified create_repository '{"repo_path": "/path/to/repo", "repo_id": "my-repo"}'

# Access resources
mcp get knowledge-graph-unified "repos://list"
mcp get knowledge-graph-unified "repo://my-repo/complexity-analysis"
mcp get knowledge-graph-unified "repo://my-repo/hotspots"
```

### With Claude Desktop

Add to your Claude Desktop MCP configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "KnowledgeGraphMCP": {
      "command": "C:\\Users\\ItamarZand\\Desktop\\kgv2\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\ItamarZand\\Desktop\\kgv2\\server.py"],
      "env": {}
    }
  }
}
```

**Important notes:**

- Use the full path to your virtual environment's Python executable
- Use the full path to your server.py file
- On Windows, use double backslashes (`\\`) in JSON paths
- The server name in the config should match the FastMCP name: `"KnowledgeGraphMCP"`

### Programmatic Usage

```python
from mcp.client import Client

# Connect to the unified server
async with Client("knowledge-graph-unified") as client:
    # Create a repository
    result = await client.call_tool("create_repository", {
        "repo_path": "/path/to/repo",
        "repo_id": "my-repo"
    })

    # Get repository statistics
    stats = await client.get_resource("repo://my-repo/statistics")

    # Analyze code complexity
    complexity = await client.get_resource("repo://my-repo/complexity-analysis")

    # Find code hotspots
    hotspots = await client.get_resource("repo://my-repo/hotspots")

    # Search for entities
    entities = await client.get_resource("repo://my-repo/search")
```

## 🔍 Key Features

### Repository Analysis

- **Code Parsing**: Automatic analysis of Python repositories
- **Entity Extraction**: Functions, classes, variables, imports
- **Relationship Mapping**: Calls, imports, inheritance, usage patterns
- **Multi-language Support**: Extensible to other programming languages

### Advanced Analytics

- **Complexity Analysis**: Identify complex functions and potential refactoring candidates
- **Hotspot Detection**: Find central, critical code components using graph centrality
- **Dependency Analysis**: Circular dependencies, high coupling detection
- **Impact Analysis**: Understand the ripple effects of code changes
- **Quality Metrics**: Overall code quality scoring and recommendations

### Search & Context

- **Fuzzy Search**: Find entities by partial names, types, file paths
- **Graph Traversal**: Navigate relationships between code entities
- **Context Retrieval**: Get full context including code, dependencies, callers
- **Audit Trails**: Track all changes and operations

### Graph Operations

- **Subgraphs**: Extract focused views around specific entities
- **Neighborhoods**: Find directly connected entities
- **Tree Traversal**: Call trees, inheritance hierarchies
- **Bulk Operations**: Efficient batch processing of multiple entities

## 🛡️ Error Handling

The unified server includes comprehensive error handling:

- Repository existence validation
- Entity existence checks
- Graceful handling of graph operations on disconnected components
- Detailed error messages for troubleshooting

## 📊 Resource URI Patterns

The server uses consistent URI patterns for resources:

- `repos://list` - Global repository list
- `repo://{repo_id}/...` - Repository-specific resources
- `repo://{repo_id}/entity/{entity_uuid}/...` - Entity-specific resources

## 🔧 Configuration

### Environment Variables

- `LOG_LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)
- `STORAGE_PATH` - Custom storage directory path
- `MAX_GRAPH_SIZE` - Maximum nodes for graph operations

### Server Options

The server supports:

- `stateless_http=True` - HTTP transport mode
- Custom port configuration
- SSL/TLS support for production deployments

## 🤝 Integration

The unified MCP server integrates seamlessly with:

- **Claude Desktop** - Direct AI assistant integration
- **VS Code Extensions** - Code analysis in your editor
- **Custom Applications** - Any MCP-compatible client
- **Jupyter Notebooks** - Interactive code analysis
- **CI/CD Pipelines** - Automated code quality checks

## 📈 Performance

- **Efficient Graph Storage**: NetworkX-based graph operations
- **Indexed Search**: Fast entity lookups and filtering
- **Streaming Support**: Handle large repositories
- **Caching**: Intelligent caching of expensive operations
- **Pagination**: Built-in pagination for large result sets

## 🐛 Troubleshooting

### Common Issues

1. **"Repository not found"** - Ensure repository ID exists and is valid
2. **"Entity not found"** - Check entity UUID and repository ID
3. **Graph operation failures** - May occur with very large or disconnected graphs
4. **Import errors** - Verify all dependencies are installed
5. **"Unexpected token" errors in Claude Desktop** - Server output is interfering with JSON protocol
6. **"Server disconnected" in Claude Desktop** - Check paths in configuration are correct

### Debug Mode

For Claude Desktop debugging, check the MCP server logs in Claude Desktop's developer console.

For local debugging:

```bash
# Test the server directly (should run silently)
python server.py

# Check imports
python test_imports.py

# Test with verbose logging (avoid with Claude Desktop)
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import server
"
```

### Claude Desktop Configuration Troubleshooting

1. **Check the configuration file path:**

   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. **Verify your configuration:**

   ```json
   {
     "mcpServers": {
       "KnowledgeGraphMCP": {
         "command": "C:\\Users\\YOUR_USERNAME\\Desktop\\kgv2\\.venv\\Scripts\\python.exe",
         "args": ["C:\\Users\\YOUR_USERNAME\\Desktop\\kgv2\\server.py"],
         "env": {}
       }
     }
   }
   ```

3. **Common configuration errors:**
   - Wrong Python path (should be virtual environment path)
   - Wrong server.py path
   - Missing double backslashes in Windows paths
   - Incorrect server name (must match FastMCP name)

### Verification

Test server health:

```bash
# Test imports and basic functionality
python test_imports.py

# Run server locally (should start silently)
python server.py
```

## 🚀 Quick Start

1. **Install and activate virtual environment**:

```bash
# If using UV package manager
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. **Run the installation**:

```bash
python install_mcp.py
```

3. **Test the server**:

```bash
mcp inspect knowledge-graph-unified
```

4. **Create your first repository**:

```bash
mcp call knowledge-graph-unified create_repository '{"repo_path": "/path/to/your/code", "repo_id": "test-repo"}'
```

5. **Explore the results**:

```bash
mcp get knowledge-graph-unified "repo://test-repo/statistics"
mcp get knowledge-graph-unified "repo://test-repo/complexity-analysis"
mcp get knowledge-graph-unified "repo://test-repo/hotspots"
```

## 📝 License

This project is available under the same license as the original codebase.

## 🤝 Contributing

Contributions are welcome! Please ensure:

- New tools/resources follow MCP patterns
- Include proper error handling
- Add documentation for new features
- Test with multiple repository types
