# Quick Claude Desktop Setup

## Step 1: Find Claude Desktop Config File

**Windows:**

```
%APPDATA%\Claude\claude_desktop_config.json
```

**Full path example:**

```
C:\Users\YOUR_USERNAME\AppData\Roaming\Claude\claude_desktop_config.json
```

## Step 2: Add Server Configuration

Copy the contents of `claude_desktop_config.json` from this project, or add this configuration:

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

**⚠️ Important:** Update the paths to match your system:

- Replace `C:\\Users\\ItamarZand\\Desktop\\kgv2` with your actual project path
- Use double backslashes (`\\`) in Windows paths
- Make sure the virtual environment path is correct

## Step 3: Restart Claude Desktop

1. Close Claude Desktop completely
2. Restart Claude Desktop
3. The KnowledgeGraphMCP server should now be available

## Step 4: Test the Integration

In Claude Desktop, try:

```
Please list the available MCP tools and resources
```

You should see:

- **Tools:** create_repository, delete_repository, create_entity, etc.
- **Resources:** repos://list, repo://_/entities, repo://_/complexity-analysis, etc.

## Common Issues

1. **"Server disconnected"** - Check that paths are correct and files exist
2. **"Unexpected token" errors** - Make sure server.py runs silently (no print statements to stdout)
3. **No MCP tools available** - Verify the server name matches "KnowledgeGraphMCP"

## Test the Server Locally

Before configuring Claude Desktop, test locally:

```bash
# Check all dependencies
python test_imports.py

# Test server startup (should be silent)
python server.py
```

If you see any output or errors, the server needs fixing before Claude Desktop integration.
