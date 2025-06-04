#!/usr/bin/env python3
"""
Performance configuration for MCP analysis
"""

# Parallel processing settings
PARALLEL_CONFIG = {
    "file_batch_size": 10,      # Files processed in parallel per batch
    "storage_batch_size": 50,   # Nodes/edges stored in parallel per batch
    "max_files_limit": 0,       # Maximum files to analyze (0 = no limit)
    "enable_parallel": True,    # Enable parallel processing
    "smart_prioritization": True,  # Use smart file prioritization for large repos
}

# File filtering settings
FILTER_CONFIG = {
    "skip_test_files": True,        # Skip test_*.py files
    "skip_docs": True,              # Skip docs/ and documentation/ dirs
    "skip_examples": True,          # Skip examples/ dir
    "skip_small_init": True,        # Skip __init__.py < 100 bytes
    "skip_setup_files": True,       # Skip setup.py files
    "min_file_size": 10,            # Skip files smaller than N bytes
    "max_file_size": 1000000,       # Skip files larger than N bytes (1MB)
}

# Analysis optimization settings
ANALYSIS_CONFIG = {
    "enable_pagerank": True,        # Enable PageRank analysis
    "enable_cross_file": True,      # Enable cross-file relationship detection
    "enable_audit_events": True,    # Enable audit logging (can be slow)
    "cache_ast_parsing": False,     # Cache parsed ASTs (experimental)
}

# Timeout settings (seconds)
TIMEOUT_CONFIG = {
    "clone_timeout": 300,           # 5 minutes for cloning
    "analysis_timeout": 600,        # 10 minutes for analysis
    "file_timeout": 30,             # 30 seconds per file
    "batch_timeout": 120,           # 2 minutes per batch
}

# Logging settings
LOGGING_CONFIG = {
    "log_level": "INFO",            # DEBUG, INFO, WARNING, ERROR
    "log_progress": True,           # Log progress every N files
    "progress_interval": 10,        # Log every N files
    "detailed_timing": True,        # Log detailed timing information
}

def get_optimized_config(repo_size: str = "medium") -> dict:
    """Get optimized configuration based on repository size"""
    
    configs = {
        "small": {
            "file_batch_size": 5,
            "max_files_limit": 0,     # No limit
            "storage_batch_size": 25,
            "analysis_timeout": 300,  # 5 minutes
        },
        "medium": {
            "file_batch_size": 10,
            "max_files_limit": 0,     # No limit
            "storage_batch_size": 50,
            "analysis_timeout": 600,  # 10 minutes
        },
        "large": {
            "file_batch_size": 15,
            "max_files_limit": 0,     # No limit, use smart prioritization
            "storage_batch_size": 100,
            "analysis_timeout": 1800,  # 30 minutes
        },
        "huge": {
            "file_batch_size": 25,
            "max_files_limit": 0,     # No limit, use smart prioritization
            "storage_batch_size": 200,
            "analysis_timeout": 3600,  # 1 hour
        }
    }
    
    base_config = {
        **PARALLEL_CONFIG,
        **FILTER_CONFIG,
        **ANALYSIS_CONFIG,
        **TIMEOUT_CONFIG,
        **LOGGING_CONFIG
    }
    
    if repo_size in configs:
        base_config.update(configs[repo_size])
    
    return base_config

def estimate_repo_size(file_count: int) -> str:
    """Estimate repository size based on file count"""
    if file_count <= 20:
        return "small"
    elif file_count <= 50:
        return "medium"  
    elif file_count <= 150:
        return "large"
    else:
        return "huge"

# Export current configuration
CURRENT_CONFIG = get_optimized_config("medium")

if __name__ == "__main__":
    print("🔧 MCP Performance Configuration")
    print("="*50)
    
    for size in ["small", "medium", "large", "huge"]:
        config = get_optimized_config(size)
        print(f"\n📊 {size.upper()} repositories:")
        print(f"   - File batch size: {config['file_batch_size']}")
        print(f"   - Max files: {config['max_files_limit']}")
        print(f"   - Analysis timeout: {config['analysis_timeout']}s")
        print(f"   - Storage batch: {config['storage_batch_size']}")
    
    print(f"\n🎯 Current config (medium): {CURRENT_CONFIG['file_batch_size']} files/batch, {CURRENT_CONFIG['max_files_limit']} max files") 