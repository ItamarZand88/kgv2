"""
Semantic Processor Module - Advanced Semantic Analysis

This module provides sophisticated semantic analysis capabilities for
extracting meaning and relationships from code structures.
"""

# Entity extraction and classification
from .entity_extractor import EntityExtractor, EntityType

# Relationship building and detection
from .relationship_builder import RelationshipBuilder, RelationshipType

# Cross-file analysis (will be implemented)
# from .cross_file_analyzer import CrossFileAnalyzer

__all__ = [
    'EntityExtractor',
    'EntityType',
    'RelationshipBuilder',
    'RelationshipType'
    # 'CrossFileAnalyzer'
] 