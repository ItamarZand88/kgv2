"""
Syntax Engine Module - Advanced Code Parsing and Analysis

This module provides sophisticated code parsing capabilities using multiple
parsing strategies including AST, tree-sitter, and hybrid approaches.
"""

# Base parser interface and result container
from .base_parser import BaseParser, ParseResult

# Hybrid approach combining AST and tree-sitter parsing strategies
from .hybrid_parser import HybridParser

__all__ = [
    'BaseParser',
    'ParseResult',
    'HybridParser'
] 