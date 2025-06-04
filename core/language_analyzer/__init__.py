"""
Language Analyzer Module - Multi-Language Code Analysis

This module provides language detection and analysis capabilities
for multiple programming languages with specialized analyzers.
"""

# Language detection and classification
from .detector import LanguageDetector, SupportedLanguage, LanguageInfo

# Multi-language analysis capabilities
from .multi_lang_analyzer import (
    MultiLanguageAnalyzer,
    LanguageSpecificAnalyzer,
    PythonAnalyzer,
    JavaScriptAnalyzer
)

__all__ = [
    'LanguageDetector',
    'SupportedLanguage', 
    'LanguageInfo',
    'MultiLanguageAnalyzer',
    'LanguageSpecificAnalyzer',
    'PythonAnalyzer',
    'JavaScriptAnalyzer'
] 