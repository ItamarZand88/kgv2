"""
Pattern Matchers Module - Advanced Pattern Recognition and Rule Application

This module provides sophisticated pattern matching capabilities including
rule engines, pattern detection, and code quality analysis.
"""

# Rule engine for pattern-based analysis
from .rule_engine import (
    RuleEngine,
    Rule,
    RuleMatch,
    RuleType,
    RuleSeverity,
    PatternMatcher,
    RegexPatternMatcher,
    SemanticPatternMatcher
)

# Advanced pattern detection
from .pattern_detector import (
    PatternDetector,
    PatternMatch,
    PatternType
)

__all__ = [
    'RuleEngine',
    'Rule',
    'RuleMatch', 
    'RuleType',
    'RuleSeverity',
    'PatternMatcher',
    'RegexPatternMatcher',
    'SemanticPatternMatcher',
    'PatternDetector',
    'PatternMatch',
    'PatternType'
] 