"""
Rule Engine - Advanced Pattern Recognition and Rule Application

This module provides a sophisticated rule engine for detecting code patterns,
applying transformations, and enforcing coding standards.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Callable, Pattern, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from models import GraphNode, GraphEdge

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of rules supported by the engine."""
    SYNTAX_PATTERN = "syntax_pattern"
    SEMANTIC_PATTERN = "semantic_pattern"
    ARCHITECTURE_PATTERN = "architecture_pattern"
    ANTI_PATTERN = "anti_pattern"
    QUALITY_RULE = "quality_rule"
    SECURITY_RULE = "security_rule"
    PERFORMANCE_RULE = "performance_rule"


class RuleSeverity(Enum):
    """Severity levels for rule violations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RuleMatch:
    """Represents a match found by a rule."""
    rule_id: str
    rule_name: str
    rule_type: RuleType
    severity: RuleSeverity
    file_path: str
    line_number: int
    column: Optional[int] = None
    matched_text: str = ""
    context: str = ""
    message: str = ""
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Rule:
    """Definition of a pattern matching rule."""
    id: str
    name: str
    description: str
    rule_type: RuleType
    severity: RuleSeverity
    pattern: Union[str, Pattern, Callable]
    matcher: Callable[[str, str], List[RuleMatch]]
    enabled: bool = True
    languages: List[str] = field(default_factory=lambda: ["*"])
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternMatcher(ABC):
    """Abstract base for pattern matchers."""
    
    @abstractmethod
    def match(self, content: str, file_path: str) -> List[RuleMatch]:
        """Find pattern matches in content."""
        pass


class RegexPatternMatcher(PatternMatcher):
    """Regex-based pattern matcher."""
    
    def __init__(self, rule: Rule):
        self.rule = rule
        if isinstance(rule.pattern, str):
            self.compiled_pattern = re.compile(rule.pattern, re.MULTILINE)
        elif isinstance(rule.pattern, Pattern):
            self.compiled_pattern = rule.pattern
        else:
            raise ValueError("Pattern must be string or compiled regex")
    
    def match(self, content: str, file_path: str) -> List[RuleMatch]:
        """Find regex matches in content."""
        matches = []
        
        for match in self.compiled_pattern.finditer(content):
            line_number = content[:match.start()].count('\n') + 1
            matched_text = match.group(0)
            
            # Extract context (line containing the match)
            lines = content.split('\n')
            context = lines[line_number - 1] if line_number <= len(lines) else ""
            
            rule_match = RuleMatch(
                rule_id=self.rule.id,
                rule_name=self.rule.name,
                rule_type=self.rule.rule_type,
                severity=self.rule.severity,
                file_path=file_path,
                line_number=line_number,
                column=match.start() - content.rfind('\n', 0, match.start()),
                matched_text=matched_text,
                context=context.strip(),
                message=self.rule.description,
                metadata={"groups": match.groups()}
            )
            matches.append(rule_match)
        
        return matches


class SemanticPatternMatcher(PatternMatcher):
    """Semantic pattern matcher for complex code patterns."""
    
    def __init__(self, rule: Rule):
        self.rule = rule
        if not callable(rule.pattern):
            raise ValueError("Semantic patterns require callable pattern function")
        self.pattern_func = rule.pattern
    
    def match(self, content: str, file_path: str) -> List[RuleMatch]:
        """Find semantic pattern matches."""
        try:
            return self.pattern_func(content, file_path, self.rule)
        except Exception as e:
            logger.error(f"Semantic pattern matching failed for rule {self.rule.id}: {e}")
            return []


class RuleEngine:
    """
    Advanced rule engine for pattern detection and code analysis.
    
    This engine applies various types of rules to detect patterns,
    anti-patterns, quality issues, and architectural violations.
    """
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.matchers: Dict[str, PatternMatcher] = {}
        self.rule_sets: Dict[str, List[str]] = {}
        self._initialize_default_rules()
    
    def add_rule(self, rule: Rule) -> None:
        """Add a new rule to the engine."""
        self.rules[rule.id] = rule
        
        # Create appropriate matcher
        if rule.rule_type in [RuleType.SYNTAX_PATTERN, RuleType.ANTI_PATTERN]:
            self.matchers[rule.id] = RegexPatternMatcher(rule)
        elif rule.rule_type in [RuleType.SEMANTIC_PATTERN, RuleType.ARCHITECTURE_PATTERN]:
            self.matchers[rule.id] = SemanticPatternMatcher(rule)
        else:
            # Default to regex matcher
            self.matchers[rule.id] = RegexPatternMatcher(rule)
    
    def apply_rules(
        self, 
        content: str, 
        file_path: str, 
        language: str = "python",
        rule_set: Optional[str] = None
    ) -> List[RuleMatch]:
        """Apply rules to content and return matches."""
        matches = []
        
        # Determine which rules to apply
        rules_to_apply = self._get_applicable_rules(language, rule_set)
        
        for rule_id in rules_to_apply:
            rule = self.rules[rule_id]
            if not rule.enabled:
                continue
            
            matcher = self.matchers.get(rule_id)
            if matcher:
                try:
                    rule_matches = matcher.match(content, file_path)
                    matches.extend(rule_matches)
                except Exception as e:
                    logger.error(f"Rule {rule_id} failed: {e}")
        
        return matches
    
    def create_rule_set(self, name: str, rule_ids: List[str]) -> None:
        """Create a named set of rules."""
        self.rule_sets[name] = rule_ids
    
    def enable_rule(self, rule_id: str) -> None:
        """Enable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
    
    def disable_rule(self, rule_id: str) -> None:
        """Disable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
    
    def get_rules_by_type(self, rule_type: RuleType) -> List[Rule]:
        """Get all rules of a specific type."""
        return [rule for rule in self.rules.values() if rule.rule_type == rule_type]
    
    def get_rules_by_severity(self, severity: RuleSeverity) -> List[Rule]:
        """Get all rules of a specific severity."""
        return [rule for rule in self.rules.values() if rule.severity == severity]
    
    def _get_applicable_rules(self, language: str, rule_set: Optional[str]) -> List[str]:
        """Get rules applicable to the given language and rule set."""
        if rule_set and rule_set in self.rule_sets:
            return self.rule_sets[rule_set]
        
        # Filter by language
        applicable_rules = []
        for rule_id, rule in self.rules.items():
            if "*" in rule.languages or language in rule.languages:
                applicable_rules.append(rule_id)
        
        return applicable_rules
    
    def _initialize_default_rules(self) -> None:
        """Initialize default rules for common patterns and anti-patterns."""
        
        # Python-specific rules
        self._add_python_rules()
        
        # JavaScript-specific rules  
        self._add_javascript_rules()
        
        # General rules
        self._add_general_rules()
        
        # Quality rules
        self._add_quality_rules()
        
        # Security rules
        self._add_security_rules()
    
    def _add_python_rules(self) -> None:
        """Add Python-specific rules."""
        
        # Anti-pattern: Bare except
        self.add_rule(Rule(
            id="python_bare_except",
            name="Bare Except Clause",
            description="Avoid bare except clauses that catch all exceptions",
            rule_type=RuleType.ANTI_PATTERN,
            severity=RuleSeverity.WARNING,
            pattern=r"except\s*:",
            matcher=lambda content, path: [],  # Will be set by RegexPatternMatcher
            languages=["python"],
            tags=["exception_handling", "anti_pattern"]
        ))
        
        # Quality: Missing docstrings
        self.add_rule(Rule(
            id="python_missing_docstring",
            name="Missing Function Docstring",
            description="Functions should have docstrings",
            rule_type=RuleType.QUALITY_RULE,
            severity=RuleSeverity.INFO,
            pattern=r'def\s+[a-zA-Z_]\w*\s*\([^)]*\):\s*\n(?!\s*"""|\s*\'\'\')',
            matcher=lambda content, path: [],
            languages=["python"],
            tags=["documentation", "quality"]
        ))
        
        # Performance: List comprehension opportunity
        self.add_rule(Rule(
            id="python_list_comprehension",
            name="List Comprehension Opportunity", 
            description="Consider using list comprehension for better performance",
            rule_type=RuleType.QUALITY_RULE,
            severity=RuleSeverity.INFO,
            pattern=r'\w+\s*=\s*\[\]\s*\n\s*for\s+\w+\s+in\s+.+:\s*\n\s*\w+\.append\(',
            matcher=lambda content, path: [],
            languages=["python"],
            tags=["performance", "pythonic"]
        ))
    
    def _add_javascript_rules(self) -> None:
        """Add JavaScript-specific rules."""
        
        # Anti-pattern: var usage
        self.add_rule(Rule(
            id="js_var_usage",
            name="Avoid var Declaration",
            description="Use 'let' or 'const' instead of 'var'",
            rule_type=RuleType.ANTI_PATTERN,
            severity=RuleSeverity.WARNING,
            pattern=r"\bvar\s+\w+",
            matcher=lambda content, path: [],
            languages=["javascript", "typescript"],
            tags=["modern_js", "scoping"]
        ))
        
        # Quality: Missing semicolons
        self.add_rule(Rule(
            id="js_missing_semicolon",
            name="Missing Semicolon",
            description="Statements should end with semicolons",
            rule_type=RuleType.QUALITY_RULE,
            severity=RuleSeverity.INFO,
            pattern=r"[^;\s]\s*\n\s*[^\/\*\s]",
            matcher=lambda content, path: [],
            languages=["javascript", "typescript"],
            tags=["syntax", "consistency"]
        ))
    
    def _add_general_rules(self) -> None:
        """Add language-agnostic rules."""
        
        # Long lines
        self.add_rule(Rule(
            id="long_lines",
            name="Line Too Long",
            description="Lines should not exceed 120 characters",
            rule_type=RuleType.QUALITY_RULE,
            severity=RuleSeverity.INFO,
            pattern=r".{121,}",
            matcher=lambda content, path: [],
            languages=["*"],
            tags=["formatting", "readability"]
        ))
        
        # TODO comments
        self.add_rule(Rule(
            id="todo_comments",
            name="TODO Comments",
            description="TODO comments found - consider addressing",
            rule_type=RuleType.QUALITY_RULE,
            severity=RuleSeverity.INFO,
            pattern=r"(?:#|//|/\*)\s*TODO\s*:?\s*(.+)",
            matcher=lambda content, path: [],
            languages=["*"],
            tags=["technical_debt", "maintenance"]
        ))
    
    def _add_quality_rules(self) -> None:
        """Add code quality rules."""
        
        # Complex function detection
        self.add_rule(Rule(
            id="complex_function",
            name="Complex Function",
            description="Function appears to be overly complex",
            rule_type=RuleType.QUALITY_RULE,
            severity=RuleSeverity.WARNING,
            pattern=r"def\s+\w+\([^)]*\):[^}]{200,}",  # Simple heuristic: function with lots of content
            matcher=lambda content, path: [],
            languages=["*"],
            tags=["complexity", "maintainability"]
        ))
    
    def _add_security_rules(self) -> None:
        """Add security-related rules."""
        
        # Hardcoded secrets
        self.add_rule(Rule(
            id="hardcoded_secret",
            name="Potential Hardcoded Secret",
            description="Potential hardcoded password or API key detected",
            rule_type=RuleType.SECURITY_RULE,
            severity=RuleSeverity.ERROR,
            pattern=r"(?:password|secret|key|token)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
            matcher=lambda content, path: [],
            languages=["*"],
            tags=["security", "credentials"]
        ))
    
    def _create_missing_docstring_pattern(self) -> Callable:
        """Create semantic pattern for missing docstrings."""
        def find_missing_docstrings(content: str, file_path: str, rule: Rule) -> List[RuleMatch]:
            matches = []
            lines = content.split('\n')
            
            # Find function definitions
            func_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(', re.MULTILINE)
            
            for match in func_pattern.finditer(content):
                func_name = match.group(1)
                line_num = content[:match.start()].count('\n') + 1
                
                # Check if next non-empty line contains docstring
                has_docstring = False
                for i in range(line_num, min(line_num + 5, len(lines))):
                    if i < len(lines):
                        line = lines[i].strip()
                        if line and (line.startswith('"""') or line.startswith("'''")):
                            has_docstring = True
                            break
                        elif line and not line.endswith(':'):
                            break
                
                if not has_docstring and not func_name.startswith('_'):
                    matches.append(RuleMatch(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        rule_type=rule.rule_type,
                        severity=rule.severity,
                        file_path=file_path,
                        line_number=line_num,
                        matched_text=match.group(0),
                        context=lines[line_num - 1] if line_num <= len(lines) else "",
                        message=f"Function '{func_name}' is missing a docstring",
                        suggestion="Add a docstring to document this function"
                    ))
            
            return matches
        
        return find_missing_docstrings
    
    def _create_list_comp_pattern(self) -> Callable:
        """Create pattern for list comprehension opportunities."""
        def find_list_comp_opportunities(content: str, file_path: str, rule: Rule) -> List[RuleMatch]:
            matches = []
            
            # Look for simple for loops that could be list comprehensions
            pattern = re.compile(
                r'(\w+)\s*=\s*\[\]\s*\n\s*for\s+(\w+)\s+in\s+(.+):\s*\n\s*\1\.append\((.+)\)',
                re.MULTILINE
            )
            
            for match in pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                
                matches.append(RuleMatch(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    rule_type=rule.rule_type,
                    severity=rule.severity,
                    file_path=file_path,
                    line_number=line_num,
                    matched_text=match.group(0),
                    context=match.group(0).replace('\n', ' '),
                    message="This loop could be a list comprehension",
                    suggestion=f"{match.group(1)} = [{match.group(4)} for {match.group(2)} in {match.group(3)}]"
                ))
            
            return matches
        
        return find_list_comp_opportunities
    
    def _create_complexity_pattern(self) -> Callable:
        """Create pattern for detecting complex functions."""
        def find_complex_functions(content: str, file_path: str, rule: Rule) -> List[RuleMatch]:
            matches = []
            
            # Simple complexity heuristic: count control flow statements
            func_pattern = re.compile(r'def\s+(\w+)\s*\(.*?\):(.+?)(?=\ndef|\nclass|\Z)', 
                                   re.DOTALL | re.MULTILINE)
            
            for match in func_pattern.finditer(content):
                func_name = match.group(1)
                func_body = match.group(2)
                
                # Count complexity indicators
                complexity_indicators = [
                    'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 
                    'with ', 'and ', 'or ', 'not '
                ]
                
                complexity = sum(func_body.count(indicator) for indicator in complexity_indicators)
                
                if complexity > 10:  # Threshold for complexity
                    line_num = content[:match.start()].count('\n') + 1
                    
                    matches.append(RuleMatch(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        rule_type=rule.rule_type,
                        severity=rule.severity,
                        file_path=file_path,
                        line_number=line_num,
                        matched_text=f"def {func_name}",
                        context=f"Function with complexity score: {complexity}",
                        message=f"Function '{func_name}' has high complexity (score: {complexity})",
                        suggestion="Consider breaking this function into smaller functions",
                        metadata={"complexity_score": complexity}
                    ))
            
            return matches
        
        return find_complex_functions 