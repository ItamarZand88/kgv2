"""
Language Detector - Advanced Language Detection and Classification

This module provides sophisticated language detection capabilities
for identifying programming languages from file content and metadata.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from models import GraphNode


class SupportedLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    UNKNOWN = "unknown"


@dataclass
class LanguageInfo:
    """Information about detected language."""
    language: SupportedLanguage
    confidence: float
    detection_method: str
    file_extension: Optional[str] = None
    syntax_indicators: List[str] = None
    metadata: Dict = None


class LanguageDetector:
    """
    Advanced language detector using multiple detection strategies.
    
    This detector combines file extension analysis, syntax pattern
    recognition, and heuristic analysis for accurate language detection.
    """
    
    def __init__(self):
        self.extension_map = self._build_extension_map()
        self.syntax_patterns = self._build_syntax_patterns()
        self.heuristics = self._build_heuristics()
    
    def detect_language(self, file_path: str, content: str = "") -> LanguageInfo:
        """
        Detect programming language from file path and content.
        
        Args:
            file_path: Path to the file
            content: File content (optional)
            
        Returns:
            LanguageInfo with detection results
        """
        # Strategy 1: File extension detection
        ext_result = self._detect_by_extension(file_path)
        
        # Strategy 2: Content-based detection (if content provided)
        content_result = None
        if content:
            content_result = self._detect_by_content(content)
        
        # Strategy 3: Heuristic analysis
        heuristic_result = None
        if content:
            heuristic_result = self._detect_by_heuristics(content)
        
        # Combine results and choose best match
        return self._combine_detection_results(
            ext_result, content_result, heuristic_result, file_path
        )
    
    def get_language_features(self, language: SupportedLanguage) -> Dict:
        """Get language-specific features and characteristics."""
        features = {
            SupportedLanguage.PYTHON: {
                "paradigms": ["object-oriented", "functional", "procedural"],
                "typing": "dynamic",
                "indentation_significant": True,
                "common_frameworks": ["Django", "Flask", "FastAPI", "NumPy"],
                "package_managers": ["pip", "conda", "poetry"],
                "file_extensions": [".py", ".pyw", ".pyx"],
                "test_frameworks": ["pytest", "unittest", "nose"],
                "async_support": True
            },
            SupportedLanguage.JAVASCRIPT: {
                "paradigms": ["functional", "object-oriented", "event-driven"],
                "typing": "dynamic",
                "indentation_significant": False,
                "common_frameworks": ["React", "Vue", "Angular", "Express", "Node.js"],
                "package_managers": ["npm", "yarn", "pnpm"],
                "file_extensions": [".js", ".jsx", ".mjs"],
                "test_frameworks": ["Jest", "Mocha", "Jasmine"],
                "async_support": True
            },
            SupportedLanguage.TYPESCRIPT: {
                "paradigms": ["object-oriented", "functional"],
                "typing": "static",
                "indentation_significant": False,
                "common_frameworks": ["Angular", "React", "Vue", "NestJS"],
                "package_managers": ["npm", "yarn", "pnpm"],
                "file_extensions": [".ts", ".tsx", ".d.ts"],
                "test_frameworks": ["Jest", "Mocha", "Vitest"],
                "async_support": True
            },
            SupportedLanguage.JAVA: {
                "paradigms": ["object-oriented"],
                "typing": "static",
                "indentation_significant": False,
                "common_frameworks": ["Spring", "Hibernate", "Struts"],
                "package_managers": ["Maven", "Gradle"],
                "file_extensions": [".java"],
                "test_frameworks": ["JUnit", "TestNG"],
                "async_support": True
            },
            SupportedLanguage.GO: {
                "paradigms": ["procedural", "concurrent"],
                "typing": "static",
                "indentation_significant": False,
                "common_frameworks": ["Gin", "Echo", "Fiber"],
                "package_managers": ["go mod"],
                "file_extensions": [".go"],
                "test_frameworks": ["testing"],
                "async_support": True
            }
        }
        
        return features.get(language, {})
    
    def _build_extension_map(self) -> Dict[str, SupportedLanguage]:
        """Build mapping from file extensions to languages."""
        return {
            # Python
            '.py': SupportedLanguage.PYTHON,
            '.pyw': SupportedLanguage.PYTHON,
            '.pyx': SupportedLanguage.PYTHON,
            
            # JavaScript
            '.js': SupportedLanguage.JAVASCRIPT,
            '.jsx': SupportedLanguage.JAVASCRIPT,
            '.mjs': SupportedLanguage.JAVASCRIPT,
            
            # TypeScript
            '.ts': SupportedLanguage.TYPESCRIPT,
            '.tsx': SupportedLanguage.TYPESCRIPT,
            
            # Java
            '.java': SupportedLanguage.JAVA,
            
            # C/C++
            '.c': SupportedLanguage.C,
            '.h': SupportedLanguage.C,
            '.cpp': SupportedLanguage.CPP,
            '.cxx': SupportedLanguage.CPP,
            '.cc': SupportedLanguage.CPP,
            '.hpp': SupportedLanguage.CPP,
            
            # C#
            '.cs': SupportedLanguage.CSHARP,
            
            # Go
            '.go': SupportedLanguage.GO,
            
            # Rust
            '.rs': SupportedLanguage.RUST,
            
            # PHP
            '.php': SupportedLanguage.PHP,
            '.phtml': SupportedLanguage.PHP,
            
            # Ruby
            '.rb': SupportedLanguage.RUBY,
            '.rbw': SupportedLanguage.RUBY,
            
            # Swift
            '.swift': SupportedLanguage.SWIFT,
            
            # Kotlin
            '.kt': SupportedLanguage.KOTLIN,
            '.kts': SupportedLanguage.KOTLIN,
            
            # Scala
            '.scala': SupportedLanguage.SCALA,
            '.sc': SupportedLanguage.SCALA,
        }
    
    def _build_syntax_patterns(self) -> Dict[SupportedLanguage, List[Tuple[str, float]]]:
        """Build syntax patterns for content-based detection."""
        return {
            SupportedLanguage.PYTHON: [
                (r'^\s*def\s+\w+\s*\(.*\):', 0.8),
                (r'^\s*class\s+\w+.*:', 0.8),
                (r'^\s*import\s+\w+', 0.7),
                (r'^\s*from\s+\w+\s+import', 0.7),
                (r'^\s*if\s+__name__\s*==\s*["\']__main__["\']:', 0.9),
                (r'print\s*\(', 0.6),
                (r'^\s*#.*', 0.3),
            ],
            SupportedLanguage.JAVASCRIPT: [
                (r'function\s+\w+\s*\(', 0.8),
                (r'const\s+\w+\s*=', 0.7),
                (r'let\s+\w+\s*=', 0.7),
                (r'var\s+\w+\s*=', 0.6),
                (r'require\s*\(', 0.8),
                (r'module\.exports', 0.8),
                (r'console\.log\s*\(', 0.7),
                (r'=>', 0.6),
            ],
            SupportedLanguage.TYPESCRIPT: [
                (r'interface\s+\w+', 0.9),
                (r'type\s+\w+\s*=', 0.8),
                (r':\s*\w+(\[\])?(\s*\|\s*\w+)*\s*[=;]', 0.8),
                (r'function\s+\w+\s*\([^)]*:\s*\w+', 0.8),
                (r'import.*from\s+["\']', 0.7),
                (r'export\s+(default\s+)?', 0.7),
            ],
            SupportedLanguage.JAVA: [
                (r'public\s+class\s+\w+', 0.9),
                (r'public\s+static\s+void\s+main', 0.9),
                (r'import\s+[\w.]+;', 0.8),
                (r'package\s+[\w.]+;', 0.8),
                (r'@\w+', 0.7),
                (r'System\.out\.println', 0.7),
            ],
            SupportedLanguage.GO: [
                (r'package\s+\w+', 0.9),
                (r'func\s+\w+\s*\(', 0.8),
                (r'import\s*\(', 0.8),
                (r'fmt\.Print', 0.8),
                (r'go\s+\w+\s*\(', 0.7),
                (r'defer\s+', 0.7),
            ]
        }
    
    def _build_heuristics(self) -> Dict[SupportedLanguage, List[Tuple[str, float]]]:
        """Build heuristic patterns for advanced detection."""
        return {
            SupportedLanguage.PYTHON: [
                (r'__init__\.py', 0.9),
                (r'requirements\.txt', 0.8),
                (r'setup\.py', 0.8),
                (r'pytest', 0.7),
                (r'django|flask|fastapi', 0.6),
            ],
            SupportedLanguage.JAVASCRIPT: [
                (r'package\.json', 0.9),
                (r'node_modules', 0.8),
                (r'npm|yarn', 0.7),
                (r'react|vue|angular', 0.6),
            ],
            SupportedLanguage.TYPESCRIPT: [
                (r'tsconfig\.json', 0.9),
                (r'\.d\.ts', 0.8),
                (r'@types/', 0.7),
            ],
            SupportedLanguage.JAVA: [
                (r'pom\.xml', 0.9),
                (r'build\.gradle', 0.8),
                (r'\.jar', 0.7),
                (r'maven|gradle', 0.6),
            ]
        }
    
    def _detect_by_extension(self, file_path: str) -> Optional[LanguageInfo]:
        """Detect language by file extension."""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in self.extension_map:
            language = self.extension_map[ext]
            return LanguageInfo(
                language=language,
                confidence=0.9,
                detection_method="extension",
                file_extension=ext
            )
        
        return None
    
    def _detect_by_content(self, content: str) -> Optional[LanguageInfo]:
        """Detect language by content patterns."""
        scores = {}
        indicators = {}
        
        for language, patterns in self.syntax_patterns.items():
            score = 0
            found_indicators = []
            
            for pattern, weight in patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                if matches:
                    score += weight * min(len(matches), 5)  # Cap contribution per pattern
                    found_indicators.append(pattern)
            
            if score > 0:
                scores[language] = score
                indicators[language] = found_indicators
        
        if scores:
            best_language = max(scores, key=scores.get)
            confidence = min(scores[best_language] / 10.0, 1.0)  # Normalize to 0-1
            
            return LanguageInfo(
                language=best_language,
                confidence=confidence,
                detection_method="content",
                syntax_indicators=indicators[best_language]
            )
        
        return None
    
    def _detect_by_heuristics(self, content: str) -> Optional[LanguageInfo]:
        """Detect language using heuristic analysis."""
        scores = {}
        
        for language, heuristics in self.heuristics.items():
            score = 0
            
            for pattern, weight in heuristics:
                if re.search(pattern, content, re.IGNORECASE):
                    score += weight
            
            if score > 0:
                scores[language] = score
        
        if scores:
            best_language = max(scores, key=scores.get)
            confidence = min(scores[best_language] / 5.0, 1.0)  # Normalize
            
            return LanguageInfo(
                language=best_language,
                confidence=confidence,
                detection_method="heuristic"
            )
        
        return None
    
    def _combine_detection_results(
        self, 
        ext_result: Optional[LanguageInfo],
        content_result: Optional[LanguageInfo],
        heuristic_result: Optional[LanguageInfo],
        file_path: str
    ) -> LanguageInfo:
        """Combine multiple detection results into final decision."""
        results = [r for r in [ext_result, content_result, heuristic_result] if r]
        
        if not results:
            return LanguageInfo(
                language=SupportedLanguage.UNKNOWN,
                confidence=0.0,
                detection_method="none",
                metadata={"file_path": file_path}
            )
        
        # Weight the results based on reliability
        weights = {
            "extension": 1.0,
            "content": 1.2,
            "heuristic": 0.8
        }
        
        # Calculate weighted scores
        language_scores = {}
        for result in results:
            weight = weights.get(result.detection_method, 1.0)
            weighted_score = result.confidence * weight
            
            if result.language in language_scores:
                language_scores[result.language] += weighted_score
            else:
                language_scores[result.language] = weighted_score
        
        # Select best result
        best_language = max(language_scores, key=language_scores.get)
        final_confidence = min(language_scores[best_language] / len(results), 1.0)
        
        # Combine metadata
        combined_metadata = {"file_path": file_path}
        methods_used = [r.detection_method for r in results]
        
        return LanguageInfo(
            language=best_language,
            confidence=final_confidence,
            detection_method="+".join(methods_used),
            file_extension=ext_result.file_extension if ext_result else None,
            syntax_indicators=content_result.syntax_indicators if content_result else [],
            metadata=combined_metadata
        ) 