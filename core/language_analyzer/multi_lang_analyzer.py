"""
Multi-Language Analyzer - Comprehensive Code Analysis Across Languages

This module provides sophisticated analysis capabilities for multiple programming
languages, coordinating language detection and specialized analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod

from models import GraphNode
from .detector import LanguageDetector, SupportedLanguage, LanguageInfo
from ..syntax_engine.base_parser import BaseParser, ParseResult
from ..syntax_engine.ast_parser import ASTParser
from ..syntax_engine.advanced_parser import AdvancedParser
from ..syntax_engine.hybrid_parser import HybridParser

logger = logging.getLogger(__name__)


class LanguageSpecificAnalyzer(ABC):
    """Abstract base class for language-specific analyzers."""
    
    @abstractmethod
    def analyze(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze code content and return language-specific insights."""
        pass
    
    @abstractmethod
    def get_language_patterns(self) -> Dict[str, Any]:
        """Get language-specific patterns for analysis."""
        pass


class PythonAnalyzer(LanguageSpecificAnalyzer):
    """Python-specific code analyzer."""
    
    def analyze(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze Python code for language-specific patterns."""
        analysis = {
            "imports": self._analyze_imports(content),
            "decorators": self._analyze_decorators(content),
            "async_patterns": self._analyze_async_patterns(content),
            "comprehensions": self._analyze_comprehensions(content),
            "context_managers": self._analyze_context_managers(content),
            "type_hints": self._analyze_type_hints(content),
            "docstrings": self._analyze_docstrings(content),
            "frameworks": self._detect_frameworks(content)
        }
        return analysis
    
    def get_language_patterns(self) -> Dict[str, Any]:
        """Get Python-specific patterns."""
        return {
            "import_patterns": [
                r"import\s+(\w+)",
                r"from\s+(\w+(?:\.\w+)*)\s+import\s+(.+)",
                r"import\s+(\w+(?:\.\w+)*)\s+as\s+(\w+)"
            ],
            "decorator_patterns": [
                r"@(\w+(?:\.\w+)*)",
                r"@(\w+)\(.*?\)"
            ],
            "async_patterns": [
                r"async\s+def\s+(\w+)",
                r"await\s+(\w+)",
                r"asyncio\."
            ],
            "framework_indicators": {
                "Django": ["django", "models.Model", "HttpResponse"],
                "Flask": ["flask", "@app.route", "request", "Response"],
                "FastAPI": ["fastapi", "@app.get", "@app.post", "Depends"],
                "NumPy": ["numpy", "np.array", "np."],
                "Pandas": ["pandas", "pd.DataFrame", "pd."]
            }
        }
    
    def _analyze_imports(self, content: str) -> List[Dict[str, str]]:
        """Analyze Python import statements."""
        import re
        imports = []
        
        # Standard imports
        for match in re.finditer(r"import\s+(\w+(?:\.\w+)*)", content):
            imports.append({
                "type": "standard",
                "module": match.group(1),
                "alias": None
            })
        
        # From imports
        for match in re.finditer(r"from\s+(\w+(?:\.\w+)*)\s+import\s+(.+)", content):
            module = match.group(1)
            items = [item.strip() for item in match.group(2).split(',')]
            for item in items:
                imports.append({
                    "type": "from",
                    "module": module,
                    "item": item
                })
        
        # Aliased imports
        for match in re.finditer(r"import\s+(\w+(?:\.\w+)*)\s+as\s+(\w+)", content):
            imports.append({
                "type": "aliased",
                "module": match.group(1),
                "alias": match.group(2)
            })
        
        return imports
    
    def _analyze_decorators(self, content: str) -> List[Dict[str, str]]:
        """Analyze Python decorators."""
        import re
        decorators = []
        
        for match in re.finditer(r"@(\w+(?:\.\w+)*)", content, re.MULTILINE):
            decorators.append({
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })
        
        return decorators
    
    def _analyze_async_patterns(self, content: str) -> Dict[str, List]:
        """Analyze async/await patterns."""
        import re
        
        async_funcs = re.findall(r"async\s+def\s+(\w+)", content)
        await_calls = re.findall(r"await\s+(\w+)", content)
        asyncio_usage = bool(re.search(r"asyncio\.", content))
        
        return {
            "async_functions": async_funcs,
            "await_calls": await_calls,
            "uses_asyncio": asyncio_usage
        }
    
    def _analyze_comprehensions(self, content: str) -> Dict[str, int]:
        """Analyze list/dict/set comprehensions."""
        import re
        
        list_comps = len(re.findall(r"\[.+\s+for\s+.+\s+in\s+.+\]", content))
        dict_comps = len(re.findall(r"\{.+:\s*.+\s+for\s+.+\s+in\s+.+\}", content))
        set_comps = len(re.findall(r"\{.+\s+for\s+.+\s+in\s+.+\}", content))
        
        return {
            "list_comprehensions": list_comps,
            "dict_comprehensions": dict_comps,
            "set_comprehensions": set_comps
        }
    
    def _analyze_context_managers(self, content: str) -> List[str]:
        """Analyze context manager usage."""
        import re
        
        with_statements = re.findall(r"with\s+(\w+(?:\.\w+)*)", content)
        return with_statements
    
    def _analyze_type_hints(self, content: str) -> Dict[str, bool]:
        """Analyze type hint usage."""
        import re
        
        has_type_hints = bool(re.search(r":\s*\w+", content))
        has_return_annotations = bool(re.search(r"->\s*\w+", content))
        has_typing_imports = bool(re.search(r"from\s+typing\s+import", content))
        
        return {
            "has_type_hints": has_type_hints,
            "has_return_annotations": has_return_annotations,
            "uses_typing_module": has_typing_imports
        }
    
    def _analyze_docstrings(self, content: str) -> Dict[str, int]:
        """Analyze docstring usage."""
        import re
        
        triple_quote_docs = len(re.findall(r'""".*?"""', content, re.DOTALL))
        single_quote_docs = len(re.findall(r"'''.*?'''", content, re.DOTALL))
        
        return {
            "triple_quote_docstrings": triple_quote_docs,
            "single_quote_docstrings": single_quote_docs,
            "total_docstrings": triple_quote_docs + single_quote_docs
        }
    
    def _detect_frameworks(self, content: str) -> List[str]:
        """Detect Python frameworks in use."""
        frameworks = []
        patterns = self.get_language_patterns()["framework_indicators"]
        
        for framework, indicators in patterns.items():
            if any(indicator in content for indicator in indicators):
                frameworks.append(framework)
        
        return frameworks


class JavaScriptAnalyzer(LanguageSpecificAnalyzer):
    """JavaScript-specific code analyzer."""
    
    def analyze(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze JavaScript code for language-specific patterns."""
        analysis = {
            "modules": self._analyze_modules(content),
            "arrow_functions": self._analyze_arrow_functions(content),
            "promises": self._analyze_promises(content),
            "classes": self._analyze_es6_classes(content),
            "destructuring": self._analyze_destructuring(content),
            "template_literals": self._analyze_template_literals(content),
            "frameworks": self._detect_frameworks(content)
        }
        return analysis
    
    def get_language_patterns(self) -> Dict[str, Any]:
        """Get JavaScript-specific patterns."""
        return {
            "module_patterns": [
                r"import\s+.+\s+from\s+['\"](.+)['\"]",
                r"require\s*\(\s*['\"](.+)['\"]\s*\)",
                r"export\s+(?:default\s+)?(.+)"
            ],
            "arrow_function_patterns": [
                r"=>\s*{",
                r"=>\s*\w+"
            ],
            "framework_indicators": {
                "React": ["React", "useState", "useEffect", "jsx"],
                "Vue": ["Vue", "v-if", "v-for", "@click"],
                "Angular": ["@Component", "@Injectable", "ngOnInit"],
                "Express": ["express", "app.get", "app.post"],
                "Node.js": ["require", "module.exports", "process."]
            }
        }
    
    def _analyze_modules(self, content: str) -> Dict[str, List]:
        """Analyze JavaScript module usage."""
        import re
        
        imports = re.findall(r"import\s+.+\s+from\s+['\"](.+)['\"]", content)
        requires = re.findall(r"require\s*\(\s*['\"](.+)['\"]\s*\)", content)
        exports = re.findall(r"export\s+(?:default\s+)?(.+)", content)
        
        return {
            "es6_imports": imports,
            "commonjs_requires": requires,
            "exports": exports
        }
    
    def _analyze_arrow_functions(self, content: str) -> int:
        """Count arrow function usage."""
        import re
        return len(re.findall(r"=>\s*{|=>\s*\w+", content))
    
    def _analyze_promises(self, content: str) -> Dict[str, bool]:
        """Analyze Promise and async/await usage."""
        return {
            "uses_promises": "Promise" in content,
            "uses_async_await": "async" in content and "await" in content,
            "uses_then_catch": ".then(" in content or ".catch(" in content
        }
    
    def _analyze_es6_classes(self, content: str) -> int:
        """Count ES6 class definitions."""
        import re
        return len(re.findall(r"class\s+\w+", content))
    
    def _analyze_destructuring(self, content: str) -> Dict[str, int]:
        """Analyze destructuring patterns."""
        import re
        
        array_destructuring = len(re.findall(r"\[\s*\w+", content))
        object_destructuring = len(re.findall(r"{\s*\w+", content))
        
        return {
            "array_destructuring": array_destructuring,
            "object_destructuring": object_destructuring
        }
    
    def _analyze_template_literals(self, content: str) -> int:
        """Count template literal usage."""
        import re
        return len(re.findall(r"`[^`]*\$\{[^}]*\}[^`]*`", content))
    
    def _detect_frameworks(self, content: str) -> List[str]:
        """Detect JavaScript frameworks in use."""
        frameworks = []
        patterns = self.get_language_patterns()["framework_indicators"]
        
        for framework, indicators in patterns.items():
            if any(indicator in content for indicator in indicators):
                frameworks.append(framework)
        
        return frameworks


class MultiLanguageAnalyzer:
    """
    Multi-language code analyzer that coordinates language-specific analysis.
    
    This analyzer detects the programming language and delegates to
    appropriate language-specific analyzers for detailed analysis.
    """
    
    def __init__(self):
        self.detector = LanguageDetector()
        self.analyzers = self._initialize_analyzers()
        self.parser_factory = self._initialize_parser_factory()
    
    def analyze_file(
        self, 
        file_path: str, 
        content: str,
        repo_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multi-language analysis of a file.
        
        Args:
            file_path: Path to the file
            content: File content
            repo_id: Repository identifier
            
        Returns:
            Comprehensive analysis results
        """
        # Detect language
        language_info = self.detector.detect_language(file_path, content)
        
        # Get basic parsing
        parser = self._get_parser_for_language(
            language_info.language, file_path, repo_id, content
        )
        
        parse_result = parser.parse() if parser else None
        
        # Get language-specific analysis
        language_analysis = self._get_language_specific_analysis(
            language_info.language, content, file_path
        )
        
        # Get language features
        language_features = self.detector.get_language_features(language_info.language)
        
        return {
            "language_info": {
                "language": language_info.language.value,
                "confidence": language_info.confidence,
                "detection_method": language_info.detection_method,
                "file_extension": language_info.file_extension,
                "syntax_indicators": language_info.syntax_indicators or []
            },
            "language_features": language_features,
            "parse_result": {
                "nodes": parse_result.nodes if parse_result else [],
                "edges": parse_result.edges if parse_result else [],
                "imports": parse_result.imports if parse_result else {},
                "calls": parse_result.calls if parse_result else []
            },
            "language_analysis": language_analysis,
            "metadata": {
                "file_path": file_path,
                "repo_id": repo_id,
                "analyzer": "MultiLanguageAnalyzer"
            }
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [lang.value for lang in SupportedLanguage if lang != SupportedLanguage.UNKNOWN]
    
    def get_language_capabilities(self, language: str) -> Dict[str, Any]:
        """Get capabilities for a specific language."""
        try:
            lang_enum = SupportedLanguage(language)
            features = self.detector.get_language_features(lang_enum)
            has_analyzer = lang_enum in self.analyzers
            has_parser = self._can_parse_language(lang_enum)
            
            return {
                "language": language,
                "features": features,
                "has_specialized_analyzer": has_analyzer,
                "has_parser_support": has_parser,
                "analysis_depth": "full" if has_analyzer else "basic"
            }
        except ValueError:
            return {"error": f"Unsupported language: {language}"}
    
    def _initialize_analyzers(self) -> Dict[SupportedLanguage, LanguageSpecificAnalyzer]:
        """Initialize language-specific analyzers."""
        return {
            SupportedLanguage.PYTHON: PythonAnalyzer(),
            SupportedLanguage.JAVASCRIPT: JavaScriptAnalyzer(),
            # TODO: Add more language analyzers
            # SupportedLanguage.TYPESCRIPT: TypeScriptAnalyzer(),
            # SupportedLanguage.JAVA: JavaAnalyzer(),
            # SupportedLanguage.GO: GoAnalyzer(),
        }
    
    def _initialize_parser_factory(self) -> Dict[SupportedLanguage, Type[BaseParser]]:
        """Initialize parser factory for different languages."""
        return {
            SupportedLanguage.PYTHON: HybridParser,  # Use hybrid for Python
            # For JavaScript, we'll use our simple parser directly
            # SupportedLanguage.JAVASCRIPT: AdvancedParser,  # Use tree-sitter for JS
            # SupportedLanguage.TYPESCRIPT: AdvancedParser,  # Use tree-sitter for TS
            # SupportedLanguage.JAVA: AdvancedParser,  # Use tree-sitter for Java
            # SupportedLanguage.GO: AdvancedParser,  # Use tree-sitter for Go
            # Add more as needed
        }
    
    def _get_parser_for_language(
        self, 
        language: SupportedLanguage, 
        file_path: str, 
        repo_id: str, 
        content: str
    ) -> Optional[BaseParser]:
        """Get appropriate parser for the detected language."""
        parser_class = self.parser_factory.get(language)
        
        if parser_class:
            try:
                return parser_class(file_path, repo_id, content)
            except Exception as e:
                logger.warning(f"Failed to create parser for {language.value}: {e}")
                # For JavaScript, try our simple parser if tree-sitter fails
                if language == SupportedLanguage.JAVASCRIPT:
                    return self._create_simple_js_parser(file_path, repo_id, content)
                # Fallback to AST parser for Python
                elif language == SupportedLanguage.PYTHON:
                    return ASTParser(file_path, repo_id, content)
        
        # For JavaScript without tree-sitter, use our simple parser
        if language == SupportedLanguage.JAVASCRIPT:
            return self._create_simple_js_parser(file_path, repo_id, content)
        
        return None
    
    def _create_simple_js_parser(self, file_path: str, repo_id: str, content: str):
        """Create a simple JavaScript parser that extracts basic entities."""
        from ..syntax_engine.base_parser import BaseParser, ParseResult
        import re
        import uuid
        from datetime import datetime
        
        class SimpleJSParser(BaseParser):
            def supports_language(self, language: str) -> bool:
                """Check if this parser supports the given language."""
                return language.lower() in ['javascript', 'js']
            
            def parse(self) -> ParseResult:
                nodes = []
                
                # Extract classes
                class_pattern = re.compile(r'class\s+(\w+)\s*(?:extends\s+\w+)?\s*\{', re.MULTILINE)
                for match in class_pattern.finditer(self.content):
                    class_name = match.group(1)
                    line_num = self.content[:match.start()].count('\n') + 1
                    
                    node = GraphNode(
                        uuid=str(uuid.uuid4()),
                        repo_id=self.repo_id,
                        name=class_name,
                        type="class",
                        file_path=self.file_path,
                        start_line=line_num,
                        end_line=line_num,
                        tags=["JAVASCRIPT", "CLASS"],
                        metadata={"language": "javascript"},
                        created_by="SimpleJSParser",
                        created_at=datetime.now()
                    )
                    nodes.append(node)
                
                # Extract functions and methods
                func_patterns = [
                    (r'function\s+(\w+)\s*\(', 'function'),
                    (r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{', 'method'),
                    (r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>', 'arrow_function')
                ]
                
                for pattern, func_type in func_patterns:
                    func_pattern = re.compile(pattern, re.MULTILINE)
                    for match in func_pattern.finditer(self.content):
                        func_name = match.group(1)
                        line_num = self.content[:match.start()].count('\n') + 1
                        
                        # Skip some common non-function patterns
                        if func_name in ['if', 'for', 'while', 'switch', 'catch']:
                            continue
                        
                        node = GraphNode(
                            uuid=str(uuid.uuid4()),
                            repo_id=self.repo_id,
                            name=func_name,
                            type="function",
                            file_path=self.file_path,
                            start_line=line_num,
                            end_line=line_num,
                            tags=["JAVASCRIPT", "FUNCTION", func_type.upper()],
                            metadata={"language": "javascript", "function_type": func_type},
                            created_by="SimpleJSParser",
                            created_at=datetime.now()
                        )
                        nodes.append(node)
                
                return ParseResult(nodes=nodes, edges=[], imports={}, calls=[])
        
        return SimpleJSParser(file_path, repo_id, content)
    
    def _get_language_specific_analysis(
        self, 
        language: SupportedLanguage, 
        content: str, 
        file_path: str
    ) -> Dict[str, Any]:
        """Get language-specific analysis if available."""
        analyzer = self.analyzers.get(language)
        
        if analyzer:
            try:
                return analyzer.analyze(content, file_path)
            except Exception as e:
                logger.error(f"Language-specific analysis failed for {language.value}: {e}")
                return {"error": str(e)}
        
        return {"message": f"No specialized analyzer for {language.value}"}
    
    def _can_parse_language(self, language: SupportedLanguage) -> bool:
        """Check if we can parse the given language."""
        return language in self.parser_factory 