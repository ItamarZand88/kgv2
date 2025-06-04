"""
Tree-sitter Integration Utilities

This module provides utilities for working with tree-sitter parsers
and language detection, inspired by potpie's approach.
"""

# Tree-sitter imports (potpie-style)
try:
    import tree_sitter_languages as ts_languages
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
    
    def get_language(lang: str):
        """Get tree-sitter language"""
        language_map = {
            'python': ts_languages.get_language('python'),
            'javascript': ts_languages.get_language('javascript'),
            'typescript': ts_languages.get_language('typescript'),
            'java': ts_languages.get_language('java'),
            'c': ts_languages.get_language('c'),
            'cpp': ts_languages.get_language('cpp'),
            'go': ts_languages.get_language('go'),
            'rust': ts_languages.get_language('rust'),
        }
        return language_map.get(lang)
    
    def get_parser(lang: str):
        """Get tree-sitter parser"""
        parser = Parser()
        language = get_language(lang)
        if language:
            parser.set_language(language)
            return parser
        return None
    
    def filename_to_lang(filename: str) -> str:
        """Convert filename to language"""
        if filename.endswith('.py'):
            return 'python'
        elif filename.endswith(('.js', '.jsx')):
            return 'javascript'
        elif filename.endswith(('.ts', '.tsx')):
            return 'typescript'
        elif filename.endswith('.java'):
            return 'java'
        elif filename.endswith('.c'):
            return 'c'
        elif filename.endswith(('.cpp', '.cxx', '.cc')):
            return 'cpp'
        elif filename.endswith('.go'):
            return 'go'
        elif filename.endswith('.rs'):
            return 'rust'
        return None
        
except ImportError:
    print("Tree-sitter not available, falling back to AST parsing")
    TREE_SITTER_AVAILABLE = False
    
    # Fallback implementations
    def get_language(lang: str):
        return None
    
    def get_parser(lang: str):
        return None
    
    def filename_to_lang(filename: str) -> str:
        return None 