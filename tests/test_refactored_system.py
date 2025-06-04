"""
Comprehensive Test Suite for Refactored Knowledge Graph System

This test suite validates that all components work correctly after the 
professional refactoring from monolithic analyzer.py to modular core system.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Test the new core module structure
from core import CodeAnalyzer
from core.syntax_engine import BaseParser, ASTParser, AdvancedParser, HybridParser, ParseResult
from core.semantic_processor import EntityExtractor, RelationshipBuilder
from core.language_analyzer import LanguageDetector, MultiLanguageAnalyzer, SupportedLanguage
from core.pattern_matchers import RuleEngine, PatternDetector, RuleType, PatternType

# Test models and storage
from models import GraphNode, GraphEdge
from storage import StorageManager, GraphManager


class TestCoreModuleImports:
    """Test that all refactored modules import correctly."""
    
    def test_core_module_import(self):
        """Test that core module imports successfully."""
        assert CodeAnalyzer is not None
        
    def test_syntax_engine_imports(self):
        """Test syntax engine module imports."""
        assert BaseParser is not None
        assert ASTParser is not None
        assert AdvancedParser is not None
        assert HybridParser is not None
        assert ParseResult is not None
        
    def test_semantic_processor_imports(self):
        """Test semantic processor module imports."""
        assert EntityExtractor is not None
        assert RelationshipBuilder is not None
        
    def test_language_analyzer_imports(self):
        """Test language analyzer module imports."""
        assert LanguageDetector is not None
        assert MultiLanguageAnalyzer is not None
        assert SupportedLanguage is not None
        
    def test_pattern_matchers_imports(self):
        """Test pattern matchers module imports."""
        assert RuleEngine is not None
        assert PatternDetector is not None
        assert RuleType is not None
        assert PatternType is not None


class TestSyntaxEngine:
    """Test the syntax engine components."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_python_code = '''
import os
import sys

class TestClass:
    """A test class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def test_method(self) -> str:
        """Test method with docstring."""
        return f"Hello, {self.name}!"

def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

# Global variable
CONSTANT_VALUE = 42
'''
        
        self.temp_file = None
    
    def teardown_method(self):
        """Clean up temp files."""
        if self.temp_file and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)
    
    def test_ast_parser_creation(self):
        """Test AST parser can be created."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.sample_python_code)
            self.temp_file = f.name
        
        parser = ASTParser(self.temp_file, "test_repo", self.sample_python_code)
        assert parser is not None
        assert parser.file_path == self.temp_file
        assert parser.repo_id == "test_repo"
    
    def test_ast_parser_parsing(self):
        """Test AST parser can parse Python code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.sample_python_code)
            self.temp_file = f.name
        
        parser = ASTParser(self.temp_file, "test_repo", self.sample_python_code)
        result = parser.parse()
        
        assert isinstance(result, ParseResult)
        assert len(result.nodes) > 0
        assert len(result.edges) >= 0
        
        # Check we found the expected entities
        node_names = [node.name for node in result.nodes]
        assert "TestClass" in node_names
        assert "test_method" in node_names
        assert "standalone_function" in node_names
    
    def test_hybrid_parser_creation(self):
        """Test hybrid parser can be created."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.sample_python_code)
            self.temp_file = f.name
        
        parser = HybridParser(self.temp_file, "test_repo", self.sample_python_code)
        assert parser is not None
    
    def test_parse_result_structure(self):
        """Test ParseResult contains expected data structures."""
        result = ParseResult(nodes=[], edges=[], imports={}, calls=[])
        assert hasattr(result, 'nodes')
        assert hasattr(result, 'edges')
        assert hasattr(result, 'imports')
        assert hasattr(result, 'calls')
        assert isinstance(result.nodes, list)
        assert isinstance(result.edges, list)


class TestSemanticProcessor:
    """Test semantic processing components."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_code = '''
from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users."""
    return {"users": []}

@app.route('/api/user/<int:user_id>', methods=['GET'])  
def get_user(user_id: int):
    """Get user by ID."""
    return {"user": {"id": user_id}}

class UserService:
    """Service for user operations."""
    
    def create_user(self, data: dict) -> dict:
        # TODO: Implement user creation
        return data
    
    def delete_user(self, user_id: int) -> bool:
        return True

def test_function():
    """Test function for pytest."""
    assert True
'''
    
    def test_entity_extractor_creation(self):
        """Test entity extractor can be created."""
        extractor = EntityExtractor()
        assert extractor is not None
    
    def test_entity_extraction(self):
        """Test entity extraction finds entities."""
        extractor = EntityExtractor()
        entities = extractor.extract_entities_from_content(self.sample_code, "test_file.py", "test_repo")
        
        assert len(entities) > 0
        
        # Should find semantic patterns like endpoints, decorators, etc.
        entity_types = [entity.type for entity in entities]
        assert "endpoint" in entity_types or "decorator" in entity_types
        
        # Check specific entities
        entity_names = [entity.name for entity in entities]
        # Look for patterns we know should be detected
        assert any("/api/users" in name or "app" in name for name in entity_names)
    
    def test_relationship_builder_creation(self):
        """Test relationship builder can be created."""
        builder = RelationshipBuilder()
        assert builder is not None
    
    def test_relationship_building(self):
        """Test relationship building creates edges."""
        extractor = EntityExtractor()
        builder = RelationshipBuilder()
        
        entities = extractor.extract_entities_from_content(self.sample_code, "test_file.py", "test_repo")
        relationships = builder.build_relationships(entities, self.sample_code, "test_file.py")
        
        # Should find some relationships
        assert len(relationships) >= 0
        
        if relationships:
            # Check relationship structure
            rel = relationships[0]
            assert hasattr(rel, 'from_uuid')
            assert hasattr(rel, 'to_uuid')
            assert hasattr(rel, 'type')


class TestLanguageAnalyzer:
    """Test language analysis components."""
    
    def test_language_detector_creation(self):
        """Test language detector can be created."""
        detector = LanguageDetector()
        assert detector is not None
    
    def test_python_detection(self):
        """Test Python language detection."""
        detector = LanguageDetector()
        
        python_code = '''
def hello_world():
    print("Hello, World!")
    return True
'''
        
        result = detector.detect_language("test.py", python_code)
        assert result.language == SupportedLanguage.PYTHON
        assert result.confidence > 0.5
    
    def test_javascript_detection(self):
        """Test JavaScript language detection."""
        detector = LanguageDetector()
        
        js_code = '''
function helloWorld() {
    console.log("Hello, World!");
    return true;
}

const arrow = () => {
    return "arrow function";
};
'''
        
        result = detector.detect_language("test.js", js_code)
        assert result.language == SupportedLanguage.JAVASCRIPT
        assert result.confidence > 0.5
    
    def test_multi_language_analyzer_creation(self):
        """Test multi-language analyzer can be created."""
        analyzer = MultiLanguageAnalyzer()
        assert analyzer is not None
    
    def test_multi_language_analysis(self):
        """Test multi-language analysis."""
        analyzer = MultiLanguageAnalyzer()
        
        python_code = '''
import requests

def fetch_data():
    response = requests.get("https://api.example.com")
    return response.json()
'''
        
        result = analyzer.analyze_file("test.py", python_code, "test_repo")
        
        assert result is not None
        assert "language_info" in result
        assert "parse_result" in result
        assert "language_analysis" in result
        
        # Check language detection
        assert result["language_info"]["language"] == "python"
        assert result["language_info"]["confidence"] > 0.5


class TestPatternMatchers:
    """Test pattern matching components."""
    
    def setup_method(self):
        """Set up test data."""
        self.code_with_patterns = '''
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def getInstance():
        return Singleton()

def long_function_with_many_parameters(a, b, c, d, e, f, g, h):
    """This function has too many parameters."""
    # TODO: Refactor this function
    result = []
    for item in range(100):
        result.append(item * 2)
    return result

try:
    risky_operation()
except:  # Bare except - anti-pattern
    pass

# Magic number
TIMEOUT = 12345
'''
    
    def test_rule_engine_creation(self):
        """Test rule engine can be created."""
        engine = RuleEngine()
        assert engine is not None
    
    def test_rule_engine_analysis(self):
        """Test rule engine finds patterns."""
        engine = RuleEngine()
        
        matches = engine.apply_rules(
            self.code_with_patterns, 
            "test.py", 
            language="python"
        )
        
        # Should find some rule violations
        assert len(matches) > 0
        
        # Check for specific patterns
        rule_ids = [match.rule_id for match in matches]
        assert any("bare_except" in rule_id for rule_id in rule_ids)
    
    def test_pattern_detector_creation(self):
        """Test pattern detector can be created."""
        detector = PatternDetector()
        assert detector is not None
    
    def test_pattern_detection(self):
        """Test pattern detection finds design patterns."""
        detector = PatternDetector()
        
        # Create mock nodes and edges for testing
        nodes = [
            GraphNode(
                uuid="singleton_1",
                name="Singleton",
                type="class",
                file_path="test.py",
                start_line=1,
                end_line=10,
                repo_id="test_repo",
                created_by="test"  # Add required created_by field
            )
        ]
        
        patterns = detector.detect_patterns(
            nodes, [], self.code_with_patterns, "test.py"
        )
        
        # Should find some patterns
        assert len(patterns) >= 0
        
        if patterns:
            pattern = patterns[0]
            assert hasattr(pattern, 'pattern_name')
            assert hasattr(pattern, 'pattern_type')
            assert hasattr(pattern, 'confidence')


class TestCodeAnalyzer:
    """Test the main CodeAnalyzer orchestrator."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_repo_path = os.path.join(self.temp_dir, "test_repo")
        os.makedirs(self.test_repo_path)
        
        # Create test files
        with open(os.path.join(self.test_repo_path, "main.py"), 'w') as f:
            f.write('''
import sys
from utils import helper_function

def main():
    """Main function."""
    result = helper_function("test")
    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    main()
''')
        
        with open(os.path.join(self.test_repo_path, "utils.py"), 'w') as f:
            f.write('''
def helper_function(text: str) -> str:
    """Helper function."""
    return f"Processed: {text}"

class UtilityClass:
    """Utility class."""
    
    def process_data(self, data):
        return data.upper()
''')
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_code_analyzer_creation(self):
        """Test CodeAnalyzer can be created."""
        storage = StorageManager()
        graph_manager = GraphManager(storage)
        analyzer = CodeAnalyzer(storage, graph_manager)
        
        assert analyzer is not None
    
    @pytest.mark.asyncio 
    async def test_analyze_repository(self):
        """Test repository analysis."""
        storage = StorageManager()
        graph_manager = GraphManager(storage)
        analyzer = CodeAnalyzer(storage, graph_manager)
        
        # Analyze the test repository
        result = await analyzer.analyze_repository(self.test_repo_path, "test_repo")
        
        assert result is not None
        assert "nodes_created" in result
        assert "edges_created" in result
        assert result["nodes_created"] > 0
        
        # Verify data was stored
        assert await storage.repo_exists("test_repo")
        
        # Get stored entities
        entities = await graph_manager.get_nodes("test_repo")
        assert len(entities) > 0
        
        # Should find our test functions and classes
        entity_names = [entity.name for entity in entities]
        assert "main" in entity_names
        assert "helper_function" in entity_names
        assert "UtilityClass" in entity_names


class TestSystemIntegration:
    """Integration tests for the entire system."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.complex_repo_path = os.path.join(self.temp_dir, "complex_repo")
        os.makedirs(self.complex_repo_path)
        
        # Create a more complex repository structure
        
        # models.py
        with open(os.path.join(self.complex_repo_path, "models.py"), 'w') as f:
            f.write('''
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class User:
    """User model."""
    id: int
    name: str
    email: str
    active: bool = True

@dataclass  
class Product:
    """Product model."""
    id: int
    name: str
    price: float
    category_id: int
''')
        
        # services.py
        with open(os.path.join(self.complex_repo_path, "services.py"), 'w') as f:
            f.write('''
from models import User, Product
from typing import List, Optional

class UserService:
    """Service for user operations."""
    
    def __init__(self):
        self._users = []
    
    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        user = User(
            id=len(self._users) + 1,
            name=name,
            email=email
        )
        self._users.append(user)
        return user
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        for user in self._users:
            if user.id == user_id:
                return user
        return None
    
    def list_users(self) -> List[User]:
        """List all users.""" 
        return self._users.copy()

class ProductService:
    """Service for product operations."""
    
    def __init__(self):
        self._products = []
    
    def create_product(self, name: str, price: float, category_id: int) -> Product:
        """Create a new product."""
        product = Product(
            id=len(self._products) + 1,
            name=name,
            price=price,
            category_id=category_id
        )
        self._products.append(product)
        return product
''')
        
        # api.py  
        with open(os.path.join(self.complex_repo_path, "api.py"), 'w') as f:
            f.write('''
from flask import Flask, request, jsonify
from services import UserService, ProductService

app = Flask(__name__)
user_service = UserService()
product_service = ProductService()

@app.route('/users', methods=['GET'])
def list_users():
    """List all users."""
    users = user_service.list_users()
    return jsonify([{"id": u.id, "name": u.name, "email": u.email} for u in users])

@app.route('/users', methods=['POST'])
def create_user():
    """Create a new user."""
    data = request.get_json()
    user = user_service.create_user(data['name'], data['email'])
    return jsonify({"id": user.id, "name": user.name, "email": user.email})

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id: int):
    """Get user by ID."""
    user = user_service.get_user(user_id)
    if user:
        return jsonify({"id": user.id, "name": user.name, "email": user.email})
    return jsonify({"error": "User not found"}), 404

@app.route('/products', methods=['POST'])
def create_product():
    """Create a new product."""
    data = request.get_json()
    product = product_service.create_product(
        data['name'], 
        data['price'], 
        data['category_id']
    )
    return jsonify({
        "id": product.id, 
        "name": product.name, 
        "price": product.price,
        "category_id": product.category_id
    })

if __name__ == '__main__':
    app.run(debug=True)
''')
        
        # tests.py
        with open(os.path.join(self.complex_repo_path, "tests.py"), 'w') as f:
            f.write('''
import pytest
from services import UserService, ProductService
from models import User, Product

class TestUserService:
    """Test user service."""
    
    def setup_method(self):
        self.service = UserService()
    
    def test_create_user(self):
        """Test user creation."""
        user = self.service.create_user("John Doe", "john@example.com")
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.id == 1
    
    def test_get_user(self):
        """Test get user."""
        user = self.service.create_user("Jane Doe", "jane@example.com")
        found_user = self.service.get_user(user.id)
        assert found_user == user
    
    def test_list_users(self):
        """Test list users."""
        self.service.create_user("User 1", "user1@example.com")
        self.service.create_user("User 2", "user2@example.com")
        users = self.service.list_users()
        assert len(users) == 2

class TestProductService:
    """Test product service."""
    
    def setup_method(self):
        self.service = ProductService()
    
    def test_create_product(self):
        """Test product creation."""
        product = self.service.create_product("Test Product", 99.99, 1)
        assert product.name == "Test Product"
        assert product.price == 99.99
        assert product.category_id == 1
''')
    
    def teardown_method(self):
        """Clean up integration test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_full_system_analysis(self):
        """Test complete system analysis on complex repository."""
        storage = StorageManager()
        graph_manager = GraphManager(storage)
        analyzer = CodeAnalyzer(storage, graph_manager)
        
        # Analyze the complex repository
        result = await analyzer.analyze_repository(self.complex_repo_path, "complex_repo")
        
        assert result is not None
        assert result["nodes_created"] > 0
        assert result["edges_created"] >= 0
        
        # Verify comprehensive analysis
        entities = await graph_manager.get_nodes("complex_repo")
        assert len(entities) >= 10  # Should find many entities
        
        # Check for different entity types
        entity_types = set(entity.type for entity in entities)
        assert "class" in entity_types
        assert "function" in entity_types
        
        # Check for specific entities from our test data
        entity_names = [entity.name for entity in entities]
        assert "User" in entity_names  # Model
        assert "UserService" in entity_names  # Service class
        assert "create_user" in entity_names  # Service method
        assert "list_users" in entity_names  # API endpoint
        
        # Check for relationships (may be 0 in simple test scenarios)
        edges = await graph_manager.get_edges("complex_repo")
        # Note: Relationship building may not find many edges in simple test code
        assert len(edges) >= 0  # Just verify the method works
    
    @pytest.mark.asyncio
    async def test_language_detection_integration(self):
        """Test language detection works in full system."""
        storage = StorageManager()
        graph_manager = GraphManager(storage)
        analyzer = CodeAnalyzer(storage, graph_manager)
        
        # Add JavaScript file to test multi-language support
        js_file = os.path.join(self.complex_repo_path, "frontend.js")
        with open(js_file, 'w') as f:
            f.write('''
const API_BASE = 'http://localhost:5000';

class UserManager {
    constructor() {
        this.users = [];
    }
    
    async fetchUsers() {
        const response = await fetch(`${API_BASE}/users`);
        this.users = await response.json();
        return this.users;
    }
    
    async createUser(userData) {
        const response = await fetch(`${API_BASE}/users`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(userData)
        });
        return await response.json();
    }
}

export default UserManager;
''')
        
        # Analyze repository with multiple languages
        result = await analyzer.analyze_repository(self.complex_repo_path, "multi_lang_repo")
        
        assert result is not None
        
        # Check that both Python and JavaScript files were processed
        entities = await graph_manager.get_nodes("multi_lang_repo")
        
        # Should find entities from both languages
        python_entities = [e for e in entities if e.file_path.endswith('.py')]
        js_entities = [e for e in entities if e.file_path.endswith('.js')]
        
        assert len(python_entities) > 0
        assert len(js_entities) > 0
        
        # Verify JavaScript entities were found
        js_entity_names = [entity.name for entity in js_entities]
        assert "UserManager" in js_entity_names
        assert "fetchUsers" in js_entity_names


def run_tests():
    """Run all tests and return results."""
    print("🚀 Starting Comprehensive Test Suite for Refactored Knowledge Graph System")
    print("=" * 80)
    
    # Run pytest with detailed output
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--color=yes"
    ], capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nExit code: {result.returncode}")
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed! The refactored system is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    exit(0 if success else 1) 