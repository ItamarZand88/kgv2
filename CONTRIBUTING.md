# Contributing to Knowledge Graph V2

Thank you for your interest in contributing to Knowledge Graph V2! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Clone the repository:
   ```
   git clone https://github.com/ItamarZand88/kgv2.git
   cd kgv2
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. Run tests to ensure everything is working:
   ```
   pytest
   ```

## Project Structure

- `core/` - Core analysis engine
  - `language_analyzer/` - Language detection and analysis
  - `pattern_matchers/` - Pattern matching and rule engines
  - `semantic_processor/` - Semantic analysis and processing
  - `syntax_engine/` - Syntax parsing and analysis
- `data/` - Data storage directory
- `queries/` - Query templates and patterns
- `src/` - Source code
- `tests/` - Test cases
- `utils/` - Utility functions

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Coding Standards

- Follow PEP 8 guidelines for Python code
- Include docstrings for all functions, classes, and modules
- Write tests for new functionality
- Keep functions small and focused on a single responsibility