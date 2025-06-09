# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses UV for dependency management and packaging:

- **Install dependencies**: `uv sync` (installs from uv.lock)
- **Run tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_ner.py::test_function_name`
- **Lint code**: `uv run ruff check`
- **Format code**: `uv run ruff format`
- **Build package**: `uv build`

## Architecture Overview

Encite is a Python library for Named Entity Recognition (NER) using Anthropic Claude's citations feature. The core architecture consists of:

### Text Processing Pipeline (encite/ner.py)
1. **Text Chunking**: `_preprocess_text()` splits input text into chunks based on separators (" ", ".", ",", "\n", "\t"), tracking character positions for accurate entity location mapping
2. **Model Input Formatting**: `_format_model_input()` creates input to the Claude API with citations enabled
3. **Entity Extraction**: `_parse_model_output()` parses Claude's responses with citations using regex patterns to extract `<entity_type>` and `<name>` tags
4. **Public API**: `find_entities()` orchestrates the pipeline

### Key Design Patterns
- **Citation-based NER**: Uses Claude's document citations to get precise character positions of entities in original text
- **Chunk-to-Character Mapping**: Maintains bidirectional mapping between text chunks and absolute character positions
- **Structured Output**: Returns `Entity` objects with type, name, and character boundaries

### Dependencies
- **langchain-anthropic**: Primary interface to Claude models (requires ANTHROPIC_API_KEY)
- **pytest**: Testing framework
- **ruff**: Linting and formatting

The library expects users to initialize their own `ChatAnthropic` model instance and pass it to `find_entities()` along with target entity types.