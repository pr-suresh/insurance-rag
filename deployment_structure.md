# 📁 Deployment-Ready Project Structure

## Current State Analysis
- **Total Files**: 39 (excluding .venv and .git)
- **Main Components**: Core RAG system, UI apps, testing suite, evaluation tools

## 🎯 Proposed Clean Folder Structure

```
insurance-rag/
├── src/                          # Core application code
│   ├── core/                     # Core RAG functionality
│   │   ├── __init__.py
│   │   ├── enhanced_rag.py       # Main RAG implementation
│   │   └── config.py             # Configuration settings
│   │
│   ├── agents/                   # Agent implementations
│   │   ├── __init__.py
│   │   ├── adjuster_agent.py     # Production adjuster agent
│   │   └── debug_agent.py        # Debug-enabled agent
│   │
│   └── ui/                       # User interfaces
│       ├── __init__.py
│       ├── streamlit_app.py      # Main Streamlit UI
│       ├── gradio_app.py         # Gradio alternative
│       └── flask_app.py          # Flask API
│
├── tests/                        # All test files
│   ├── __init__.py
│   ├── test_hybrid_search.py
│   ├── test_langsmith.py
│   ├── test_state_detection.py
│   └── test_ragas_single_hop.py
│
├── evaluation/                   # Evaluation and benchmarking
│   ├── __init__.py
│   ├── ragas_evaluation.py
│   ├── create_ragas_dataset.py
│   └── evaluate_with_dataset.py
│
├── scripts/                      # Utility scripts
│   ├── automated_test_runner.py
│   ├── langsmith_setup.py
│   └── analyze_retrieval.py
│
├── data/                         # Data files
│   └── insurance_claims.csv
│
├── docs/                         # Documentation
│   ├── README.md
│   ├── DEPLOYMENT.md
│   ├── API.md
│   └── results/
│       ├── RAGAS_RESULTS.md
│       └── RETRIEVAL_COMPARISON.md
│
├── config/                       # Configuration files
│   ├── .env.example
│   └── settings.yaml
│
├── docker/                       # Docker files
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .gitignore
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── pyproject.toml
├── README.md                     # Main readme
└── run.py                        # Main entry point
```

## 🗑️ Files to Remove/Archive

### Delete (Redundant/Old):
- `rag.py` - Old version
- `test_claim_search.py` - Redundant test
- `demo_*.py` - Demo files
- `simple_ragas_eval.py` - Simplified version
- `quick_search_test.py` - Quick test
- `interactive_search_test.py` - Interactive test
- Old CSV results files

### Archive (Keep for reference):
- Session JSON files → `archive/sessions/`
- Old evaluation CSVs → `archive/evaluations/`
- Debug logs → `archive/logs/`

## 📋 File Mapping

### Core Files → `src/core/`
- `enhanced_rag.py` → `src/core/enhanced_rag.py`

### Agents → `src/agents/`
- `adjuster_agent.py` → `src/agents/adjuster_agent.py`
- `debug_adjuster_agent.py` → `src/agents/debug_agent.py`

### UI → `src/ui/`
- `streamlit_app.py` → `src/ui/streamlit_app.py`
- `gradio_app.py` → `src/ui/gradio_app.py`
- `flask_app.py` → `src/ui/flask_app.py`

### Tests → `tests/`
- `test_hybrid_search.py` → `tests/test_hybrid_search.py`
- `test_langsmith.py` → `tests/test_langsmith.py`
- `test_state_detection.py` → `tests/test_state_detection.py`
- `test_ragas_single_hop.py` → `tests/test_ragas_single_hop.py`

### Evaluation → `evaluation/`
- `ragas_evaluation.py` → `evaluation/ragas_evaluation.py`
- `create_ragas_dataset.py` → `evaluation/create_ragas_dataset.py`
- `evaluate_with_ragas_dataset.py` → `evaluation/evaluate_with_dataset.py`

### Scripts → `scripts/`
- `automated_test_runner.py` → `scripts/automated_test_runner.py`
- `langsmith_setup.py` → `scripts/langsmith_setup.py`
- `analyze_retrieval_failure.py` → `scripts/analyze_retrieval.py`

## 🚀 Deployment Benefits

1. **Clear Separation**: Core, UI, tests, and evaluation separated
2. **Import Friendly**: Proper package structure with `__init__.py`
3. **Docker Ready**: Easy to containerize with clear structure
4. **CI/CD Compatible**: Tests isolated for automation
5. **Documentation**: Centralized docs folder
6. **Configuration**: Separate config folder for settings
7. **Entry Point**: Single `run.py` for starting the application

## 🔧 Implementation Steps

1. Create folder structure
2. Move files to appropriate locations
3. Update all imports in Python files
4. Create `__init__.py` files for packages
5. Update paths in code (data loading, etc.)
6. Create proper requirements.txt
7. Test all functionality after reorganization
8. Create deployment documentation