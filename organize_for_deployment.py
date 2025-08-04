#!/usr/bin/env python3
"""
Organize project structure for deployment
Run this script to reorganize files into a clean folder structure
"""

import os
import shutil
from pathlib import Path
import re

def create_folder_structure():
    """Create the deployment folder structure."""
    
    folders = [
        'src/core',
        'src/agents',
        'src/ui',
        'tests',
        'evaluation',
        'scripts',
        'docs/results',
        'config',
        'docker',
        'archive/sessions',
        'archive/evaluations',
        'archive/logs'
    ]
    
    print("📁 Creating folder structure...")
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {folder}")
    
    return folders

def move_files():
    """Move files to their appropriate locations."""
    
    file_mappings = {
        # Core files
        'enhanced_rag.py': 'src/core/enhanced_rag.py',
        
        # Agents
        'adjuster_agent.py': 'src/agents/adjuster_agent.py',
        'debug_adjuster_agent.py': 'src/agents/debug_agent.py',
        
        # UI
        'streamlit_app.py': 'src/ui/streamlit_app.py',
        'gradio_app.py': 'src/ui/gradio_app.py',
        'flask_app.py': 'src/ui/flask_app.py',
        
        # Tests
        'test_hybrid_search.py': 'tests/test_hybrid_search.py',
        'test_langsmith.py': 'tests/test_langsmith.py',
        'test_state_detection.py': 'tests/test_state_detection.py',
        'test_ragas_single_hop.py': 'tests/test_ragas_single_hop.py',
        
        # Evaluation
        'ragas_evaluation.py': 'evaluation/ragas_evaluation.py',
        'create_ragas_dataset.py': 'evaluation/create_ragas_dataset.py',
        'evaluate_with_ragas_dataset.py': 'evaluation/evaluate_with_dataset.py',
        
        # Scripts
        'automated_test_runner.py': 'scripts/automated_test_runner.py',
        'langsmith_setup.py': 'scripts/langsmith_setup.py',
        'analyze_retrieval_failure.py': 'scripts/analyze_retrieval.py',
        
        # Config
        '.env.example': 'config/.env.example',
        
        # Documentation
        'RAGAS_GENERATED_RESULTS.md': 'docs/results/RAGAS_RESULTS.md',
        'RETRIEVAL_COMPARISON_REPORT.md': 'docs/results/RETRIEVAL_COMPARISON.md',
    }
    
    print("\n📦 Moving files...")
    for src, dst in file_mappings.items():
        if os.path.exists(src):
            try:
                # Create parent directory if needed
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)
                print(f"  ✓ Moved {src} → {dst}")
            except Exception as e:
                print(f"  ✗ Failed to move {src}: {e}")
    
    # Archive session files
    print("\n📚 Archiving session files...")
    for file in Path('.').glob('debug_session_*.json'):
        shutil.move(str(file), f'archive/sessions/{file.name}')
        print(f"  ✓ Archived {file.name}")
    
    # Archive old results
    for file in Path('.').glob('*results*.csv'):
        if 'ragas_enhanced_results' not in str(file):
            shutil.move(str(file), f'archive/evaluations/{file.name}')
            print(f"  ✓ Archived {file.name}")
    
    # Archive logs
    for file in Path('.').glob('*.log'):
        shutil.move(str(file), f'archive/logs/{file.name}')
        print(f"  ✓ Archived {file.name}")

def create_init_files():
    """Create __init__.py files for Python packages."""
    
    init_locations = [
        'src/__init__.py',
        'src/core/__init__.py',
        'src/agents/__init__.py',
        'src/ui/__init__.py',
        'tests/__init__.py',
        'evaluation/__init__.py',
        'scripts/__init__.py'
    ]
    
    print("\n🐍 Creating __init__.py files...")
    for init_file in init_locations:
        Path(init_file).touch()
        print(f"  ✓ Created {init_file}")

def update_imports_in_file(filepath, import_mappings):
    """Update import statements in a Python file."""
    
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Update imports
    for old_import, new_import in import_mappings.items():
        content = re.sub(
            f'from {old_import} import',
            f'from {new_import} import',
            content
        )
        content = re.sub(
            f'import {old_import}',
            f'import {new_import}',
            content
        )
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    
    return False

def update_all_imports():
    """Update import statements in all Python files."""
    
    import_mappings = {
        'enhanced_rag': 'src.core.enhanced_rag',
        'adjuster_agent': 'src.agents.adjuster_agent',
        'debug_adjuster_agent': 'src.agents.debug_agent',
        'streamlit_app': 'src.ui.streamlit_app',
        'langsmith_setup': 'scripts.langsmith_setup',
    }
    
    print("\n🔧 Updating imports...")
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip venv and git
        if '.venv' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    updated_count = 0
    for filepath in python_files:
        if update_imports_in_file(filepath, import_mappings):
            print(f"  ✓ Updated imports in {filepath}")
            updated_count += 1
    
    print(f"  Updated {updated_count} files")

def create_requirements():
    """Create requirements.txt file."""
    
    requirements = """# Core dependencies
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10
langchain-core>=0.1.0
langgraph>=0.0.20
langsmith>=0.0.70

# Vector stores and search
qdrant-client>=1.7.0
rank-bm25>=0.2.2

# Data processing
pandas>=2.0.0
numpy>=1.24.0

# UI frameworks
streamlit>=1.28.0
gradio>=4.0.0
flask>=3.0.0

# Evaluation
ragas>=0.1.0

# Utilities
python-dotenv>=1.0.0
tavily-python>=0.3.0

# Optional (for enhanced features)
cohere>=4.0.0
watchdog>=3.0.0
"""
    
    requirements_dev = """# Development dependencies
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
ipython>=8.0.0
jupyter>=1.0.0
"""
    
    print("\n📝 Creating requirements files...")
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("  ✓ Created requirements.txt")
    
    with open('requirements-dev.txt', 'w') as f:
        f.write(requirements_dev)
    print("  ✓ Created requirements-dev.txt")

def create_main_entry_point():
    """Create main entry point for the application."""
    
    run_py = '''#!/usr/bin/env python3
"""
Main entry point for Insurance RAG System
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main function to run the application."""
    
    print("🚀 Insurance RAG System")
    print("=" * 50)
    print("Select mode:")
    print("1. Streamlit UI (recommended)")
    print("2. Gradio UI")
    print("3. Flask API")
    print("4. Run tests")
    print("5. Evaluation mode")
    
    choice = input("\\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        print("Starting Streamlit UI...")
        from src.ui.streamlit_app import main as streamlit_main
        streamlit_main()
    elif choice == "2":
        print("Starting Gradio UI...")
        from src.ui.gradio_app import main as gradio_main
        gradio_main()
    elif choice == "3":
        print("Starting Flask API...")
        from src.ui.flask_app import main as flask_main
        flask_main()
    elif choice == "4":
        print("Running tests...")
        os.system("pytest tests/")
    elif choice == "5":
        print("Running evaluation...")
        from evaluation.ragas_evaluation import main as eval_main
        eval_main()
    else:
        print("Invalid choice!")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open('run.py', 'w') as f:
        f.write(run_py)
    
    print("\n✓ Created run.py entry point")

def create_readme():
    """Create deployment README."""
    
    readme = """# 🚀 Insurance RAG System - Production Deployment

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp config/.env.example .env
# Edit .env with your API keys
```

### 3. Run Application
```bash
python run.py
```

## Deployment Options

### Docker
```bash
docker build -t insurance-rag .
docker run -p 8501:8501 insurance-rag
```

### Direct Streamlit
```bash
streamlit run src/ui/streamlit_app.py
```

### API Mode
```bash
python -m src.ui.flask_app
```

## Project Structure
- `src/` - Core application code
- `tests/` - Test suite
- `evaluation/` - RAGAS evaluation tools
- `data/` - Insurance claims dataset
- `config/` - Configuration files

## Features
- Hybrid search (BM25 + Vector)
- Legal research integration
- Multi-UI support
- Comprehensive testing
- RAGAS evaluation
"""
    
    with open('README_DEPLOYMENT.md', 'w') as f:
        f.write(readme)
    
    print("✓ Created README_DEPLOYMENT.md")

def cleanup_old_files():
    """Remove old/redundant files."""
    
    files_to_delete = [
        'rag.py',
        'test_claim_search.py',
        'demo_results_table.py',
        'simple_ragas_eval.py',
        'quick_search_test.py',
        'interactive_search_test.py',
        'hybrid_query.py',
        'smart_query.py',
        'evaluate_rag.py',
        'debug_claims.py',
        'generate_ragas_testset.py',
        'deployment_structure.md'  # This file itself after reorganization
    ]
    
    print("\n🗑️ Cleaning up old files...")
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"  ✓ Deleted {file}")

def main():
    """Main organization function."""
    
    print("🚀 ORGANIZING PROJECT FOR DEPLOYMENT")
    print("=" * 50)
    
    response = input("This will reorganize your project. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Create folder structure
    create_folder_structure()
    
    # Move files
    move_files()
    
    # Create init files
    create_init_files()
    
    # Update imports
    update_all_imports()
    
    # Create requirements
    create_requirements()
    
    # Create entry point
    create_main_entry_point()
    
    # Create README
    create_readme()
    
    # Cleanup
    cleanup_old_files()
    
    print("\n✅ PROJECT ORGANIZED FOR DEPLOYMENT!")
    print("\nNext steps:")
    print("1. Review the new structure")
    print("2. Test functionality: python run.py")
    print("3. Update .env file in root directory")
    print("4. Deploy using Docker or cloud platform")

if __name__ == "__main__":
    main()