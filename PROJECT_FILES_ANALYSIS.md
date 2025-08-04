# Insurance RAG Project - File Analysis

## 🔴 ESSENTIAL FILES (DO NOT DELETE)

### Core Application Files
1. **enhanced_rag.py** - Main RAG implementation with hybrid search
2. **streamlit_app.py** - Primary UI application
3. **adjuster_agent.py** - Legal research agent with Tavily integration
4. **data/insurance_claims.csv** - Your claims database

### Configuration
5. **pyproject.toml** - Project dependencies
6. **uv.lock** - Dependency lock file

### Alternative UIs (Keep if you want multiple interfaces)
7. **flask_app.py** - Flask web application
8. **templates/dashboard.html** - Flask UI template
9. **gradio_app.py** - Gradio interface

## 🟡 UTILITY FILES (Keep for development/testing)

### Testing & Debugging
10. **test_hybrid_search.py** - Comprehensive test suite
11. **quick_search_test.py** - Quick validation tests
12. **interactive_search_test.py** - Interactive testing tool
13. **test_claim_search.py** - Claim search tester
14. **debug_claims.py** - Debugging utility

### Evaluation & Analysis
15. **evaluate_rag.py** - RAGAS evaluation system
16. **test_questions.csv** - Test questions for evaluation
17. **enhanced_test_questions.csv** - Enhanced test dataset
18. **ragas_detailed_scores.csv** - Evaluation results
19. **ragas_summary.csv** - Evaluation summary

## 🟢 CAN BE DELETED (Not needed for production)

### Old/Superseded Files
20. **rag.py** - Old RAG implementation (replaced by enhanced_rag.py)
21. **smart_query.py** - Old query system (functionality in enhanced_rag.py)
22. **hybrid_query.py** - Old hybrid search (integrated in enhanced_rag.py)

### Documentation (optional)
23. **Task 1.md** - Task documentation
24. **Task 2.md** - Task documentation
25. **Task 3.md** - Task documentation
26. **Task 4.md** - Task documentation
27. **Task 5.md** - Task documentation

## 📋 RECOMMENDED ACTIONS

### Minimum Production Setup
Keep only these files for a minimal working app:
```
insurance-rag/
├── enhanced_rag.py          # Core RAG engine
├── streamlit_app.py         # Main UI
├── adjuster_agent.py        # Legal research
├── data/
│   └── insurance_claims.csv # Data
├── pyproject.toml           # Dependencies
└── uv.lock                  # Lock file
```

### Full Development Setup
Keep everything in ESSENTIAL + UTILITY sections for:
- Testing capabilities
- Multiple UI options
- Evaluation tools
- Debugging utilities

### Clean Up Commands
To remove old/superseded files:
```bash
# Remove old implementations
rm rag.py smart_query.py hybrid_query.py

# Remove task documentation (if not needed)
rm Task*.md

# Remove evaluation results (can regenerate)
rm ragas_*.csv
```

### Storage Impact
- **Essential files**: ~500KB
- **Old files to delete**: ~100KB
- **Test/Debug utilities**: ~150KB
- **Evaluation files**: ~50KB

## 🎯 RECOMMENDATION

For production deployment, you only need:
1. enhanced_rag.py
2. streamlit_app.py (or flask_app.py/gradio_app.py)
3. adjuster_agent.py
4. data/insurance_claims.csv
5. pyproject.toml & uv.lock

Everything else is for development, testing, and evaluation.