# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Insurance Claims Intelligence System that combines Retrieval-Augmented Generation (RAG) with legal research capabilities. The system helps insurance claims adjusters analyze historical claims and access state-specific insurance laws using hybrid search (BM25 + Vector embeddings) and agentic workflows.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys: OPENAI_API_KEY, TAVILY_API_KEY, LANGCHAIN_API_KEY
```

### Running the Application
```bash
# Primary UI (Streamlit)
streamlit run streamlit_app.py

# Alternative UIs
python gradio_app.py
python flask_app.py

# Debug mode with comprehensive logging
python debug_adjuster_agent.py
```

### Testing and Evaluation
```bash
# Run comprehensive test suite
python automated_test_runner.py

# Test specific components
python test_hybrid_search.py
python test_langsmith.py
python test_state_detection.py

# Run RAGAS evaluation
python ragas_evaluation.py

# Generate test datasets
python create_ragas_dataset.py
```

### Container Deployment
```bash
# Build and run with Docker
docker build -t insurance-rag .
docker run -p 8501:8501 --env-file .env insurance-rag

# Use docker-compose for full stack
docker-compose up -d
```

## Architecture Overview

### Core System Flow
1. **User Query** → **ClaimsAdjusterAgent** (LangGraph workflow)
2. **Claims Search** → **EnhancedInsuranceRAG** (Hybrid BM25 + Vector search)
3. **Legal Research** → **Tavily API** (State-specific insurance laws)
4. **Analysis** → **GPT-3.5-turbo** → **Structured guidance**

### Key Components

**EnhancedInsuranceRAG** (`enhanced_rag.py`):
- Hybrid search combining BM25 keyword matching with vector similarity
- Qdrant in-memory vector store with OpenAI embeddings
- Claim number detection with fuzzy matching
- Smart state detection from claim metadata
- Optional Cohere reranking for improved precision

**ClaimsAdjusterAgent** (`adjuster_agent.py`):
- LangGraph-based agentic workflow with tool calling
- Two main tools: `search_claims_database` and `search_state_insurance_laws`
- Automatically uses Loss State from claims for legal research
- Multi-step reasoning with conversation state management

**Debug Agent** (`debug_adjuster_agent.py`):
- Enhanced version with comprehensive logging
- LangSmith integration for trace monitoring
- Performance metrics collection
- Session-based debug log saving

### Data Architecture
- **Claims Database**: 157 insurance claims in CSV format (data/insurance_claims.csv)
- **Chunking Strategy**: 1000 characters with 200 character overlap
- **Vector Store**: Qdrant in-memory with 362 searchable chunks
- **BM25 Index**: rank-bm25 for keyword search

## Important Implementation Details

### State Detection and Legal Research
The system automatically extracts Loss State from claim metadata and uses it for state-specific legal research. The `search_state_insurance_laws` tool will auto-detect the state from recent claim searches if no state is explicitly provided.

### Hybrid Search Weighting
- BM25 excels at factual queries (claim numbers, company names, amounts)
- Vector search better for semantic/conceptual queries
- Current implementation uses equal weighting but can be adjusted based on query classification

### Environment Variables Required
- `OPENAI_API_KEY`: Required for LLM and embeddings
- `TAVILY_API_KEY`: Required for legal research functionality
- `LANGCHAIN_API_KEY`: Optional, enables LangSmith monitoring
- `COHERE_API_KEY`: Optional, enables result reranking

### Performance Characteristics
- RAGAS evaluation score: 0.767 (B grade)
- Query processing time: ~3.5s for complex agentic workflows
- BM25 vs Vector comparison: BM25 significantly outperforms on factual single-hop queries (50% vs 0% accuracy)

### Testing Framework
The project includes comprehensive testing:
- **automated_test_runner.py**: Full test automation with success rate reporting
- **RAGAS evaluation**: Uses both manual and auto-generated test datasets
- **LangSmith monitoring**: Production-ready trace monitoring
- **Retrieval comparison**: Detailed analysis of search method performance

### UI Options
Three UI implementations are available:
- **Streamlit** (primary): Full-featured web interface with three search modes
- **Gradio**: Alternative web interface
- **Flask**: API-based interface for custom integrations

Each UI integrates the complete RAG + Legal research workflow through the ClaimsAdjusterAgent.

## Critical Code Patterns

When modifying the search functionality, be aware that:
- Claim numbers are normalized to "GL-YYYY-NNNN" format
- The system maintains `last_searched_states` for automatic legal research
- BM25 and vector searches are combined using score normalization
- All major operations include comprehensive error handling and logging

When working with the agent workflow:
- State is managed through `AdjusterState` TypedDict
- Tool functions are bound to the LLM and called automatically
- Debug mode provides detailed step-by-step logging
- LangSmith tracing can be enabled for production monitoring