# 🏢 Insurance Claims Intelligence System

A sophisticated AI-powered system that combines **Retrieval-Augmented Generation (RAG)** with **legal research capabilities** to assist insurance claims adjusters in making informed decisions. The system provides comprehensive analysis of historical claims data and relevant state-specific insurance laws.

## 🎯 Project Purpose

This system addresses the critical need for insurance adjusters to:
- **Quickly analyze similar historical claims** for precedent and settlement guidance
- **Access relevant state-specific insurance laws** and regulations
- **Make data-driven decisions** based on comprehensive claim analysis
- **Reduce research time** from hours to minutes
- **Improve consistency** in claims handling across different adjusters

## 🏗️ System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Claims Intelligence System               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Streamlit UI  │    │   Gradio UI     │                │
│  │   (Primary)     │    │  (Alternative)  │                │
│  └─────────────────┘    └─────────────────┘                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              ClaimsAdjusterAgent                        │ │
│  │  ┌─────────────────┐    ┌─────────────────┐            │ │
│  │  │  Enhanced RAG   │    │  Tavily Search  │            │ │
│  │  │  (Claims DB)    │    │  (Legal Laws)   │            │ │
│  │  └─────────────────┘    └─────────────────┘            │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Qdrant DB    │    │   BM25 Index    │                │
│  │  (Vector Store)│    │ (Keyword Search)│                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Query** → ClaimsAdjusterAgent
2. **Claims Search** → Enhanced RAG (Hybrid Search)
3. **Legal Research** → Tavily Search (State-specific laws)
4. **Analysis** → LLM (GPT-3.5-turbo)
5. **Response** → Structured adjudication guidance

## 🛠️ Technology Stack

### Core AI/ML Technologies
- **LangChain** (v0.3.0+) - Framework for LLM applications
- **LangGraph** (v0.2.0+) - Agent workflow orchestration
- **OpenAI GPT-3.5-turbo** - Primary LLM for analysis
- **OpenAI text-embedding-3-small** - Vector embeddings
- **Qdrant** - Vector database for similarity search
- **BM25** - Keyword-based search algorithm

### Search & Retrieval
- **Hybrid Search** - Combines vector similarity + keyword matching
- **Tavily Search** - Real-time legal research and legislation search
- **Cohere Rerank** (optional) - Result reranking for improved relevance

### Data Processing
- **Pandas** - Data manipulation and cleaning
- **RecursiveCharacterTextSplitter** - Intelligent document chunking
- **Custom claim number detection** - Pattern matching for claim IDs

### User Interface
- **Streamlit** (v1.28.0+) - Primary web interface
- **Gradio** (v4.0.0+) - Alternative UI framework
- **Flask** (v2.3.0+) - Custom web framework option

### Development & Monitoring
- **LangSmith** - LLM application monitoring
- **RAGAS** (v0.2.10) - RAG system evaluation
- **Rich** - Enhanced console output
- **Python-dotenv** - Environment variable management

## 📊 Data Structure

### Claims Database (`data/insurance_claims.csv`)
The system processes a comprehensive dataset containing:
- **157 insurance claims** across various types
- **Key fields**: Claim ID, Type, State, Amount, Description, Adjuster Notes
- **Claim types**: Slip & Fall, Product Liability, Professional E&O, Cyber Liability, etc.
- **Geographic coverage**: Claims from 30+ US states
- **Time period**: 2023-2024 claims data

### Sample Claim Record
```csv
Claim Number,Claim Feature,Policy Number,Loss Description,Type of injury,Loss state,Paid Indemnity,Adjuster Notes
GL-2024-0025,Cyber Liability,POL-GL-890123,Data breach exposed customer credit card information,Personal Injury,VA,0,"05/20/2024 - BREACH DISCOVERED: 10,000 customer credit cards exposed..."
```

## 🚀 Key Features

### 1. Intelligent Claims Search
- **Hybrid retrieval** combining semantic similarity and keyword matching
- **Claim number detection** with fuzzy matching (e.g., "GL20240025" → "GL-2024-0025")
- **State-aware searching** - automatically detects and uses claim location
- **Multi-modal search** - search by claim type, injury, amount, or description

### 2. Legal Research Integration
- **State-specific law search** using Tavily API
- **Automatic state detection** from claim metadata
- **Real-time legislation updates** for current legal context
- **Multi-state comparison** when relevant

### 3. Advanced RAG Capabilities
- **Enhanced chunking** with 1000-character chunks and 200-character overlap
- **BM25 keyword indexing** for traditional search
- **Vector similarity search** using OpenAI embeddings
- **Optional reranking** with Cohere for improved relevance
- **Metadata filtering** for targeted searches

### 4. Agent-Based Workflow
- **LangGraph orchestration** for complex multi-step analysis
- **Tool-based architecture** for modular functionality
- **State management** for conversation context
- **Streaming responses** for real-time feedback

### 5. User-Friendly Interface
- **Streamlit web app** with modern UI design
- **Real-time search** with instant results
- **Visual result formatting** with source attribution
- **Mobile-responsive** design

## 🎮 Usage Examples

### Example 1: Claim-Specific Analysis
```
Query: "Analyze claim GL-2024-0025"
Response: 
- Finds exact claim (cyber liability case in Virginia)
- Searches Virginia insurance laws automatically
- Provides settlement guidance based on similar cases
- Includes legal compliance requirements
```

### Example 2: Legal Research
```
Query: "slip and fall liability in Texas"
Response:
- Searches Texas-specific insurance laws
- Finds relevant case precedents
- Provides liability assessment guidelines
- Includes settlement range recommendations
```

### Example 3: Pattern Analysis
```
Query: "cyber liability cases with data breaches"
Response:
- Identifies similar cyber claims across database
- Analyzes settlement patterns
- Provides risk assessment guidance
- Includes regulatory compliance notes
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- UV package manager (recommended) or pip
- OpenAI API key
- Tavily API key (for legal research)
- Cohere API key (optional, for reranking)

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd insurance-rag
```

2. **Install dependencies**
```bash
uv sync
# or
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key

# Optional
COHERE_API_KEY=your_cohere_key
LANGCHAIN_API_KEY=your_langsmith_key
```

## 📈 Performance & Evaluation

### RAGAS Evaluation Results
The system has been evaluated using RAGAS metrics:
- **Context Relevancy**: Measures how relevant retrieved documents are
- **Faithfulness**: Assesses if generated answers are faithful to retrieved context
- **Answer Relevancy**: Evaluates overall answer quality

### Optimization Features
- **Cost optimization**: Uses GPT-3.5-turbo instead of GPT-4 for efficiency
- **Caching**: Vector embeddings cached for faster retrieval
- **Streaming**: Real-time response generation
- **Hybrid search**: Combines multiple retrieval methods for better results

## 🔧 Configuration Options

### RAG Configuration
```python
# Enhanced RAG settings
chunk_size = 1000
chunk_overlap = 200
top_k_results = 7
use_reranking = True  # Requires Cohere API
```

### Agent Configuration
```python
# LLM settings
model = "gpt-3.5-turbo"
temperature = 0
streaming = True

# Search settings
claims_search_limit = 5
legal_search_states = ["CA", "TX", "NY", "FL"]  # Default states
```

## 🧪 Testing & Development

### Running Tests
```bash
# Test state detection
python test_state_detection.py

# Test hybrid search
python test_hybrid_search.py

# Test agent workflow
python debug_adjuster_agent.py
```

### Evaluation Scripts
- `ragas_evaluation.py` - RAG system evaluation
- `evaluate_with_ragas_dataset.py` - Dataset-based evaluation
- `generate_ragas_testset.py` - Test set generation

## 📁 Project Structure

```
insurance-rag/
├── adjuster_agent.py          # Main agent implementation
├── enhanced_rag.py            # Enhanced RAG system
├── streamlit_app.py           # Primary web interface
├── gradio_app.py              # Alternative UI
├── flask_app.py               # Custom web framework
├── data/
│   └── insurance_claims.csv   # Claims database
├── templates/                 # HTML templates
├── pyproject.toml            # Dependencies and project config
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🤝 Contributing

This project is designed for insurance professionals and AI developers. Contributions are welcome in the following areas:

- **Data enhancement**: Adding more claims data or new claim types
- **Legal research**: Expanding state-specific law coverage
- **UI improvements**: Enhancing user experience
- **Performance optimization**: Improving search and retrieval speed
- **Evaluation**: Adding new evaluation metrics

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data privacy regulations when using real claims data.

## 🆘 Support

For questions or issues:
1. Check the existing documentation
2. Review the test files for usage examples
3. Ensure all API keys are properly configured
4. Verify the claims data file is in the correct location

---

**Built with ❤️ for the insurance industry** 