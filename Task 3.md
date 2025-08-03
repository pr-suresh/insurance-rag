# Task 3: Dealing with the Data

**You are an AI Systems Engineer.**  The AI Solutions Engineer has handed off the plan to you.  Now *you must identify some source data* that you can use for your application.  

Assume that you’ll be doing at least RAG (e.g., a PDF) with a general agentic search (e.g., a search API like [Tavily](https://tavily.com/) or [SERP](https://serpapi.com/)).

<aside>
📝

Task 3: Collect data for (at least) RAG and choose (at least) one external API

*Hint:*  

- *Ask other real people (ideally the people you’re building for!) what they think.*
- *What are the specific questions that your user is likely to ask of your application?  **Write these down**.*
</aside>

**✅ Deliverables**

1. Describe all of your data sources and external APIs, and describe what you’ll use them for.

    1. Historical Claims Database (insurance_claims.csv)

    - Source: Local CSV file containing 157 historical insurance claims records created synthetically using Claude with my domain knowledge working for a claims organization

    - Structure: Comprehensive claims data including:
        - Claim identification (Claim Number, Policy Number)
        - Claim details (Loss Description, Type of Injury, Loss State, Loss Date)
        - Financial data (Paid Indemnity, Outstanding amounts, DCC costs)
        - Company information (Insured Company Name)
        - Detailed adjuster notes with investigation findings, settlement details, and closure
        actions
        - Use: Primary knowledge base for finding precedents, analyzing settlement patterns,
        identifying similar claims, and providing context for adjudication decisions
        - Processing: Converted to vector embeddings for semantic search and indexed with BM25 for
        keyword matching

    2. Generated Evaluation Datasets

        - Source: Synthetic test datasets created by RAGAS from the claims CSV
        - Use: Automated evaluation of RAG system performance using metrics like faithfulness,
        answer relevancy, and context precision
        - Processing: Claims data is transformed into question-answer pairs for systematic testing

    Tavily Search API

    - Service: Real-time web search and research platform
    - Use: Dynamic research of current state insurance laws, regulations, and legal requirements
    - Implementation: Integrated as an agent tool to search for jurisdiction-specific compliance
    requirements, statutory deadlines, and regulatory changes
    - Search Focus: State-specific insurance laws, claims adjuster requirements, liability
    statutes, and compliance deadlines

2. Describe the default chunking strategy that you will use.  Why did you make this decision?

    Default Chunking Strategy

    RecursiveCharacterTextSplitter Configuration

    Splitter Type: RecursiveCharacterTextSplitter from LangChain
  - Chunk Size: 500 characters maximum
  - Overlap: 50 characters between adjacent chunks
  - Processing: Applied to structured claim documents after CSV-to-document conversion

    500 characters captures complete logical units like:
  - A single investigation finding with context
  - One medical assessment or legal update
  - Complete settlement negotiation details
  - Specific liability assessments with reasoning

    
    Focused Context: Each chunk contains actionable intelligence without overwhelming the LLM
    with entire case histories.

    Efficient Processing: 500-character chunks fit well within embedding model limits while
    maintaining semantic coherence.

    Larger chunks (1000+ characters) would include complete claim narratives but would:
  - Dilute retrieval precision for specific adjudication questions
  - Increase processing costs without proportional benefit
  - Risk overwhelming the context window with less relevant details