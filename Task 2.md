# Task 2: Propose a Solution

Now that you’ve defined a problem and a user, *there are many possible solutions*.

Choose one, and articulate it.

<aside>
📝

Task 2: Articulate your proposed solution

*Hint:*  

- *Paint a picture of the “better world” that your user will live in.  How will they save time, make money, or produce higher-quality output?*
- *Recall the [LLM Application stack](https://a16z.com/emerging-architectures-for-llm-applications/) we’ve discussed at length*
</aside>

**✅ Deliverables**

1. Write 1-2 paragraphs on your proposed solution.  How will it look and feel to the user?

    Claim Adjusters can interact with the application by asking questions regarding the claims that are assigned to them 


2. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice.
    1. LLM
    2. Embedding Model
    3. Orchestration
    4. Vector Database
    5. Monitoring
    6. Evaluation
    7. User Interface
    8. (Optional) Serving & Inference

        1. LLM: OpenAI GPT-4o-mini and GPT-3.5-turbo - Chose GPT-4o-mini for its superior reasoning
    capabilities in complex claim analysis while being cost-effective for production use.

        2. Embedding Model: OpenAI text-embedding-3-small - Selected for its optimal balance of
        performance and cost, providing high-quality semantic representations for insurance claim
        documents with reasonable API pricing.

        3. Orchestration: LangGraph - Chosen for its ability to create structured, stateful
        workflows that can orchestrate complex multi-step processes like claims analysis,
        legislation search, and adjudication guidance in a reliable manner.

        4. Vector Database: Qdrant (in-memory) - Selected for its ease of deployment without Docker
        requirements, excellent performance for semantic search, and built-in support for hybrid
        search combining vector and keyword approaches.

        5. Monitoring: LangSmith - Integrated for comprehensive tracing and debugging of LLM chains,
        enabling performance monitoring and optimization of the RAG system in production
        environments.

        6. Evaluation: RAGAS (Retrieval Augmented Generation Assessment) - Chosen for its
        specialized RAG evaluation metrics including faithfulness, answer relevancy, and context
        precision, specifically designed to assess retrieval-augmented generation systems.

        7. User Interface: 


    3. Where will you use an agent or agents?  What will you use “agentic reasoning” for in your app?

        The application uses LangGraph to create intelligent agents that can reason about complex
        claims adjudication scenarios.

        1. Intelligent Tool Selection & Orchestration

    The agent autonomously decides which tools to use based on the adjuster's query:
    - Claims Database Search: For finding historical precedents and similar cases
    - State Law Research: For checking compliance and legal requirements
    - Smart Retrieval Routing: Automatically choosing between BM25 (keyword), vector (semantic),
    or hybrid search based on query characteristics

    2. Multi-Step Decision Making Workflow

    The agent follows a structured reasoning process:
    Query Analysis → Tool Selection → Information Gathering → Legal Compliance Check → Final
    Guidance

    The agent reasons about:
    - Context Detection: Automatically detects which state's laws to research
    - Query Classification: Determines if the query needs legal research, precedent analysis, or
    both
    - Information Synthesis: Combines multiple data sources to provide comprehensive guidance

    3. Dynamic Query Routing Intelligence

    The SmartRetrieverRouter uses agentic reasoning to analyze queries and automatically choose
    the best search method:
    - BM25 (Keyword): For exact claim IDs, specific amounts, status keywords
    - Vector (Semantic): For conceptual questions, risk analysis, pattern detection
    - Hybrid: For complex multi-faceted queries requiring both approaches

    4. Professional Decision Support

    The agent reasons through complex adjudication scenarios by:
    - Precedent Analysis: Finding and analyzing similar historical claims
    - Compliance Verification: Cross-referencing actions against state regulations
    - Risk Assessment: Identifying potential red flags and liability factors
    - Action Recommendations: Providing specific, actionable next steps

    5. Contextual State Management

    The agent maintains state throughout the conversation, remembering:
    - Previous tool calls and their results
    - Detected jurisdictions and applicable laws
    - Cumulative findings from multiple data sources

        Traditional RAG systems simply retrieve and respond, but claims adjustment requires
        reasoning about:
        - Legal compliance across different state jurisdictions
        - Risk assessment based on patterns in historical data
        - Multi-source validation combining precedents with current regulations
    

        The agentic approach ensures adjusters get comprehensive, legally-compliant guidance rather
        than simple document retrieval, making the AI a true decision-support partner in the complex
        claims adjudication process.