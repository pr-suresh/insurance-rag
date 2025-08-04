# Task 7: Assessing Performance

**You are the AI Evaluation & Performance Engineer**.  It's time to assess all options for this product.

<aside>
📝

Task 7: Assess the performance of the naive agentic RAG application versus the applications with advanced retrieval tooling

</aside>

**✅ Deliverables**

1. How does the performance compare to your original RAG application?  Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements.  Provide results in a table.

    
    Since my dataset has some structured data fields and also unstructured data in the form of adjuster notes data, using a hybrid of BM25 and dense vector search works the best for the app.

    BM25 handles:
    Exact claim numbers (GL-2024-0024)
    Specific terminology ("LIABILITY ASSESSMENT: Very High")
    Field-based queries ("Loss State: Texas")


    Vector search handles:
    Semantic similarity ("car accident" → "auto-related")
    Conceptual matching ("severe injury" → high severity scores)
    Context understanding (adjuster notes analysis)

    Heres the dataset used for evaluation for RAGAS metrics:

    | # | Question Type | Analysis Type | Question | Expected Answer Summary |
    |---|---------------|---------------|----------|------------------------|
    | 1 | **Simple** | Factual Lookup | What is the total financial exposure for claim GL-2024-0025? | $350,000 total exposure breakdown with claim details |
    | 2 | **Simple** | Factual Lookup | Which company was involved in the carbon monoxide poisoning incident? | Superior Mechanical Systems, HVAC installation error |
    | 3 | **Reasoning** | Causality Analysis | What factors contributed to the high liability assessment in the carbon monoxide case? | 95% liability due to installation negligence, health impacts |
    | 4 | **Reasoning** | Comparative Analysis | How do the settlement strategies differ between cyber liability and premises liability claims? | Different focus areas: breach costs vs. medical costs |
    | 5 | **Multi-Context** | Pattern Analysis | What are the common patterns in high-value claims across different claim types? | Cross-claim patterns in professional, cyber, premises liability |
    | 6 | **Multi-Context** | Statistical Analysis | How do claims reserves and payment patterns vary by state and claim type? | State-specific reserve and payment variations |
    | 7 | **Reasoning** | Regulatory Analysis | What compliance and regulatory issues appear most frequently across the claims portfolio? | PCI, building codes, professional standards violations |
    | 8 | **Multi-Context** | Risk Assessment | Based on injury severity scores and settlement patterns, which claim types present the highest financial risk? | Risk ranking by claim type with specific examples |

    ## 📊 Detailed Performance Results

    | # | Question Type | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Score |
    |---|---------------|-------------|------------------|-------------------|----------------|-----------|
    | 1 | Simple | **0.878** | **0.785** | **0.775** | **0.805** | **0.811** |
    | 2 | Simple | **0.897** | **0.915** | **0.898** | **0.777** | **0.872** |
    | 3 | Reasoning | **0.764** | **0.716** | **0.694** | **0.791** | **0.741** |
    | 4 | Reasoning | **0.685** | **0.750** | **0.780** | **0.799** | **0.754** |
    | 5 | Multi-Context | **0.664** | **0.778** | **0.742** | **0.641** | **0.706** |
    | 6 | Multi-Context | **0.781** | **0.800** | **0.648** | **0.671** | **0.725** |
    | 7 | Reasoning | **0.871** | **0.777** | **0.669** | **0.709** | **0.757** |
    | 8 | Multi-Context | **0.789** | **0.781** | **0.741** | **0.786** | **0.774** |

    ## 📈 Overall Performance Metrics

    | Metric | Score | Grade | Performance Level | Analysis |
    |--------|-------|-------|-------------------|----------|
    | **Faithfulness** | **0.791** | **B** | **Good** | Low hallucination, sticks to facts |
    | **Answer Relevancy** | **0.788** | **B** | **Good** | Addresses questions appropriately |
    | **Context Precision** | **0.743** | **B** | **Good-** | Some irrelevant context retrieved |
    | **Context Recall** | **0.747** | **B** | **Good-** | Adequate information coverage |
    | **Overall Average** | **0.767** | **B** | **Good** | Solid performance across all metrics |


    I also added langsmith tracing that helped me see the exact criteria being used in the calls and to optimize the results.

2. Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?

    Some of the changes I plan to make for second half of the course are:

    1. Implement multi-modal document processing
    2. Create basic analytics dashboard
    3. Implement multi-vector retrieval
    4. Develop specialized agents
    5. Add settlement amount prediction based on similar claims
    6. Create claim timeline visualization
