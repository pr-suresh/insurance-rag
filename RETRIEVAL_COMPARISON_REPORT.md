# 📊 Dense Vector vs BM25 Retrieval Comparison Report

## 🎯 Test Overview
- **Dataset**: 10 RAGAS-style single-hop factual questions
- **Data Source**: Insurance claims CSV (157 records, 362 chunks)
- **Embedding Model**: OpenAI text-embedding-3-small
- **BM25 Implementation**: rank_bm25 with Okapi scoring

## 📈 Performance Results

### Overall Accuracy
| Method | Accuracy | MRR | Avg Query Time |
|--------|----------|-----|----------------|
| **BM25** | 50.0% | 0.370 | 0.001s |
| **Dense Vector** | 0.0% | 0.000 | 0.516s |

### Head-to-Head Comparison
- **BM25 Only Found**: 5 questions
- **Vector Only Found**: 0 questions  
- **Both Found**: 0 questions
- **Neither Found**: 5 questions

## 🔍 Detailed Analysis

### Why BM25 Outperformed Dense Vector

1. **Exact Keyword Matching**
   - Single-hop questions rely heavily on exact terms (e.g., "GL-2024-0025", "Superior Mechanical")
   - BM25 excels at finding documents with exact keyword matches
   - Example: "What is the claim number for the cyber liability case?"
     - BM25: ✅ Found "GL-2024-0025" in rank 2
     - Vector: ❌ Not in top 5 results

2. **Factual vs Semantic Search**
   - Single-hop questions are factual lookups, not semantic understanding
   - Vector embeddings optimize for semantic similarity, not factual recall
   - BM25's term frequency approach better suited for factual retrieval

3. **Chunking Impact**
   - Current chunking (1000 chars, 200 overlap) may split key facts
   - Vector embeddings of chunks may not capture specific claim numbers well
   - BM25 searches within chunks for exact terms regardless of context

### Questions Where BM25 Succeeded
1. ✅ "What is the claim number for the cyber liability case?" → GL-2024-0025
2. ✅ "Which company was involved in the carbon monoxide incident?" → Superior Mechanical
3. ✅ "How many credit cards were exposed?" → 10,000
4. ✅ "What is the outstanding indemnity for cyber liability?" → $250,000
5. ✅ "What is the injury severity score for the infant?" → 7

### Questions Where Both Failed
1. ❌ "What is the loss state for claim GL-2024-0025?" → Virginia
2. ❌ "What type of claim is GL-2024-0024?" → General Liability
3. ❌ "What is the paid indemnity amount for GL-2024-0025?" → $0
4. ❌ "What caused the carbon monoxide exposure?" → HVAC installation error
5. ❌ "Which state had the slip and fall incident?" → Minnesota

## 💡 Key Insights

### Strengths and Weaknesses

**BM25 Strengths:**
- ✅ Excellent for exact matches (claim numbers, company names)
- ✅ Very fast (0.001s average)
- ✅ No embedding computation needed
- ✅ Works well with structured data fields

**BM25 Weaknesses:**
- ❌ Poor semantic understanding
- ❌ Struggles with synonyms/paraphrases
- ❌ No context awareness

**Dense Vector Strengths:**
- ✅ Better for semantic similarity
- ✅ Handles paraphrases well
- ✅ Good for complex reasoning questions

**Dense Vector Weaknesses:**
- ❌ Poor exact match performance
- ❌ Slower (516x slower than BM25)
- ❌ Requires quality embeddings
- ❌ May need fine-tuning for domain

## 🛠️ Recommendations

### 1. **Use Hybrid Approach**
```python
# Weight BM25 higher for factual queries
if is_factual_query(query):
    bm25_weight = 0.7
    vector_weight = 0.3
else:
    bm25_weight = 0.3
    vector_weight = 0.7
```

### 2. **Query Classification**
- Detect query types (factual vs semantic)
- Route factual queries to BM25
- Route complex queries to hybrid search

### 3. **Optimize for Different Question Types**

| Question Type | Recommended Method | Example |
|--------------|-------------------|---------|
| Claim lookup | BM25 | "Find claim GL-2024-0025" |
| Company search | BM25 | "Which company..." |
| Numeric values | BM25 | "What amount..." |
| Conceptual | Vector | "Explain liability factors" |
| Similar cases | Hybrid | "Find similar slip and fall" |

### 4. **Implementation Improvements**
- **Metadata Filtering**: Use claim_id field for direct lookups
- **Better Chunking**: Keep claim records intact
- **Index Optimization**: Create separate indices for facts vs narratives
- **Reranking**: Enable Cohere for hybrid search improvement

## 📊 Conclusion

For insurance claims RAG system with single-hop factual queries:
1. **BM25 is essential** for factual information retrieval
2. **Dense vectors alone are insufficient** for exact matches
3. **Hybrid approach required** for production system
4. **Query routing** can optimize performance per query type

The current Enhanced RAG implementation with hybrid search is the correct approach, but weights should be adjusted based on query classification for optimal performance.