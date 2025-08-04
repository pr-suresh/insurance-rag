#!/usr/bin/env python3
"""
Analyze why dense vector retrieval is failing on single-hop questions
"""

from enhanced_rag import EnhancedInsuranceRAG
from langchain_openai import OpenAIEmbeddings
import numpy as np

def analyze_vector_search():
    """Analyze vector search issues."""
    
    print("🔍 ANALYZING VECTOR SEARCH FAILURES")
    print("=" * 60)
    
    # Initialize system
    rag = EnhancedInsuranceRAG(use_reranking=False)
    rag.ingest_data("data/insurance_claims.csv")
    
    # Test queries that failed
    test_queries = [
        ("What is the claim number for the cyber liability case?", "GL-2024-0025"),
        ("Which company was involved in the carbon monoxide incident?", "Superior Mechanical"),
        ("What is the loss state for claim GL-2024-0025?", "Virginia")
    ]
    
    for query, expected in test_queries:
        print(f"\n📝 Query: {query}")
        print(f"   Expected: {expected}")
        
        # Get embeddings
        query_embedding = rag.embeddings.embed_query(query)
        
        # Search Qdrant
        results = rag.qdrant_client.search(
            collection_name=rag.collection_name,
            query_vector=query_embedding,
            limit=5
        )
        
        print(f"\n   Top 5 Vector Results:")
        found = False
        for i, hit in enumerate(results):
            content = hit.payload.get('content', '')[:100]
            score = hit.score
            
            # Check if expected answer is in content
            if expected.lower() in content.lower():
                found = True
                print(f"   {i+1}. ✅ Score: {score:.3f} - {content}...")
            else:
                print(f"   {i+1}. ❌ Score: {score:.3f} - {content}...")
        
        if not found:
            print(f"   ⚠️ Expected answer '{expected}' not found in top 5")
        
        # Now test BM25
        print(f"\n   BM25 Results:")
        query_tokens = query.lower().split()
        scores = rag.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:5]
        
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:
                doc = rag.documents[idx]
                content = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                
                if expected.lower() in content.lower():
                    print(f"   {i+1}. ✅ Score: {scores[idx]:.3f} - {content}...")
                else:
                    print(f"   {i+1}. ❌ Score: {scores[idx]:.3f} - {content}...")
    
    print("\n" + "=" * 60)
    print("💡 ANALYSIS INSIGHTS:")
    print("-" * 40)
    print("1. BM25 performs better on exact keyword matches")
    print("2. Vector search may need better embeddings or chunking")
    print("3. Single-hop factual questions favor keyword matching")
    print("4. Consider hybrid approach with BM25 weighting for factual queries")

if __name__ == "__main__":
    analyze_vector_search()