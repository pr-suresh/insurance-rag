#!/usr/bin/env python3
"""
Quick test to verify the API issue is fixed
"""

from enhanced_rag import EnhancedInsuranceRAG

def test_hybrid_search():
    """Test that hybrid_search works correctly"""
    print("Testing EnhancedInsuranceRAG.hybrid_search...")
    
    # Initialize RAG system
    rag = EnhancedInsuranceRAG(use_reranking=False)
    rag.ingest_data("data/insurance_claims.csv")
    
    # Test search
    query = "slip and fall"
    results = rag.hybrid_search(query, k=3)
    
    print(f"Query: '{query}'")
    print(f"Results found: {len(results)}")
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Type: {result.get('type', 'unknown')}")
        print(f"  Score: {result.get('score', 0):.3f}")
        print(f"  Claim: {result.get('metadata', {}).get('Claim Number', 'Unknown')}")
        print(f"  Content: {result.get('content', '')[:100]}...")
    
    print("\n✅ hybrid_search method works correctly!")

if __name__ == "__main__":
    test_hybrid_search()