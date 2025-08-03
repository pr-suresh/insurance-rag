#!/usr/bin/env python3
"""
Simple claim search tester - helps debug search issues
"""

from enhanced_rag import EnhancedInsuranceRAG

def test_claim_search():
    """Interactive claim search tester."""
    
    print("🔍 CLAIM SEARCH TESTER")
    print("=" * 50)
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = EnhancedInsuranceRAG(use_reranking=False)
    rag.ingest_data('data/insurance_claims.csv')
    print(f"✅ Loaded {len(rag.documents)} documents\n")
    
    while True:
        # Get user input
        query = input("🔍 Enter claim number to search (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not query:
            print("Please enter a claim number.\n")
            continue
            
        print(f"\n--- Searching for: '{query}' ---")
        
        # Test detection
        detected = rag._detect_claim_number(query)
        print(f"🔍 Detected: {detected}")
        
        # Test normalization  
        normalized = rag._normalize_claim_number(query)
        print(f"📝 Normalized: {normalized}")
        
        # Perform search
        try:
            results = rag.search_by_claim_number(query)
            
            if results:
                print(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results[:3], 1):
                    claim_id = result['metadata'].get('claim_id', 'Unknown')
                    score = result['score']
                    search_type = result['type']
                    content_preview = result['content'][:100].replace('\n', ' ')
                    
                    print(f"\n{i}. Claim: {claim_id}")
                    print(f"   Score: {score:.3f}")
                    print(f"   Type: {search_type}")
                    print(f"   Preview: {content_preview}...")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error during search: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    test_claim_search()