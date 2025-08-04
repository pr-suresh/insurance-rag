#!/usr/bin/env python3
"""
Interactive Hybrid Search Tester
Test your searches in real-time
"""

from enhanced_rag import EnhancedInsuranceRAG
import time

def interactive_test():
    """Interactive search testing."""
    
    print("🔍 INTERACTIVE HYBRID SEARCH TESTER")
    print("=" * 50)
    
    # Initialize
    print("🚀 Loading system...")
    rag = EnhancedInsuranceRAG(use_reranking=False)
    rag.ingest_data("data/insurance_claims.csv")
    print(f"✅ Ready! Loaded {len(rag.documents)} documents\n")
    
    print("💡 Try these example searches:")
    print("   - GL-2024-0024 (exact claim)")
    print("   - auto accident (semantic)")
    print("   - Virginia cyber (hybrid)")
    print("   - claims over 100000 (content)")
    print("   - Superior Mechanical (company)")
    print("\nType 'quit' to exit\n")
    
    while True:
        query = input("🔍 Search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not query:
            continue
            
        print(f"\n--- Searching for: '{query}' ---")
        
        start_time = time.time()
        results = rag.hybrid_search(query, k=5)
        duration = time.time() - start_time
        
        if results:
            print(f"✅ Found {len(results)} results in {duration:.3f}s\n")
            
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                claim_id = metadata.get('claim_id', 'Unknown')
                claim_type = metadata.get('claim_type', 'Unknown')
                total_exposure = metadata.get('total_exposure', 0)
                state = metadata.get('loss_state', 'Unknown')
                score = result['score']
                search_type = result['type']
                
                print(f"{i}. {claim_id} | {claim_type}")
                print(f"   💰 ${total_exposure:,.2f} | 📍 {state}")
                print(f"   📊 Score: {score:.3f} ({search_type})")
                print(f"   📝 {result['content'][:120]}...")
                print()
        else:
            print(f"❌ No results found in {duration:.3f}s\n")
        
        print("-" * 50 + "\n")

if __name__ == "__main__":
    interactive_test()