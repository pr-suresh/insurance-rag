#!/usr/bin/env python3
"""
Quick Hybrid Search Test - Key scenarios with concise output
"""

from enhanced_rag import EnhancedInsuranceRAG
import time

def test_key_scenarios():
    """Test the most important hybrid search scenarios."""
    
    print("🧪 HYBRID SEARCH - KEY SCENARIOS TEST")
    print("=" * 50)
    
    # Initialize RAG
    print("🚀 Initializing system...")
    rag = EnhancedInsuranceRAG(use_reranking=False)
    rag.ingest_data("data/insurance_claims.csv")
    print(f"✅ Ready with {len(rag.documents)} documents\n")
    
    # Test scenarios
    scenarios = [
        # Exact claim searches
        ("Exact Claim", "GL-2024-0024", "GL-2024-0024"),
        ("No Hyphens", "GL20240025", "GL-2024-0025"),
        ("Partial", "2024-0024", "GL-2024-0024"),
        
        # Semantic searches  
        ("Auto Claims", "auto accident claims", None),
        ("Cyber Breach", "data breach credit card", "GL-2024-0025"),
        ("CO Poisoning", "carbon monoxide", "GL-2024-0024"),
        ("Slip Fall", "slip and fall", None),
        
        # Hybrid combinations
        ("State + Type", "Virginia cyber liability", "GL-2024-0025"),
        ("Company Name", "Superior Mechanical", "GL-2024-0024"),
        ("High Value", "claims over 200000", None),
    ]
    
    results = []
    
    for test_name, query, expected_claim in scenarios:
        print(f"🔍 {test_name}: '{query}'")
        
        start_time = time.time()
        search_results = rag.hybrid_search(query, k=5)
        duration = time.time() - start_time
        
        if search_results:
            # Get top result
            top_result = search_results[0]
            top_claim = top_result['metadata'].get('claim_id', 'Unknown')
            score = top_result['score']
            search_type = top_result['type']
            
            # Check if expected claim was found
            found_expected = expected_claim in [r['metadata'].get('claim_id') for r in search_results] if expected_claim else True
            
            status = "✅" if found_expected else "⚠️"
            print(f"   {status} Found {len(search_results)} results in {duration:.3f}s")
            print(f"   Top: {top_claim} (score: {score:.3f}, type: {search_type})")
            
            if expected_claim and not found_expected:
                print(f"   ⚠️ Expected '{expected_claim}' not found")
                
        else:
            print(f"   ❌ No results found")
            found_expected = False
        
        results.append({
            'test': test_name,
            'query': query,
            'found': len(search_results) > 0,
            'expected_found': found_expected,
            'duration': duration,
            'result_count': len(search_results)
        })
        print()
    
    # Summary
    print("📊 SUMMARY")
    print("-" * 30)
    
    total_tests = len(results)
    successful = sum(1 for r in results if r['expected_found'])
    found_results = sum(1 for r in results if r['found'])
    avg_time = sum(r['duration'] for r in results) / total_tests
    
    print(f"Tests Run: {total_tests}")
    print(f"Successful: {successful}/{total_tests} ({successful/total_tests*100:.1f}%)")
    print(f"Found Results: {found_results}/{total_tests}")
    print(f"Avg Search Time: {avg_time:.3f}s")
    
    # Performance categories
    fast_searches = sum(1 for r in results if r['duration'] < 0.1)
    medium_searches = sum(1 for r in results if 0.1 <= r['duration'] < 0.5)
    slow_searches = sum(1 for r in results if r['duration'] >= 0.5)
    
    print(f"\nPerformance:")
    print(f"  Fast (<0.1s):   {fast_searches}")
    print(f"  Medium (0.1-0.5s): {medium_searches}")
    print(f"  Slow (>0.5s):   {slow_searches}")
    
    if successful == total_tests:
        print(f"\n🎉 All tests passed! Hybrid search is working perfectly.")
    else:
        failed = total_tests - successful
        print(f"\n⚠️ {failed} tests had issues. Check results above.")
    
    return results

if __name__ == "__main__":
    test_key_scenarios()