#!/usr/bin/env python3
"""
Comprehensive Test Script for Hybrid Search Functionality
Tests claim number search, semantic search, and hybrid combinations
"""

import time
from typing import List, Dict
from enhanced_rag import EnhancedInsuranceRAG

class HybridSearchTester:
    def __init__(self):
        """Initialize the test environment."""
        print("🧪 HYBRID SEARCH TESTING SUITE")
        print("=" * 60)
        
        print("🚀 Initializing Enhanced RAG system...")
        self.rag = EnhancedInsuranceRAG(use_reranking=False)
        self.rag.ingest_data("data/insurance_claims.csv")
        print(f"✅ Loaded {len(self.rag.documents)} documents\n")
        
        self.test_results = []
    
    def run_search_test(self, test_name: str, query: str, expected_results: int = None, 
                       should_find_claim: str = None) -> Dict:
        """Run a single search test and collect results."""
        print(f"🔍 TEST: {test_name}")
        print(f"   Query: '{query}'")
        
        start_time = time.time()
        try:
            results = self.rag.hybrid_search(query, k=10)
            duration = time.time() - start_time
            
            # Analyze results
            found_claims = set()
            search_types = set()
            scores = []
            
            for result in results:
                claim_id = result['metadata'].get('claim_id', 'Unknown')
                found_claims.add(claim_id)
                search_types.add(result['type'])
                scores.append(result['score'])
            
            # Check if expected claim was found
            claim_found = should_find_claim in found_claims if should_find_claim else True
            
            # Check result count
            count_ok = (len(results) >= expected_results) if expected_results else True
            
            test_result = {
                'test_name': test_name,
                'query': query,
                'success': claim_found and count_ok,
                'results_count': len(results),
                'expected_count': expected_results,
                'duration': duration,
                'found_claims': list(found_claims)[:5],  # Top 5 claims
                'search_types': list(search_types),
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'max_score': max(scores) if scores else 0,
                'target_claim_found': claim_found,
                'target_claim': should_find_claim
            }
            
            # Display results
            status = "✅ PASS" if test_result['success'] else "❌ FAIL"
            print(f"   {status} - Found {len(results)} results in {duration:.3f}s")
            
            if results:
                print(f"   Search Types: {', '.join(search_types)}")
                print(f"   Avg Score: {test_result['avg_score']:.3f}, Max: {test_result['max_score']:.3f}")
                print(f"   Top Claims: {', '.join(list(found_claims)[:3])}")
                
                if should_find_claim:
                    if claim_found:
                        print(f"   ✅ Target claim '{should_find_claim}' found")
                    else:
                        print(f"   ❌ Target claim '{should_find_claim}' NOT found")
            else:
                print("   ❌ No results returned")
            
            self.test_results.append(test_result)
            print()
            return test_result
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            error_result = {
                'test_name': test_name,
                'query': query,
                'success': False,
                'error': str(e)
            }
            self.test_results.append(error_result)
            print()
            return error_result
    
    def test_exact_claim_searches(self):
        """Test exact claim number searches."""
        print("📋 TESTING EXACT CLAIM NUMBER SEARCHES")
        print("-" * 40)
        
        # Known claim numbers from the data
        test_cases = [
            ("Perfect Format", "GL-2024-0024", 1, "GL-2024-0024"),
            ("Perfect Format 2", "GL-2024-0025", 1, "GL-2024-0025"),
            ("No Hyphens", "GL20240024", 1, "GL-2024-0024"),
            ("Lowercase", "gl-2024-0024", 1, "GL-2024-0024"),
            ("Partial Number", "2024-0024", 1, "GL-2024-0024"),
            ("Just Numbers", "20240024", 1, "GL-2024-0024"),
            ("Non-existent Claim", "GL-9999-9999", 0, None),
        ]
        
        for test_name, query, expected, target in test_cases:
            self.run_search_test(test_name, query, expected, target)
    
    def test_semantic_searches(self):
        """Test semantic/content-based searches."""
        print("🧠 TESTING SEMANTIC SEARCHES")
        print("-" * 40)
        
        test_cases = [
            ("Auto Accidents", "auto accident claims", 3),
            ("Slip and Fall", "slip and fall incidents", 2),
            ("Property Damage", "property damage", 2),
            ("Cyber Liability", "data breach cyber", 2),
            ("High Value Claims", "claims over 100000", 3),
            ("Professional Liability", "professional liability errors", 2),
            ("Medical Issues", "bodily injury medical", 3),
            ("State Specific", "California claims", 2),
            ("Carbon Monoxide", "carbon monoxide poisoning", 1, "GL-2024-0024"),
            ("Credit Card Breach", "credit card data breach", 1, "GL-2024-0025"),
        ]
        
        for test_name, query, expected, *target in test_cases:
            target_claim = target[0] if target else None
            self.run_search_test(test_name, query, expected, target_claim)
    
    def test_hybrid_combinations(self):
        """Test hybrid search combinations."""
        print("🔀 TESTING HYBRID COMBINATIONS")
        print("-" * 40)
        
        test_cases = [
            ("Claim + Context", "GL-2024-0024 carbon monoxide", 1, "GL-2024-0024"),
            ("Amount Range", "claims between 50000 and 200000", 3),
            ("State + Type", "Virginia cyber liability", 1, "GL-2024-0025"),
            ("Company Name", "Superior Mechanical Systems", 1, "GL-2024-0024"),
            ("Date Range", "2024 claims", 5),
            ("Injury Type", "bodily injury claims", 3),
            ("Complex Query", "professional liability engineering firm", 2),
            ("Legal Terms", "statute of limitations", 2),
        ]
        
        for test_name, query, expected, *target in test_cases:
            target_claim = target[0] if target else None
            self.run_search_test(test_name, query, expected, target_claim)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("⚠️  TESTING EDGE CASES")
        print("-" * 40)
        
        test_cases = [
            ("Empty Query", "", 0),
            ("Special Characters", "GL-2024-0024!@#", 1, "GL-2024-0024"),
            ("Very Long Query", "this is a very long query with many words that might test the system limits and see how it handles complex multi-word searches", 1),
            ("Numbers Only", "123456", 1),
            ("Single Character", "a", 3),
            ("SQL Injection", "'; DROP TABLE --", 0),
            ("Unicode Characters", "clàim número", 0),
            ("Mixed Case", "Gl-2024-0024", 1, "GL-2024-0024"),
        ]
        
        for test_name, query, expected, *target in test_cases:
            target_claim = target[0] if target else None
            self.run_search_test(test_name, query, expected, target_claim)
    
    def test_performance(self):
        """Test search performance with various query sizes."""
        print("⚡ TESTING PERFORMANCE")
        print("-" * 40)
        
        queries = [
            "GL-2024-0024",  # Exact match
            "auto accident claims",  # Semantic search
            "claims over $100000 with property damage in California",  # Complex
        ]
        
        for query in queries:
            print(f"🔍 Performance test: '{query}'")
            times = []
            
            for i in range(3):  # Run 3 times
                start = time.time()
                results = self.rag.hybrid_search(query, k=5)
                duration = time.time() - start
                times.append(duration)
                print(f"   Run {i+1}: {duration:.3f}s ({len(results)} results)")
            
            avg_time = sum(times) / len(times)
            print(f"   Average: {avg_time:.3f}s\n")
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        print("📊 TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get('success', False))
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        # Failed tests details
        if failed_tests > 0:
            print("❌ FAILED TESTS:")
            for result in self.test_results:
                if not result.get('success', False):
                    print(f"   - {result['test_name']}: {result['query']}")
                    if 'error' in result:
                        print(f"     Error: {result['error']}")
            print()
        
        # Performance summary
        durations = [r.get('duration', 0) for r in self.test_results if 'duration' in r]
        if durations:
            avg_duration = sum(durations) / len(durations)
            print(f"Average Search Time: {avg_duration:.3f}s")
            print(f"Fastest Search: {min(durations):.3f}s")
            print(f"Slowest Search: {max(durations):.3f}s")
        
        print("\n🎯 RECOMMENDATIONS:")
        if failed_tests == 0:
            print("   ✅ All tests passed! Hybrid search is working correctly.")
        else:
            print(f"   ⚠️  {failed_tests} tests failed. Review failed cases above.")
        
        if durations:
            if avg_duration > 2.0:
                print("   ⚠️  Average search time > 2s. Consider optimizing.")
            else:
                print("   ✅ Good search performance.")

def main():
    """Run the complete test suite."""
    tester = HybridSearchTester()
    
    # Run all test categories
    tester.test_exact_claim_searches()
    tester.test_semantic_searches()
    tester.test_hybrid_combinations()
    tester.test_edge_cases()
    tester.test_performance()
    
    # Generate final report
    tester.generate_report()

if __name__ == "__main__":
    main()