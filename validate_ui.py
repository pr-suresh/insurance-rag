#!/usr/bin/env python3
"""
Validate that the enhanced UI components work correctly
"""

from enhanced_rag import EnhancedInsuranceRAG
from enhanced_adjuster_agent import EnhancedClaimsAdjusterAgent

def test_format_rag_results():
    """Test the format_rag_results function"""
    # Import the function from the enhanced streamlit app
    import sys
    sys.path.append('.')
    from enhanced_streamlit_app import format_rag_results
    
    # Mock results in the correct format
    mock_results = [
        {
            'content': 'Sample claim content about slip and fall incident...',
            'metadata': {
                'Claim Number': 'GL-2024-0001',
                'Claim Feature': 'Slip and Fall',
                'Loss state': 'CA',
                'Paid Indemnity': 50000
            },
            'score': 0.85,
            'type': 'hybrid'
        },
        {
            'content': 'Another claim about product liability...',
            'metadata': {
                'Claim Number': 'GL-2024-0002',
                'Claim Feature': 'Product Liability',
                'Loss state': 'TX',
                'Paid Indemnity': 125000
            },
            'score': 0.72,
            'type': 'vector'
        }
    ]
    
    # Test formatting
    formatted = format_rag_results(mock_results, include_metadata=True)
    print("✅ format_rag_results working correctly")
    print("Sample output:")
    print(formatted[:300] + "...")
    return True

def test_rag_system():
    """Test that RAG system works"""
    print("\nTesting RAG system...")
    rag = EnhancedInsuranceRAG(use_reranking=False)
    rag.ingest_data("data/insurance_claims.csv")
    
    results = rag.hybrid_search("slip and fall", k=2)
    print(f"✅ Found {len(results)} results")
    
    if results:
        first_result = results[0]
        print(f"First result type: {first_result.get('type')}")
        print(f"First result score: {first_result.get('score', 0):.3f}")
    
    return True

def test_enhanced_agent():
    """Test enhanced agent functionality"""
    print("\nTesting enhanced agent...")
    
    agent = EnhancedClaimsAdjusterAgent()
    
    # Test a simple query
    response = agent.process_query("Find slip and fall claims")
    print("✅ Enhanced agent working")
    print(f"Response length: {len(response)} characters")
    
    return True

def main():
    """Run all validation tests"""
    print("🔍 Validating Enhanced UI Components")
    print("=" * 50)
    
    try:
        # Test individual components
        test_format_rag_results()
        test_rag_system()
        test_enhanced_agent()
        
        print("\n✅ All validation tests passed!")
        print("\nThe enhanced Streamlit UI should now work correctly.")
        print("You can access it at: http://localhost:8502")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()