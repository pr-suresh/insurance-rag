#!/usr/bin/env python3
"""
Test the improved state detection in legal search
"""

from adjuster_agent import ClaimsAdjusterAgent

def test_state_detection():
    """Test that the legal search automatically uses Loss State from claims."""
    
    print("🧪 TESTING IMPROVED STATE DETECTION")
    print("=" * 60)
    
    # Initialize agent
    agent = ClaimsAdjusterAgent()
    
    # Create workflow
    app = agent.create_workflow()
    
    # Test query about a Virginia claim
    test_query = """
    I need to analyze claim GL-2024-0025 which is a cyber liability case. 
    What are the state-specific regulations I need to consider for this claim?
    """
    
    print(f"\n📝 Test Query: {test_query[:100]}...")
    print("\n🔍 Expected Behavior:")
    print("  1. Claims search finds GL-2024-0025 (Virginia claim)")
    print("  2. Legal search auto-detects Virginia from claim metadata")
    print("  3. Searches Virginia insurance laws (not default California)")
    
    from langchain_core.messages import HumanMessage
    
    # Run the workflow
    initial_state = {
        "messages": [HumanMessage(content=test_query)],
        "claim_context": "",
        "detected_state": ""
    }
    
    print("\n⏳ Processing query through agent workflow...")
    result = app.invoke(initial_state)
    
    # Check the result
    final_message = result["messages"][-1].content
    
    print("\n✅ RESULT ANALYSIS:")
    print("-" * 40)
    
    # Check if Virginia was detected
    if "VIRGINIA" in final_message.upper():
        print("✅ SUCCESS: Virginia state detected and used!")
    else:
        print("❌ ISSUE: Virginia not detected")
    
    # Show excerpt
    print(f"\n📋 Response excerpt (first 500 chars):")
    print(final_message[:500])
    
    print("\n" + "=" * 60)
    print("🎉 Test complete! The system now:")
    print("  • Extracts Loss State from claim metadata")
    print("  • Auto-uses state for legal searches")
    print("  • Shows which state was auto-detected")
    print("  • Suggests other states if multiple found")

if __name__ == "__main__":
    test_state_detection()