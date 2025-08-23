#!/usr/bin/env python3
"""
Test script for new Pattern Analysis and Financial Analytics features
Demonstrates the enhanced capabilities of the insurance claims system
"""

import os
from dotenv import load_dotenv
from enhanced_adjuster_agent import EnhancedClaimsAdjusterAgent
from pattern_analysis_agent import PatternAnalysisAgent
from financial_agent import FinancialAnalysisAgent
import json

# Load environment variables
load_dotenv()

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def test_pattern_analysis():
    """Test pattern analysis capabilities"""
    print_section("PATTERN ANALYSIS AGENT TESTS")
    
    agent = PatternAnalysisAgent()
    
    # Test 1: Analyze claim patterns by type
    print("📊 Test 1: Analyzing Slip and Fall Claims")
    print("-" * 40)
    
    tool = agent.tools[0]  # analyze_claim_patterns_by_type
    result = tool.invoke({"claim_type": "Slip and Fall"})
    
    try:
        data = json.loads(result)
        print(f"Total claims: {data['total_claims']}")
        print(f"Average paid indemnity: ${data['avg_paid_indemnity']:,.2f}")
        print(f"Total exposure: ${data['total_exposure']:,.2f}")
        print(f"Top states: {list(data['by_state'].keys())[:3]}")
    except:
        print(result)
    
    # Test 2: Identify risk indicators
    print("\n🎯 Test 2: Identifying Risk Indicators for California")
    print("-" * 40)
    
    tool = agent.tools[1]  # identify_risk_indicators
    result = tool.invoke({"state": "CA"})
    
    try:
        data = json.loads(result)
        print(f"Claims analyzed: {data['total_claims_analyzed']}")
        print(f"Average liability: {data['average_liability_assessment']}%")
        print(f"High liability claims: {data['high_liability_claims']}")
        print(f"Common risk factors: {data['common_risk_factors']}")
    except:
        print(result)
    
    # Test 3: Predict claim outcome
    print("\n🔮 Test 3: Predicting Product Liability Outcome in Texas")
    print("-" * 40)
    
    tool = agent.tools[2]  # predict_claim_outcome
    result = tool.invoke({
        "claim_feature": "Product Liability",
        "injury_type": "Bodily Injury",
        "state": "TX"
    })
    
    try:
        data = json.loads(result)
        print(f"Based on {data['based_on_claims']} similar claims:")
        print(f"Predicted indemnity: ${data['predicted_indemnity']['average']:,.2f}")
        print(f"Range: ${data['predicted_indemnity']['minimum']:,.2f} - ${data['predicted_indemnity']['maximum']:,.2f}")
        print(f"Settlement likelihood: {data['settlement_likelihood']}")
        print(f"Confidence: {data['confidence']}")
    except:
        print(result)

def test_financial_analysis():
    """Test financial analysis capabilities"""
    print_section("FINANCIAL ANALYSIS AGENT TESTS")
    
    agent = FinancialAnalysisAgent()
    
    # Test 1: Calculate total exposure
    print("💰 Test 1: Calculating Total Exposure for California")
    print("-" * 40)
    
    tool = agent.tools[0]  # calculate_total_exposure
    result = tool.invoke({"state": "CA"})
    
    try:
        data = json.loads(result)
        summary = data['financial_summary']
        print(f"Total claims: {data['total_claims']}")
        print(f"Total paid: ${summary['total_paid']:,.2f}")
        print(f"Total outstanding: ${summary['total_outstanding']:,.2f}")
        print(f"Total exposure: ${summary['total_exposure']:,.2f}")
        print(f"Closure rate: {data['claim_status']['closure_rate']}")
    except:
        print(result)
    
    # Test 2: Analyze reserve adequacy
    print("\n📈 Test 2: Analyzing Reserve Adequacy")
    print("-" * 40)
    
    tool = agent.tools[1]  # analyze_reserve_adequacy
    result = tool.invoke({})
    
    try:
        data = json.loads(result)
        closed = data['closed_claims_analysis']
        open_claims = data['open_claims_analysis']
        print(f"Closed claims: {closed['total_closed']}")
        print(f"Average paid: ${closed['average_paid']:,.2f}")
        print(f"Open claims: {open_claims['total_open']}")
        print(f"Total outstanding: ${open_claims['total_outstanding']:,.2f}")
        print(f"Recommendations: {', '.join(data.get('recommendations', []))}")
    except:
        print(result)
    
    # Test 3: Generate cost driver report
    print("\n📊 Test 3: Top 5 Cost Drivers")
    print("-" * 40)
    
    tool = agent.tools[2]  # generate_cost_driver_report
    result = tool.invoke({"top_n": 5})
    
    try:
        data = json.loads(result)
        print(f"Total exposure analyzed: ${data['total_exposure']:,.2f}")
        print(f"Total claims: {data['total_claims_analyzed']}")
        
        # Show top cost drivers by type
        by_type = data['cost_drivers_by_category']['by_claim_type']
        print("\nTop Cost Drivers by Claim Type:")
        for claim_type, metrics in list(by_type.items())[:3]:
            print(f"  - {claim_type}: ${metrics['total_cost']:,.2f} ({metrics['claim_count']} claims)")
    except:
        print(result)

def test_enhanced_agent():
    """Test the enhanced agent with combined capabilities"""
    print_section("ENHANCED ADJUSTER AGENT TESTS")
    
    agent = EnhancedClaimsAdjusterAgent()
    
    # Test queries that combine multiple capabilities
    test_queries = [
        {
            "query": "What patterns exist in slip and fall claims and what's our financial exposure?",
            "description": "Combined Pattern + Financial Analysis"
        },
        {
            "query": "Analyze product liability claims in Texas - show patterns, predict outcomes, and calculate exposure",
            "description": "Multi-tool Analysis"
        },
        {
            "query": "Generate a financial summary for 2023 and identify the main risk patterns",
            "description": "Financial Summary + Risk Analysis"
        }
    ]
    
    for test in test_queries:
        print(f"\n🤖 {test['description']}")
        print(f"Query: {test['query']}")
        print("-" * 40)
        
        response = agent.process_query(test['query'])
        
        # Truncate long responses for display
        if len(response) > 500:
            print(response[:500] + "...\n[Response truncated for display]")
        else:
            print(response)

def main():
    """Run all tests"""
    print("\n" + "🚀 TESTING NEW INSURANCE CLAIMS FEATURES 🚀".center(60))
    print("=" * 60)
    
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return
    
    print("✅ API keys loaded successfully")
    
    try:
        # Run individual agent tests
        test_pattern_analysis()
        test_financial_analysis()
        
        # Run enhanced agent tests
        test_enhanced_agent()
        
        print_section("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        
        # Print summary of new capabilities
        print("""
📋 SUMMARY OF NEW FEATURES:

1. PATTERN ANALYSIS CAPABILITIES:
   ✓ Analyze claim patterns by type, state, or company
   ✓ Identify risk indicators from historical data
   ✓ Predict claim outcomes based on similar cases
   ✓ Analyze settlement trends over time

2. FINANCIAL ANALYTICS CAPABILITIES:
   ✓ Calculate total exposure with flexible filters
   ✓ Analyze reserve adequacy
   ✓ Generate cost driver reports
   ✓ Measure settlement efficiency
   ✓ Create executive financial summaries

3. ENHANCED AGENT INTEGRATION:
   ✓ Combines all capabilities in one workflow
   ✓ Uses multiple tools automatically
   ✓ Provides comprehensive analysis
   ✓ Accessible via Streamlit UI

To run the enhanced UI:
   streamlit run enhanced_streamlit_app.py

To use programmatically:
   from enhanced_adjuster_agent import EnhancedClaimsAdjusterAgent
   agent = EnhancedClaimsAdjusterAgent()
   response = agent.process_query("your query here")
        """)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()