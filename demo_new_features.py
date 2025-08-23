#!/usr/bin/env python3
"""
Demo script for new Pattern Analysis and Financial Analytics features
Shows the structure and capabilities without requiring API keys
"""

import pandas as pd
import json
from datetime import datetime

def demo_pattern_analysis():
    """Demonstrate pattern analysis capabilities"""
    print("\n" + "="*60)
    print("  📊 PATTERN ANALYSIS CAPABILITIES DEMO")
    print("="*60 + "\n")
    
    # Load sample data
    df = pd.read_csv("data/insurance_claims.csv")
    
    print("1. CLAIM PATTERN ANALYSIS")
    print("-" * 40)
    
    # Analyze slip and fall patterns
    slip_falls = df[df['Claim Feature'].str.contains('Slip', case=False, na=False)]
    
    pattern_analysis = {
        'claim_type': 'Slip and Fall',
        'total_claims': len(slip_falls),
        'states_affected': slip_falls['Loss state'].nunique(),
        'average_payout': float(slip_falls['Paid Indemnity'].mean()),
        'total_exposure': float(slip_falls['Paid Indemnity'].sum() + slip_falls['Outstanding Indemnity'].sum()),
        'top_states': slip_falls['Loss state'].value_counts().head(3).to_dict(),
        'injury_types': slip_falls['Type of injury'].value_counts().to_dict()
    }
    
    print(f"Analyzing: {pattern_analysis['claim_type']}")
    print(f"Total claims found: {pattern_analysis['total_claims']}")
    print(f"Average payout: ${pattern_analysis['average_payout']:,.2f}")
    print(f"Total exposure: ${pattern_analysis['total_exposure']:,.2f}")
    print(f"Top affected states: {list(pattern_analysis['top_states'].keys())}")
    
    print("\n2. RISK INDICATOR IDENTIFICATION")
    print("-" * 40)
    
    # Identify high-risk claims
    high_risk = df[df['Paid Indemnity'] > df['Paid Indemnity'].quantile(0.75)]
    
    risk_indicators = {
        'high_risk_count': len(high_risk),
        'common_features': high_risk['Claim Feature'].value_counts().head(3).to_dict(),
        'high_risk_states': high_risk['Loss state'].value_counts().head(3).to_dict(),
        'average_high_risk_payout': float(high_risk['Paid Indemnity'].mean()),
        'risk_multiplier': float(high_risk['Paid Indemnity'].mean() / df['Paid Indemnity'].mean())
    }
    
    print(f"High-risk claims identified: {risk_indicators['high_risk_count']}")
    print(f"Average high-risk payout: ${risk_indicators['average_high_risk_payout']:,.2f}")
    print(f"Risk multiplier: {risk_indicators['risk_multiplier']:.2f}x average")
    print(f"Top high-risk claim types: {list(risk_indicators['common_features'].keys())[:2]}")
    
    print("\n3. OUTCOME PREDICTION")
    print("-" * 40)
    
    # Predict outcome for a hypothetical claim
    product_liability = df[df['Claim Feature'].str.contains('Product', case=False, na=False)]
    
    prediction = {
        'claim_type': 'Product Liability',
        'state': 'TX',
        'based_on_claims': len(product_liability),
        'predicted_payout': {
            'minimum': float(product_liability['Paid Indemnity'].min()),
            'average': float(product_liability['Paid Indemnity'].mean()),
            'maximum': float(product_liability['Paid Indemnity'].max()),
            'median': float(product_liability['Paid Indemnity'].median())
        },
        'settlement_rate': f"{len(product_liability[product_liability['Paid Indemnity'] > 0]) / len(product_liability) * 100:.1f}%",
        'confidence': 'Medium' if len(product_liability) >= 5 else 'Low'
    }
    
    print(f"Predicting outcome for: {prediction['claim_type']} in {prediction['state']}")
    print(f"Based on {prediction['based_on_claims']} similar claims")
    print(f"Predicted payout range: ${prediction['predicted_payout']['minimum']:,.0f} - ${prediction['predicted_payout']['maximum']:,.0f}")
    print(f"Average expected: ${prediction['predicted_payout']['average']:,.2f}")
    print(f"Settlement likelihood: {prediction['settlement_rate']}")

def demo_financial_analysis():
    """Demonstrate financial analysis capabilities"""
    print("\n" + "="*60)
    print("  💰 FINANCIAL ANALYTICS CAPABILITIES DEMO")
    print("="*60 + "\n")
    
    # Load sample data
    df = pd.read_csv("data/insurance_claims.csv")
    
    # Convert numeric columns
    numeric_cols = ['Paid Indemnity', 'Paid DCC', 'Outstanding Indemnity', 'Outstanding DCC']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    print("1. TOTAL EXPOSURE CALCULATION")
    print("-" * 40)
    
    # Calculate exposures
    total_paid = df['Paid Indemnity'].sum() + df['Paid DCC'].sum()
    total_outstanding = df['Outstanding Indemnity'].sum() + df['Outstanding DCC'].sum()
    total_exposure = total_paid + total_outstanding
    
    exposure_summary = {
        'total_claims': len(df),
        'total_paid': float(total_paid),
        'total_outstanding': float(total_outstanding),
        'total_exposure': float(total_exposure),
        'average_per_claim': float(total_exposure / len(df)),
        'closed_claims': len(df[(df['Outstanding Indemnity'] == 0) & (df['Outstanding DCC'] == 0)]),
        'open_claims': len(df[(df['Outstanding Indemnity'] > 0) | (df['Outstanding DCC'] > 0)])
    }
    
    print(f"Total claims: {exposure_summary['total_claims']}")
    print(f"Total paid to date: ${exposure_summary['total_paid']:,.2f}")
    print(f"Total outstanding: ${exposure_summary['total_outstanding']:,.2f}")
    print(f"Total exposure: ${exposure_summary['total_exposure']:,.2f}")
    print(f"Open claims: {exposure_summary['open_claims']} ({exposure_summary['open_claims']/exposure_summary['total_claims']*100:.1f}%)")
    
    print("\n2. RESERVE ADEQUACY ANALYSIS")
    print("-" * 40)
    
    # Analyze reserves
    closed_claims = df[(df['Outstanding Indemnity'] == 0) & (df['Outstanding DCC'] == 0)]
    open_claims = df[(df['Outstanding Indemnity'] > 0) | (df['Outstanding DCC'] > 0)]
    
    reserve_analysis = {
        'closed_claims': {
            'count': len(closed_claims),
            'average_paid': float(closed_claims['Paid Indemnity'].mean()) if len(closed_claims) > 0 else 0,
            'total_paid': float(closed_claims['Paid Indemnity'].sum())
        },
        'open_claims': {
            'count': len(open_claims),
            'total_outstanding': float(open_claims['Outstanding Indemnity'].sum() + open_claims['Outstanding DCC'].sum()),
            'average_reserve': float((open_claims['Outstanding Indemnity'] + open_claims['Outstanding DCC']).mean()) if len(open_claims) > 0 else 0
        },
        'adequacy_indicator': 'Adequate' if len(open_claims) == 0 or (open_claims['Outstanding Indemnity'].sum() > closed_claims['Paid Indemnity'].mean() * len(open_claims) * 0.8) else 'Review Needed'
    }
    
    print(f"Closed claims: {reserve_analysis['closed_claims']['count']}")
    print(f"Average closed claim payout: ${reserve_analysis['closed_claims']['average_paid']:,.2f}")
    print(f"Open claims: {reserve_analysis['open_claims']['count']}")
    print(f"Total reserves: ${reserve_analysis['open_claims']['total_outstanding']:,.2f}")
    print(f"Reserve adequacy: {reserve_analysis['adequacy_indicator']}")
    
    print("\n3. COST DRIVER ANALYSIS")
    print("-" * 40)
    
    # Identify cost drivers
    df['Total Cost'] = df['Paid Indemnity'] + df['Outstanding Indemnity'] + df['Paid DCC'] + df['Outstanding DCC']
    
    # By claim type
    cost_by_type = df.groupby('Claim Feature')['Total Cost'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False).head(5)
    
    print("Top 5 Cost Drivers by Claim Type:")
    for idx, row in cost_by_type.iterrows():
        print(f"  {idx[:30]:30} - Total: ${row['sum']:,.0f} ({int(row['count'])} claims, Avg: ${row['mean']:,.0f})")
    
    # By state
    print("\nTop 5 Cost Drivers by State:")
    cost_by_state = df.groupby('Loss state')['Total Cost'].sum().sort_values(ascending=False).head(5)
    for state, cost in cost_by_state.items():
        claim_count = len(df[df['Loss state'] == state])
        print(f"  {state}: ${cost:,.0f} ({claim_count} claims)")

def demo_enhanced_capabilities():
    """Demonstrate how the enhanced agent combines capabilities"""
    print("\n" + "="*60)
    print("  🤖 ENHANCED AGENT CAPABILITIES")
    print("="*60 + "\n")
    
    print("The Enhanced Claims Adjuster Agent combines all capabilities:")
    print()
    print("📋 AVAILABLE TOOLS:")
    print("-" * 40)
    
    tools = [
        ("search_claims_database", "Search historical claims for precedents"),
        ("search_state_insurance_laws", "Research state-specific regulations"),
        ("analyze_claim_patterns_by_type", "Analyze patterns by claim type"),
        ("identify_risk_indicators", "Identify risk factors from data"),
        ("predict_claim_outcome", "Predict likely claim outcomes"),
        ("analyze_settlement_trends", "Analyze settlement trends"),
        ("calculate_total_exposure", "Calculate financial exposure"),
        ("analyze_reserve_adequacy", "Analyze if reserves are adequate"),
        ("generate_cost_driver_report", "Identify cost drivers"),
        ("calculate_settlement_efficiency", "Analyze settlement efficiency"),
        ("generate_financial_summary", "Generate financial summaries")
    ]
    
    for tool_name, description in tools:
        print(f"  • {tool_name:35} - {description}")
    
    print("\n📝 EXAMPLE MULTI-TOOL QUERIES:")
    print("-" * 40)
    
    example_queries = [
        "What patterns exist in slip and fall claims and what's our financial exposure?",
        "Analyze product liability claims in Texas - show patterns, predict outcomes, and calculate exposure",
        "Generate a financial summary for 2023 and identify the main risk patterns",
        "Find similar claims to GL-2024-0024 and analyze the financial implications",
        "What are the cost drivers for construction defect claims and how can we reduce them?"
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    print("\n🎯 KEY BENEFITS:")
    print("-" * 40)
    print("• Comprehensive Analysis: Combines pattern, financial, and legal insights")
    print("• Predictive Intelligence: Forecasts outcomes based on historical data")
    print("• Risk Management: Identifies high-risk patterns and indicators")
    print("• Financial Optimization: Analyzes reserves and settlement efficiency")
    print("• Automated Workflow: Agent automatically selects relevant tools")

def main():
    """Run the demo"""
    print("\n" + "🚀 NEW FEATURES DEMONSTRATION 🚀".center(60))
    print("=" * 60)
    print("Insurance Claims Intelligence System v2.0")
    print("Demonstrating Pattern Analysis & Financial Analytics")
    
    try:
        # Check if data exists
        import os
        if not os.path.exists("data/insurance_claims.csv"):
            print("\n❌ Error: data/insurance_claims.csv not found")
            print("Please ensure the claims data file is in the correct location")
            return
        
        # Run demos
        demo_pattern_analysis()
        demo_financial_analysis()
        demo_enhanced_capabilities()
        
        print("\n" + "="*60)
        print("  ✅ DEMONSTRATION COMPLETED")
        print("="*60)
        
        print("""
📚 NEXT STEPS:

1. To use with real API keys:
   - Add your OPENAI_API_KEY to .env file
   - Add TAVILY_API_KEY for legal research (optional)
   - Add COHERE_API_KEY for reranking (optional)

2. To run the enhanced UI:
   streamlit run enhanced_streamlit_app.py

3. To test with API keys:
   python test_new_features.py

4. To integrate in your code:
   from enhanced_adjuster_agent import EnhancedClaimsAdjusterAgent
   agent = EnhancedClaimsAdjusterAgent()
   response = agent.process_query("your query here")

The new features add significant value by:
• Identifying hidden patterns in claims data
• Predicting likely outcomes for new claims
• Calculating financial exposure and risk
• Analyzing reserve adequacy
• Identifying cost reduction opportunities
        """)
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()