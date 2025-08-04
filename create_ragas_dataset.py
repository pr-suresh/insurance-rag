#!/usr/bin/env python3
"""
Create RAGAS Test Dataset - Non-interactive version
"""

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

def create_enhanced_manual_dataset():
    """Create an enhanced manual dataset that simulates RAGAS-generated questions."""
    
    print("📝 Creating enhanced test dataset from CSV data...")
    
    # Load CSV to understand the data better
    df = pd.read_csv("data/insurance_claims.csv")
    
    # Create sophisticated questions that test different RAG capabilities
    test_data = [
        # Simple factual questions
        {
            "user_input": "What is the total financial exposure for claim GL-2024-0025?",
            "reference": "Claim GL-2024-0025 has a total exposure of $350,000, consisting of $0 paid indemnity, $45,000 paid DCC, $250,000 outstanding indemnity, and $55,000 outstanding DCC. This is a cyber liability claim involving a data breach in Virginia.",
            "evolution_type": "simple",
            "question_type": "factual_lookup"
        },
        {
            "user_input": "Which company was involved in the carbon monoxide poisoning incident?",
            "reference": "Superior Mechanical Systems was the company involved in the carbon monoxide poisoning incident (claim GL-2024-0024). They performed an HVAC installation that caused improper furnace venting, leading to CO exposure of a family of 4.",
            "evolution_type": "simple", 
            "question_type": "factual_lookup"
        },
        
        # Reasoning questions
        {
            "user_input": "What factors contributed to the high liability assessment in the carbon monoxide case?",
            "reference": "The carbon monoxide case (GL-2024-0024) had a very high liability assessment (95%) due to clear installation negligence - the exhaust pipe was incorrectly connected, venting CO into the home instead of outside. This was a basic error with no excuse, especially given the serious health consequences including potential cognitive effects on a 6-month-old infant.",
            "evolution_type": "reasoning",
            "question_type": "causality_analysis"
        },
        {
            "user_input": "How do the settlement strategies differ between cyber liability and premises liability claims?",
            "reference": "Cyber liability claims like GL-2024-0025 focus on breach notification costs, regulatory compliance, and credit monitoring services, with ongoing settlement discussions. Premises liability claims like slip and fall cases focus on immediate medical costs, injury severity scoring, and liability percentages based on property maintenance standards.",
            "evolution_type": "reasoning",
            "question_type": "comparative_analysis"
        },
        
        # Multi-context questions
        {
            "user_input": "What are the common patterns in high-value claims across different claim types?",
            "reference": "High-value claims show patterns including: 1) Professional liability with design errors causing structural damage, 2) Cyber liability with large-scale data breaches affecting thousands, 3) Premises liability with serious injuries requiring long-term medical monitoring, and 4) Product liability with installation errors causing health hazards. All typically involve regulatory scrutiny and enhanced prevention measures.",
            "evolution_type": "multi_context",
            "question_type": "pattern_analysis"
        },
        {
            "user_input": "How do claims reserves and payment patterns vary by state and claim type?",
            "reference": "Claims reserves vary significantly: Virginia cyber claims show high outstanding reserves ($250K+), Minnesota premises claims have mixed paid/outstanding ratios, and professional liability claims often have complex reserve structures. Payment patterns reflect state regulatory requirements and litigation environments.",
            "evolution_type": "multi_context",
            "question_type": "statistical_analysis"
        },
        
        # Complex analytical questions
        {
            "user_input": "What compliance and regulatory issues appear most frequently across the claims portfolio?",
            "reference": "Key compliance issues include: PCI compliance failures in cyber claims, building code violations in premises liability, professional licensing and standard-of-care issues in E&O claims, and safety regulation violations in product liability cases. Regulatory bodies frequently involved include state AGs, FTC, and industry-specific regulators.",
            "evolution_type": "reasoning",
            "question_type": "regulatory_analysis"
        },
        {
            "user_input": "Based on injury severity scores and settlement patterns, which claim types present the highest financial risk?",
            "reference": "Highest financial risk claim types based on injury severity and settlements: 1) Product liability with health impacts (carbon monoxide cases score 7/10 for infants), 2) Cyber liability with large affected populations (10,000+ individuals), 3) Professional liability with structural/financial consequences, and 4) Premises liability with permanent injuries requiring ongoing medical care.",
            "evolution_type": "multi_context",
            "question_type": "risk_assessment"
        }
    ]
    
    # Convert to DataFrame
    test_df = pd.DataFrame(test_data)
    
    # Save the dataset
    test_df.to_csv("ragas_enhanced_testset.csv", index=False)
    
    print(f"✅ Created {len(test_df)} enhanced test questions")
    print("💾 Saved to 'ragas_enhanced_testset.csv'")
    
    # Display the questions
    print("\n📋 GENERATED TEST QUESTIONS:")
    print("="*60)
    
    for idx, row in test_df.iterrows():
        print(f"\n{idx+1}. [{row['evolution_type'].upper()}] {row['question_type']}")
        print(f"   Q: {row['user_input']}")
        print(f"   A: {row['reference'][:120]}...")
    
    return test_df

def main():
    print("🚀 CREATING RAGAS-STYLE TEST DATASET")
    print("="*50)
    
    # Create enhanced dataset
    test_df = create_enhanced_manual_dataset()
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"Total Questions: {len(test_df)}")
    
    # Count by type
    evolution_counts = test_df['evolution_type'].value_counts()
    print(f"\nQuestion Types:")
    for etype, count in evolution_counts.items():
        print(f"  {etype}: {count}")
    
    question_counts = test_df['question_type'].value_counts()
    print(f"\nAnalysis Types:")
    for qtype, count in question_counts.items():
        print(f"  {qtype}: {count}")
    
    print(f"\n🎯 This dataset tests:")
    print("  • Factual information retrieval")
    print("  • Causal reasoning and analysis")
    print("  • Cross-claim pattern recognition")
    print("  • Regulatory and compliance knowledge")
    print("  • Risk assessment capabilities")
    
    return test_df

if __name__ == "__main__":
    main()