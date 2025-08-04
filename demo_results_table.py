#!/usr/bin/env python3
"""
Demo RAGAS Results Table - Shows expected output format
"""

import pandas as pd

def create_demo_results():
    """Create a demo results table showing what RAGAS evaluation looks like."""
    
    print("🎯 DEMO: RAGAS Evaluation Results for Insurance RAG")
    print("=" * 70)
    
    # Sample results data (what you might get from real evaluation)
    results_data = [
        {
            "Question": "What is total exposure for GL-2024-0025?",
            "Faithfulness": 0.925,
            "Answer Relevancy": 0.887,
            "Context Precision": 0.833,
            "Context Recall": 0.791
        },
        {
            "Question": "What caused carbon monoxide incident?",
            "Faithfulness": 0.872,
            "Answer Relevancy": 0.901,
            "Context Precision": 0.756,
            "Context Recall": 0.823
        },
        {
            "Question": "Which state had cyber liability claim?",
            "Faithfulness": 0.945,
            "Answer Relevancy": 0.934,
            "Context Precision": 0.889,
            "Context Recall": 0.867
        },
        {
            "Question": "How many credit cards were exposed?",
            "Faithfulness": 0.813,
            "Answer Relevancy": 0.798,
            "Context Precision": 0.724,
            "Context Recall": 0.756
        },
        {
            "Question": "What injury type from slip and fall?",
            "Faithfulness": 0.891,
            "Answer Relevancy": 0.856,
            "Context Precision": 0.812,
            "Context Recall": 0.779
        }
    ]
    
    df = pd.DataFrame(results_data)
    
    # Display detailed results
    print("\n📋 DETAILED RESULTS BY QUESTION")
    print("-" * 70)
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Calculate and display averages
    print("\n\n📊 OVERALL PERFORMANCE METRICS")
    print("-" * 40)
    
    metrics = ['Faithfulness', 'Answer Relevancy', 'Context Precision', 'Context Recall']
    
    print(f"{'Metric':<20} {'Score':<8} {'Grade':<12} {'Interpretation'}")
    print("-" * 65)
    
    for metric in metrics:
        avg_score = df[metric].mean()
        
        # Assign grades
        if avg_score >= 0.9:
            grade = "A+ (Excellent)"
        elif avg_score >= 0.8:
            grade = "A (Good)"
        elif avg_score >= 0.7:
            grade = "B (Fair)"
        elif avg_score >= 0.6:
            grade = "C (Needs Work)"
        else:
            grade = "D (Poor)"
        
        # Interpretation
        interpretations = {
            'Faithfulness': 'Truthfulness to context',
            'Answer Relevancy': 'Addresses the question', 
            'Context Precision': 'Relevant docs retrieved',
            'Context Recall': 'Complete info coverage'
        }
        
        print(f"{metric:<20} {avg_score:.3f}    {grade:<12} {interpretations[metric]}")
    
    # Performance summary
    overall_avg = df[metrics].mean().mean()
    
    print(f"\n🎯 OVERALL RAG PERFORMANCE: {overall_avg:.3f}")
    
    if overall_avg >= 0.85:
        print("🎉 EXCELLENT: Your RAG system is performing very well!")
        recommendation = "Continue monitoring and minor optimizations"
    elif overall_avg >= 0.75:
        print("✅ GOOD: Solid performance with room for improvement")
        recommendation = "Focus on lowest scoring metrics"
    elif overall_avg >= 0.65:
        print("⚠️ FAIR: Acceptable but significant improvements needed")
        recommendation = "Review retrieval strategy and answer generation"
    else:
        print("❌ POOR: Major improvements required")
        recommendation = "Complete system review recommended"
    
    print(f"💡 Recommendation: {recommendation}")
    
    # System-specific insights
    print(f"\n🔍 SYSTEM-SPECIFIC INSIGHTS")
    print("-" * 30)
    
    faithfulness_avg = df['Faithfulness'].mean()
    precision_avg = df['Context Precision'].mean()
    
    if faithfulness_avg > 0.85:
        print("✅ Low hallucination - RAG sticks to facts")
    else:
        print("⚠️ Some hallucination detected - improve context grounding")
    
    if precision_avg > 0.8:
        print("✅ Good retrieval - BM25 + reranking working well")
    else:
        print("⚠️ Retrieval needs improvement - consider tuning search")
    
    # BM25 + Cohere specific recommendations
    print(f"\n🛠️ BM25 + COHERE RERANKING OPTIMIZATIONS")
    print("-" * 45)
    print("• Tune BM25 k1 and b parameters for better keyword matching")
    print("• Adjust reranking top-k to balance precision vs recall")
    print("• Consider hybrid weight adjustment (BM25 vs vector)")
    print("• Optimize chunk size for better context precision")
    
    # Save demo results
    df.to_csv("demo_ragas_results.csv", index=False)
    print(f"\n💾 Demo results saved to 'demo_ragas_results.csv'")
    
    return df

if __name__ == "__main__":
    create_demo_results()