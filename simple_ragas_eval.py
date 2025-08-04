#!/usr/bin/env python3
"""
Super Simple RAGAS Evaluation - Beginner Friendly
Evaluates your RAG with just a few lines of code
"""

import pandas as pd
from enhanced_rag import EnhancedInsuranceRAG
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def quick_evaluate():
    """One-function evaluation - super simple!"""
    
    print("🚀 Quick RAGAS Evaluation\n")
    
    # 1. Initialize models (cheap ones!)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Initialize your RAG
    rag = EnhancedInsuranceRAG(use_reranking=True)
    rag.ingest_data("data/insurance_claims.csv")
    
    # 3. Test questions (simple ones for your data)
    test_questions = [
        {
            "question": "What is the total exposure for claim GL-2024-0025?",
            "ground_truth": "The total exposure is $350,000"
        },
        {
            "question": "What caused the carbon monoxide incident?",
            "ground_truth": "Improper HVAC furnace venting by Superior Mechanical"
        },
        {
            "question": "Which state had the cyber liability claim?",
            "ground_truth": "Virginia had the cyber liability data breach"
        },
        {
            "question": "How many credit cards were exposed in the breach?",
            "ground_truth": "10,000 customer credit cards were exposed"
        },
        {
            "question": "What type of injury occurred from slip and fall?",
            "ground_truth": "Bodily injury from slipping on wet floors"
        }
    ]
    
    # 4. Run RAG and collect results
    eval_data = []
    for item in test_questions:
        # Get RAG response
        results = rag.hybrid_search(item["question"], k=3)
        
        # Format response
        contexts = [r['content'] for r in results]
        answer = contexts[0][:300] if contexts else "No answer found"
        
        eval_data.append({
            "user_input": item["question"],
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item["ground_truth"]
        })
    
    # 5. Evaluate with RAGAS
    eval_df = pd.DataFrame(eval_data)
    
    print("📊 Running RAGAS evaluation...\n")
    results = evaluate(
        dataset=eval_df,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )
    
    # 6. Show results
    print("=" * 50)
    print("📈 RESULTS")
    print("=" * 50)
    
    results_df = results.to_pandas()
    
    # Average scores
    print("\n📊 Average Scores:")
    print(f"  Faithfulness:      {results_df['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy:  {results_df['answer_relevancy'].mean():.3f}")
    print(f"  Context Precision: {results_df['context_precision'].mean():.3f}")
    print(f"  Context Recall:    {results_df['context_recall'].mean():.3f}")
    
    # Per-question scores
    print("\n📋 Per Question:")
    for idx, row in results_df.iterrows():
        print(f"\nQ{idx+1}: {row['user_input'][:50]}...")
        print(f"  Faith: {row['faithfulness']:.2f} | Relev: {row['answer_relevancy']:.2f}")
        print(f"  Prec: {row['context_precision']:.2f} | Recall: {row['context_recall']:.2f}")
    
    # Save
    results_df.to_csv("ragas_simple_results.csv", index=False)
    print("\n💾 Saved to 'ragas_simple_results.csv'")

if __name__ == "__main__":
    quick_evaluate()