# evaluate_rag.py - Simple RAGAS Evaluation using insurance_claims.csv
# Generates test set from CSV and evaluates naive retriever with RAGAS metrics

import os
import pandas as pd
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document

# RAGAS imports
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Import your existing RAG system
from rag import SimpleInsuranceRAG

load_dotenv()

class SimpleRAGASEvaluator:
    """Simple RAGAS evaluator that uses only insurance_claims.csv."""
    
    def __init__(self):
        # Use cheaper models
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Wrap for RAGAS
        self.ragas_llm = LangchainLLMWrapper(self.llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        # Initialize RAG system
        self.rag = SimpleInsuranceRAG()
        print("✅ RAGAS Evaluator initialized")
    
    def csv_to_documents(self, csv_path: str) -> List[Document]:
        """Convert CSV rows to documents for test generation."""
        print(f"\n📄 Loading {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Found {len(df)} claims")
        
        documents = []
        
        # Convert each claim to a document
        for idx, row in df.iterrows():
            # Create readable text from all columns
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")
            
            doc = Document(
                page_content="\n".join(text_parts),
                metadata={'claim_id': str(row.get('claim_id', idx))}
            )
            documents.append(doc)
        
        print(f"   Created {len(documents)} documents")
        return documents
    
    def generate_test_set(self, csv_path: str, num_questions: int = 20) -> pd.DataFrame:
        """Generate test questions from CSV data using RAGAS."""
        print(f"\n🔬 Generating {num_questions} test questions from your data...")
        
        # Convert CSV to documents
        documents = self.csv_to_documents(csv_path)
        
        # Initialize RAGAS test generator
        generator = TestsetGenerator(
            llm=self.ragas_llm,
            embedding_model=self.ragas_embeddings
        )
        
        try:
            # Generate test set
            testset = generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=num_questions
            )
            
            # Convert to DataFrame
            test_df = testset.to_pandas()
            
            # Ensure we have the right columns
            if 'ground_truths' in test_df.columns and 'ground_truth' not in test_df.columns:
                test_df['ground_truth'] = test_df['ground_truths']
            
            # Clean up
            test_df = test_df.dropna(subset=['question'])
            
            print(f"   ✅ Generated {len(test_df)} questions")
            
            # Show samples
            print("\n   Sample questions:")
            for i, q in enumerate(test_df['question'].head(3), 1):
                print(f"   {i}. {q}")
            
            return test_df
            
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            print("   Using fallback questions...")
            return self._create_fallback_questions()
    
    def _create_fallback_questions(self) -> pd.DataFrame:
        """Simple fallback questions if generation fails."""
        questions = [
            {
                "question": "What is the claim amount for CLM-001?",
                "ground_truth": "The claim amount should be found in the claims data."
            },
            {
                "question": "How many claims are pending?",
                "ground_truth": "The number of pending claims can be calculated from the status column."
            },
            {
                "question": "What types of insurance claims are in the database?",
                "ground_truth": "The claim types include auto, property, and other categories."
            },
            {
                "question": "What is the average claim amount?",
                "ground_truth": "The average can be calculated from all claim amounts."
            },
            {
                "question": "Which claims were filed in January 2024?",
                "ground_truth": "Claims filed in January 2024 can be found by filtering the date column."
            }
        ]
        return pd.DataFrame(questions)
    
    def evaluate_naive_retriever(self, test_df: pd.DataFrame, csv_path: str) -> tuple:
        """Evaluate the naive retriever using RAGAS metrics."""
        print(f"\n🔍 Evaluating Naive Retriever on {len(test_df)} questions...")
        
        # First, load the CSV into the RAG system
        print("   Loading data into RAG system...")
        self.rag.ingest_data(csv_path)
        
        # Prepare evaluation data
        eval_data = []
        
        for idx, row in test_df.iterrows():
            question = row['question']
            ground_truth = row.get('ground_truth', 'No ground truth provided')
            
            # Get retrieval results (top 3 chunks)
            search_results = self.rag.search(question, k=3)
            contexts = [result["content"] for result in search_results]
            
            # Generate answer
            context_text = "\n\n".join(contexts)
            prompt = f"""Based on the context below, answer the question.
            
Context:
{context_text}

Question: {question}

Answer:"""
            
            answer = self.llm.invoke(prompt).content
            
            eval_data.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })
            
            print(f"   ✓ {idx+1}/{len(test_df)}")
        
        # Create dataset for RAGAS
        dataset = Dataset.from_list(eval_data)
        
        # Run RAGAS evaluation
        print("\n📊 Calculating RAGAS metrics...")
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return results, eval_data
    
    def display_results(self, results, eval_data: List[Dict]) -> pd.DataFrame:
        """Display evaluation results in a clear table."""
        print("\n" + "="*60)
        print("📊 RAGAS EVALUATION RESULTS")
        print("="*60)
        
        # Convert results to DataFrame
        results_df = results.to_pandas()
        
        # Calculate averages
        metrics = {
            'faithfulness': results_df['faithfulness'].mean(),
            'answer_relevancy': results_df['answer_relevancy'].mean(),
            'context_precision': results_df['context_precision'].mean(),
            'context_recall': results_df['context_recall'].mean()
        }
        
        # Create summary table
        summary_data = []
        for metric, score in metrics.items():
            summary_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Score': f"{score:.3f}",
                'Percentage': f"{score*100:.1f}%",
                'Grade': self._get_grade(score)
            })
        
        # Add overall score
        overall = sum(metrics.values()) / len(metrics)
        summary_data.append({
            'Metric': 'OVERALL',
            'Score': f"{overall:.3f}",
            'Percentage': f"{overall*100:.1f}%",
            'Grade': self._get_grade(overall)
        })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        # Save detailed results
        results_df.to_csv("ragas_detailed_scores.csv", index=False)
        summary_df.to_csv("ragas_summary.csv", index=False)
        
        return summary_df, metrics
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9: return "A+ (Excellent)"
        elif score >= 0.8: return "A (Very Good)"
        elif score >= 0.7: return "B (Good)"
        elif score >= 0.6: return "C (Fair)"
        else: return "D (Needs Work)"
    
    def analyze_performance(self, metrics: Dict[str, float]):
        """Provide insights about the evaluation results."""
        print("\n" + "="*60)
        print("🔍 PERFORMANCE ANALYSIS")
        print("="*60)
        
        overall = sum(metrics.values()) / len(metrics)
        
        print(f"\n📈 Overall Performance: {overall:.1%}")
        if overall >= 0.8:
            print("   ✅ Your RAG system is performing very well!")
        elif overall >= 0.7:
            print("   🟡 Your RAG system is performing adequately with room for improvement.")
        else:
            print("   🔴 Your RAG system needs significant improvements.")
        
        # Metric-specific insights
        print("\n📋 Metric Breakdown:\n")
        
        # Faithfulness
        print(f"1. Faithfulness ({metrics['faithfulness']:.1%})")
        if metrics['faithfulness'] < 0.8:
            print("   ⚠️ Answers may contain hallucinations")
            print("   💡 Fix: Improve prompts to stick to context only")
        else:
            print("   ✅ Answers are well-grounded in retrieved context")
        
        # Answer Relevancy
        print(f"\n2. Answer Relevancy ({metrics['answer_relevancy']:.1%})")
        if metrics['answer_relevancy'] < 0.8:
            print("   ⚠️ Answers may not fully address questions")
            print("   💡 Fix: Improve prompt engineering")
        else:
            print("   ✅ Answers are relevant to questions")
        
        # Context Precision
        print(f"\n3. Context Precision ({metrics['context_precision']:.1%})")
        if metrics['context_precision'] < 0.7:
            print("   ⚠️ Retrieved chunks contain irrelevant information")
            print("   💡 Fix: Implement hybrid search or reranking")
        else:
            print("   ✅ Retrieved contexts are relevant")
        
        # Context Recall
        print(f"\n4. Context Recall ({metrics['context_recall']:.1%})")
        if metrics['context_recall'] < 0.7:
            print("   ⚠️ Missing important information in retrieval")
            print("   💡 Fix: Increase k, improve chunking strategy")
        else:
            print("   ✅ Retrieving comprehensive information")
        
        # Overall recommendations
        print("\n" + "="*60)
        print("💡 TOP RECOMMENDATIONS")
        print("="*60)
        
        recommendations = []
        if metrics['context_precision'] < 0.7:
            recommendations.append("1. Add BM25 search for better keyword matching")
        if metrics['context_recall'] < 0.7:
            recommendations.append("2. Increase retrieval k from 3 to 5")
        if metrics['faithfulness'] < 0.8:
            recommendations.append("3. Add instruction to only use provided context")
        if metrics['answer_relevancy'] < 0.8:
            recommendations.append("4. Use few-shot examples in prompts")
        
        if recommendations:
            for rec in recommendations:
                print(f"   {rec}")
        else:
            print("   ✅ System is performing well! Consider A/B testing further improvements.")

def main():
    """Run the simple RAGAS evaluation."""
    print("\n🎯 SIMPLE RAGAS EVALUATION FOR INSURANCE CLAIMS")
    print("=" * 60)
    
    # Configuration
    csv_path = "data/insurance_claims.csv"
    num_test_questions = 20
    
    # Initialize evaluator
    evaluator = SimpleRAGASEvaluator()
    
    # Step 1: Generate test set from CSV
    test_df = evaluator.generate_test_set(csv_path, num_test_questions)
    test_df.to_csv("test_questions.csv", index=False)
    print(f"\n💾 Saved {len(test_df)} test questions to test_questions.csv")
    
    # Step 2: Evaluate naive retriever
    results, eval_data = evaluator.evaluate_naive_retriever(test_df, csv_path)
    
    # Step 3: Display results
    summary_df, metrics = evaluator.display_results(results, eval_data)
    
    # Step 4: Analyze performance
    evaluator.analyze_performance(metrics)
    
    print("\n\n✅ Evaluation Complete!")
    print("\n📁 Output files:")
    print("   - test_questions.csv: Generated test questions")
    print("   - ragas_summary.csv: Metric summary table")
    print("   - ragas_detailed_scores.csv: Per-question scores")

if __name__ == "__main__":
    main()