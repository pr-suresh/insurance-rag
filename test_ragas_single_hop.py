#!/usr/bin/env python3
"""
Test Dense Vector and BM25 Retrieval using RAGAS Single-Hop Synthesizer
Generates test data and evaluates both retrieval methods independently
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
import time
from datetime import datetime

# RAGAS imports - try different import paths
try:
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple
    RAGAS_AVAILABLE = True
except ImportError:
    try:
        from ragas.testset import TestsetGenerator
        from ragas.testset.evolutions import simple
        RAGAS_AVAILABLE = True
    except ImportError:
        RAGAS_AVAILABLE = False
        print("⚠️ RAGAS testset generation not available, will use fallback")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# System imports
from enhanced_rag import EnhancedInsuranceRAG
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class RAGASSingleHopTester:
    """Test retrieval methods using RAGAS single-hop synthesized questions."""
    
    def __init__(self):
        print("🚀 Initializing RAGAS Single-Hop Tester...")
        
        # Initialize models
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize RAG system
        self.rag = EnhancedInsuranceRAG(use_reranking=False)
        self.rag.ingest_data("data/insurance_claims.csv")
        
        print(f"✅ System ready with {len(self.rag.documents)} documents")
        
        # Store test results
        self.results = {
            'dense_vector': [],
            'bm25': [],
            'test_questions': []
        }
    
    def generate_single_hop_dataset(self, num_samples=10):
        """Generate single-hop questions using RAGAS."""
        
        print(f"\n📝 Generating {num_samples} single-hop questions using RAGAS...")
        
        if not RAGAS_AVAILABLE:
            print("📋 RAGAS not available, using fallback single-hop questions...")
            return self._create_fallback_single_hop()
        
        try:
            # Convert documents for RAGAS
            ragas_docs = []
            for doc in self.rag.documents[:50]:  # Use subset for faster generation
                ragas_docs.append(Document(
                    page_content=doc['content'],
                    metadata=doc.get('metadata', {})
                ))
            
            # Initialize test generator with single-hop evolution
            generator = TestsetGenerator.from_langchain(
                generator_llm=self.llm,
                critic_llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Generate with only simple (single-hop) evolutions
            print("⏳ Generating single-hop questions (this may take a moment)...")
            
            testset = generator.generate_with_langchain_docs(
                documents=ragas_docs,
                test_size=num_samples,
                distributions={
                    simple: 1.0  # 100% single-hop questions
                },
                with_debugging_logs=False,
                run_config={"max_retries": 3}
            )
            
            # Convert to dataframe
            test_df = testset.to_pandas()
            
            # Save generated dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ragas_single_hop_{timestamp}.csv"
            test_df.to_csv(filename, index=False)
            
            print(f"✅ Generated {len(test_df)} single-hop questions")
            print(f"💾 Saved to {filename}")
            
            return test_df
            
        except Exception as e:
            print(f"⚠️ RAGAS generation failed: {e}")
            print("📋 Using fallback single-hop questions...")
            return self._create_fallback_single_hop()
    
    def _create_fallback_single_hop(self):
        """Create fallback single-hop questions if RAGAS fails."""
        
        single_hop_questions = [
            {
                "question": "What is the claim number for the cyber liability case?",
                "ground_truth": "GL-2024-0025",
                "evolution_type": "simple"
            },
            {
                "question": "Which company was involved in the carbon monoxide incident?",
                "ground_truth": "Superior Mechanical Systems",
                "evolution_type": "simple"
            },
            {
                "question": "What is the loss state for claim GL-2024-0025?",
                "ground_truth": "Virginia",
                "evolution_type": "simple"
            },
            {
                "question": "What type of claim is GL-2024-0024?",
                "ground_truth": "General Liability",
                "evolution_type": "simple"
            },
            {
                "question": "How many credit cards were exposed in the cyber breach?",
                "ground_truth": "10,000",
                "evolution_type": "simple"
            },
            {
                "question": "What is the paid indemnity amount for GL-2024-0025?",
                "ground_truth": "$0",
                "evolution_type": "simple"
            },
            {
                "question": "What caused the carbon monoxide exposure?",
                "ground_truth": "HVAC installation error",
                "evolution_type": "simple"
            },
            {
                "question": "What is the outstanding indemnity for the cyber liability claim?",
                "ground_truth": "$250,000",
                "evolution_type": "simple"
            },
            {
                "question": "Which state had the slip and fall incident?",
                "ground_truth": "Minnesota",
                "evolution_type": "simple"
            },
            {
                "question": "What is the injury severity score for the infant in the CO case?",
                "ground_truth": "7",
                "evolution_type": "simple"
            }
        ]
        
        return pd.DataFrame(single_hop_questions)
    
    def test_dense_vector_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Test dense vector retrieval only."""
        
        start_time = time.time()
        
        # Perform vector search using Qdrant client
        query_embedding = self.rag.embeddings.embed_query(query)
        
        # Search in Qdrant
        from qdrant_client.models import models
        search_result = self.rag.qdrant_client.search(
            collection_name=self.rag.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        
        duration = time.time() - start_time
        
        # Format results
        formatted_results = []
        for hit in search_result:
            formatted_results.append({
                'content': hit.payload.get('content', ''),
                'score': float(hit.score),
                'metadata': hit.payload.get('metadata', {})
            })
        
        return {
            'method': 'dense_vector',
            'query': query,
            'results': formatted_results,
            'duration': duration,
            'num_results': len(formatted_results)
        }
    
    def test_bm25_retrieval(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Test BM25 retrieval only."""
        
        start_time = time.time()
        
        # Perform BM25 search using the BM25 object directly
        from rank_bm25 import BM25Okapi
        import numpy as np
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores for all documents
        scores = self.rag.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        duration = time.time() - start_time
        
        # Format results with BM25 scores
        formatted_results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                doc = self.rag.documents[idx]
                formatted_results.append({
                    'content': doc.page_content if hasattr(doc, 'page_content') else doc.get('content', ''),
                    'score': float(scores[idx]),
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
                })
        
        return {
            'method': 'bm25',
            'query': query,
            'results': formatted_results,
            'duration': duration,
            'num_results': len(formatted_results)
        }
    
    def evaluate_retrieval_quality(self, results: Dict, ground_truth: str) -> Dict[str, float]:
        """Evaluate retrieval quality metrics."""
        
        metrics = {
            'hit_rate': 0.0,
            'mrr': 0.0,  # Mean Reciprocal Rank
            'precision_at_k': 0.0,
            'contains_answer': False
        }
        
        # Check if ground truth appears in results
        for i, result in enumerate(results['results']):
            content = result['content'].lower()
            if ground_truth.lower() in content:
                metrics['contains_answer'] = True
                metrics['hit_rate'] = 1.0
                metrics['mrr'] = 1.0 / (i + 1)
                if i < 3:  # Precision@3
                    metrics['precision_at_k'] = 1.0
                break
        
        return metrics
    
    def run_comparative_test(self, test_df: pd.DataFrame):
        """Run comparative testing on both retrieval methods."""
        
        print("\n🔬 COMPARATIVE RETRIEVAL TESTING")
        print("=" * 60)
        
        # Store detailed results
        detailed_results = []
        
        for idx, row in test_df.iterrows():
            question = row['question']
            ground_truth = row.get('ground_truth', '')
            
            print(f"\n[{idx+1}/{len(test_df)}] Testing: {question[:60]}...")
            
            # Test Dense Vector
            vector_results = self.test_dense_vector_retrieval(question)
            vector_metrics = self.evaluate_retrieval_quality(vector_results, ground_truth)
            
            # Test BM25
            bm25_results = self.test_bm25_retrieval(question)
            bm25_metrics = self.evaluate_retrieval_quality(bm25_results, ground_truth)
            
            # Store results
            detailed_results.append({
                'question': question,
                'ground_truth': ground_truth,
                'vector_hit': vector_metrics['contains_answer'],
                'vector_mrr': vector_metrics['mrr'],
                'vector_time': vector_results['duration'],
                'bm25_hit': bm25_metrics['contains_answer'],
                'bm25_mrr': bm25_metrics['mrr'],
                'bm25_time': bm25_results['duration']
            })
            
            # Print quick comparison
            v_status = "✅" if vector_metrics['contains_answer'] else "❌"
            b_status = "✅" if bm25_metrics['contains_answer'] else "❌"
            print(f"  Vector: {v_status} ({vector_results['duration']:.3f}s)")
            print(f"  BM25:   {b_status} ({bm25_results['duration']:.3f}s)")
        
        return pd.DataFrame(detailed_results)
    
    def generate_report(self, results_df: pd.DataFrame):
        """Generate comprehensive comparison report."""
        
        print("\n" + "=" * 60)
        print("📊 RETRIEVAL COMPARISON REPORT")
        print("=" * 60)
        
        # Overall metrics
        print("\n📈 OVERALL PERFORMANCE")
        print("-" * 40)
        
        vector_accuracy = results_df['vector_hit'].mean() * 100
        bm25_accuracy = results_df['bm25_hit'].mean() * 100
        
        print(f"Dense Vector Accuracy: {vector_accuracy:.1f}%")
        print(f"BM25 Accuracy:        {bm25_accuracy:.1f}%")
        
        # MRR comparison
        vector_mrr = results_df['vector_mrr'].mean()
        bm25_mrr = results_df['bm25_mrr'].mean()
        
        print(f"\nMean Reciprocal Rank:")
        print(f"  Dense Vector: {vector_mrr:.3f}")
        print(f"  BM25:        {bm25_mrr:.3f}")
        
        # Speed comparison
        vector_avg_time = results_df['vector_time'].mean()
        bm25_avg_time = results_df['bm25_time'].mean()
        
        print(f"\nAverage Query Time:")
        print(f"  Dense Vector: {vector_avg_time:.3f}s")
        print(f"  BM25:        {bm25_avg_time:.3f}s")
        
        # Win/Loss analysis
        print("\n🏆 HEAD-TO-HEAD COMPARISON")
        print("-" * 40)
        
        both_found = ((results_df['vector_hit']) & (results_df['bm25_hit'])).sum()
        vector_only = ((results_df['vector_hit']) & (~results_df['bm25_hit'])).sum()
        bm25_only = ((~results_df['vector_hit']) & (results_df['bm25_hit'])).sum()
        neither = ((~results_df['vector_hit']) & (~results_df['bm25_hit'])).sum()
        
        print(f"Both found answer:     {both_found}")
        print(f"Vector only:          {vector_only}")
        print(f"BM25 only:            {bm25_only}")
        print(f"Neither found:        {neither}")
        
        # Question type analysis
        print("\n📋 DETAILED RESULTS")
        print("-" * 40)
        
        display_df = results_df[['question', 'vector_hit', 'bm25_hit', 'vector_mrr', 'bm25_mrr']].copy()
        display_df['question'] = display_df['question'].str[:50] + '...'
        display_df['vector_hit'] = display_df['vector_hit'].map({True: '✅', False: '❌'})
        display_df['bm25_hit'] = display_df['bm25_hit'].map({True: '✅', False: '❌'})
        display_df['vector_mrr'] = display_df['vector_mrr'].apply(lambda x: f"{x:.3f}")
        display_df['bm25_mrr'] = display_df['bm25_mrr'].apply(lambda x: f"{x:.3f}")
        
        print(display_df.to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"retrieval_comparison_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        
        print(f"\n💾 Results saved to {filename}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 40)
        
        if vector_accuracy > bm25_accuracy + 10:
            print("✅ Dense Vector significantly outperforms BM25")
            print("   → Consider using vector-only for better accuracy")
        elif bm25_accuracy > vector_accuracy + 10:
            print("✅ BM25 significantly outperforms Dense Vector")
            print("   → Consider using BM25-only for better accuracy")
        else:
            print("✅ Both methods perform similarly")
            print("   → Hybrid approach recommended for best results")
        
        if bm25_avg_time < vector_avg_time * 0.5:
            print("⚡ BM25 is significantly faster")
            print("   → Use BM25 for latency-sensitive applications")
        
        print("\n" + "=" * 60)

def main():
    """Main test function."""
    
    print("🚀 RAGAS SINGLE-HOP RETRIEVAL TESTING")
    print("=" * 60)
    print("Testing Dense Vector vs BM25 with single-hop questions")
    print("=" * 60)
    
    # Initialize tester
    tester = RAGASSingleHopTester()
    
    # Generate or load test dataset
    test_df = tester.generate_single_hop_dataset(num_samples=10)
    
    # Run comparative tests
    results_df = tester.run_comparative_test(test_df)
    
    # Generate report
    tester.generate_report(results_df)
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()