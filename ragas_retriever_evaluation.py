# ragas_retriever_evaluation.py - Evaluate Dense Vector vs BM25 Retrievers using RAGAS
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import time

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# BM25 for keyword search
from rank_bm25 import BM25Okapi

# RAGAS imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    ResponseRelevancy, 
    ContextPrecision
)
from datasets import Dataset

load_dotenv()

class RetrieverEvaluator:
    """Evaluate different retrievers using RAGAS framework."""
    
    def __init__(self):
        # Initialize models
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # RAGAS models for test generation
        self.generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
        self.generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        # Storage
        self.documents = []
        self.chunks = []
        
        print("✅ RetrieverEvaluator initialized")
    
    def load_and_process_data(self, csv_path: str):
        """Load insurance claims data and create documents."""
        print(f"\n📄 Loading and processing {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Clean the data
        df = self._clean_dataframe(df)
        
        # Create documents
        documents = []
        for idx, row in df.iterrows():
            content_parts = []
            
            # Build structured content
            content_parts.append(f"Claim ID: {row.get('Claim Number', 'Unknown')}")
            content_parts.append(f"Claim Type: {row.get('Claim Feature', 'Unknown')}")
            content_parts.append(f"Policy Number: {row.get('Policy Number', 'Unknown')}")
            content_parts.append(f"Insured Company: {row.get('Insured Company Name', 'Unknown')}")
            content_parts.append(f"Loss State: {row.get('Loss state', 'Unknown')}")
            content_parts.append(f"Injury Type: {row.get('Type of injury', 'Unknown')}")
            
            # Add financial information
            paid_indemnity = float(row.get('Paid Indemnity', 0)) if pd.notna(row.get('Paid Indemnity')) else 0
            paid_dcc = float(row.get('Paid DCC', 0)) if pd.notna(row.get('Paid DCC')) else 0
            outstanding_indemnity = float(row.get('Outstanding Indemnity', 0)) if pd.notna(row.get('Outstanding Indemnity')) else 0
            outstanding_dcc = float(row.get('Outstanding DCC', 0)) if pd.notna(row.get('Outstanding DCC')) else 0
            
            content_parts.append(f"Paid Indemnity: ${paid_indemnity:,.2f}")
            content_parts.append(f"Paid DCC: ${paid_dcc:,.2f}")
            content_parts.append(f"Outstanding Indemnity: ${outstanding_indemnity:,.2f}")
            content_parts.append(f"Outstanding DCC: ${outstanding_dcc:,.2f}")
            content_parts.append(f"Total Exposure: ${(paid_indemnity + paid_dcc + outstanding_indemnity + outstanding_dcc):,.2f}")
            
            # Add description and notes if available
            if pd.notna(row.get('Loss Description')) and str(row.get('Loss Description')).strip():
                content_parts.append(f"Loss Description: {row.get('Loss Description')}")
                
            if pd.notna(row.get('Adjuster Notes')) and str(row.get('Adjuster Notes')).strip():
                content_parts.append(f"Adjuster Notes: {row.get('Adjuster Notes')}")
            
            content = "\n".join(content_parts)
            
            # Create metadata
            metadata = {
                'claim_id': str(row.get('Claim Number', '')),
                'claim_type': str(row.get('Claim Feature', '')),
                'insured_company': str(row.get('Insured Company Name', '')),
                'loss_state': str(row.get('Loss state', '')),
                'injury_type': str(row.get('Type of injury', '')),
                'total_exposure': paid_indemnity + paid_dcc + outstanding_indemnity + outstanding_dcc,
                'source': 'insurance_claims',
                'row_index': idx
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
        
        chunks = []
        for doc in documents:
            if len(doc.page_content) <= 1000:
                chunks.append(doc)
            else:
                split_docs = text_splitter.split_documents([doc])
                chunks.extend(split_docs)
        
        self.documents = documents
        self.chunks = chunks
        
        print(f"   ✓ Created {len(documents)} documents and {len(chunks)} chunks")
        return documents[:20]  # Return first 20 for test generation
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe."""
        # Handle numeric columns
        numeric_columns = ['Paid Indemnity', 'Paid DCC', 'Outstanding Indemnity', 'Outstanding DCC']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle date columns
        date_columns = ['Loss Date', 'Policy Effective Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Fill NaN values for string columns
        string_columns = ['Claim Number', 'Claim Feature', 'Policy Number', 'Insured Company Name', 
                         'Loss state', 'Type of injury', 'Loss Description', 'Adjuster Notes']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    def generate_test_dataset(self, documents: List[Document], testset_size: int = 15):
        """Generate synthetic test dataset using RAGAS."""
        print(f"\n🧪 Generating synthetic test dataset with {testset_size} questions...")
        
        generator = TestsetGenerator(
            llm=self.generator_llm, 
            embedding_model=self.generator_embeddings
        )
        
        # Generate test dataset
        dataset = generator.generate_with_langchain_docs(
            documents, 
            testset_size=testset_size
        )
        
        print(f"   ✓ Generated {len(dataset)} test questions")
        return dataset
    
    def setup_dense_retriever(self) -> Any:
        """Setup dense vector retriever."""
        print("\n🔍 Setting up Dense Vector Retriever...")
        
        # Create Qdrant client
        client = QdrantClient(":memory:")
        
        # Create collection
        client.create_collection(
            collection_name="insurance_dense",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        
        # Create vector store
        vector_store = Qdrant(
            client=client,
            collection_name="insurance_dense",
            embeddings=self.embeddings
        )
        
        # Add documents
        vector_store.add_documents(self.chunks)
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        print("   ✓ Dense vector retriever ready")
        return retriever
    
    def setup_bm25_retriever(self):
        """Setup BM25 retriever."""
        print("\n📝 Setting up BM25 Retriever...")
        
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in self.chunks:
            tokens = doc.page_content.lower().split()
            tokenized_docs.append(tokens)
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        print("   ✓ BM25 retriever ready")
        return bm25
    
    def bm25_retrieve(self, bm25, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents using BM25."""
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Return top documents
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append(self.chunks[idx])
        
        return results
    
    def evaluate_retriever(self, retriever, retriever_name: str, test_dataset, is_bm25: bool = False):
        """Evaluate a retriever using RAGAS metrics."""
        print(f"\n📊 Evaluating {retriever_name}...")
        
        eval_data = []
        
        for i, test_sample in enumerate(test_dataset.samples):
            question = test_sample.eval_sample.user_input
            ground_truth = test_sample.eval_sample.reference
            
            try:
                # Retrieve contexts
                if is_bm25:
                    contexts = self.bm25_retrieve(retriever, question, k=5)
                    context_texts = [doc.page_content for doc in contexts]
                else:
                    contexts = retriever.invoke(question)
                    context_texts = [doc.page_content for doc in contexts]
                
                # Generate answer
                context_text = "\n\n".join(context_texts)
                prompt = f"""You are an insurance claims expert. Based ONLY on the context provided below, answer the question accurately.
If the information is not in the context, say "I cannot find this information in the provided context."

Context:
{context_text}

Question: {question}

Answer:"""
                
                response = self.llm.invoke(prompt)
                answer = response.content
                
                eval_data.append({
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": context_texts,
                    "reference": ground_truth
                })
                
                print(f"   ✓ Processed question {i+1}/{len(test_dataset.samples)}")
                
            except Exception as e:
                print(f"   ⚠️ Error processing question {i+1}: {str(e)}")
                continue
        
        # Create evaluation dataset
        eval_dataset = EvaluationDataset.from_list(eval_data)
        
        # Run RAGAS evaluation
        print(f"   🔬 Running RAGAS evaluation for {retriever_name}...")
        
        metrics = [
            LLMContextRecall(),
            Faithfulness(), 
            ResponseRelevancy(),
            ContextPrecision()
        ]
        
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=self.generator_llm,
            run_config={"timeout": 300}
        )
        
        return results
    
    def run_comparison(self, csv_path: str):
        """Run complete comparison between dense and BM25 retrievers."""
        print("🚀 Starting Retriever Comparison Evaluation")
        print("=" * 60)
        
        # Step 1: Load and process data
        documents_for_generation = self.load_and_process_data(csv_path)
        
        # Step 2: Generate test dataset
        test_dataset = self.generate_test_dataset(documents_for_generation, testset_size=12)
        
        # Step 3: Setup retrievers
        dense_retriever = self.setup_dense_retriever()
        bm25_retriever = self.setup_bm25_retriever()
        
        # Step 4: Evaluate Dense Vector Retriever
        print("\n" + "="*60)
        dense_results = self.evaluate_retriever(
            dense_retriever, 
            "Dense Vector Retriever", 
            test_dataset, 
            is_bm25=False
        )
        
        # Step 5: Evaluate BM25 Retriever
        print("\n" + "="*60)
        bm25_results = self.evaluate_retriever(
            bm25_retriever, 
            "BM25 Retriever", 
            test_dataset, 
            is_bm25=True
        )
        
        # Step 6: Display Results
        self.display_comparison_results(dense_results, bm25_results)
        
        return dense_results, bm25_results
    
    def display_comparison_results(self, dense_results, bm25_results):
        """Display comparison results in a formatted table."""
        print("\n" + "="*80)
        print("📊 RETRIEVER EVALUATION RESULTS")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = {
            'Metric': ['Faithfulness', 'Response Relevancy', 'Context Precision', 'Context Recall'],
            'Dense Vector Retriever': [
                f"{dense_results['faithfulness']:.4f}",
                f"{dense_results['answer_relevancy']:.4f}",
                f"{dense_results['context_precision']:.4f}",
                f"{dense_results['context_recall']:.4f}"
            ],
            'BM25 Retriever': [
                f"{bm25_results['faithfulness']:.4f}",
                f"{bm25_results['answer_relevancy']:.4f}",
                f"{bm25_results['context_precision']:.4f}",
                f"{bm25_results['context_recall']:.4f}"
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Calculate winner for each metric
        print("\n📈 PERFORMANCE ANALYSIS:")
        print("-" * 40)
        
        metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        metric_names = ['Faithfulness', 'Response Relevancy', 'Context Precision', 'Context Recall']
        
        dense_wins = 0
        bm25_wins = 0
        
        for metric, name in zip(metrics, metric_names):
            dense_score = dense_results[metric]
            bm25_score = bm25_results[metric]
            
            if dense_score > bm25_score:
                winner = "Dense Vector"
                dense_wins += 1
                diff = dense_score - bm25_score
            elif bm25_score > dense_score:
                winner = "BM25"
                bm25_wins += 1
                diff = bm25_score - dense_score
            else:
                winner = "Tie"
                diff = 0
            
            print(f"{name:<20}: {winner:<15} (Δ {diff:+.4f})")
        
        print(f"\n🏆 OVERALL WINNER:")
        if dense_wins > bm25_wins:
            print(f"   Dense Vector Retriever ({dense_wins}/{len(metrics)} metrics)")
        elif bm25_wins > dense_wins:
            print(f"   BM25 Retriever ({bm25_wins}/{len(metrics)} metrics)")
        else:
            print(f"   Tie ({dense_wins}-{bm25_wins})")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"retriever_comparison_{timestamp}.csv", index=False)
        print(f"\n💾 Results saved to: retriever_comparison_{timestamp}.csv")

def main():
    """Main execution function."""
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize evaluator
    evaluator = RetrieverEvaluator()
    
    # Run comparison
    csv_path = "data/insurance_claims.csv"  # Update path as needed
    
    try:
        dense_results, bm25_results = evaluator.run_comparison(csv_path)
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()