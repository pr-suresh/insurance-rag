# hybrid_query.py - Simple Hybrid Search (BM25 + Vector Search)
# Combines keyword search with semantic search for better results

import os
from typing import List, Dict
from dotenv import load_dotenv
import pandas as pd
from rank_bm25 import BM25Okapi

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

class SimpleHybridSearch:
    """Combines BM25 keyword search with vector search"""
    
    def __init__(self):
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.qdrant_client = QdrantClient(":memory:")
        self.collection_name = "insurance_claims"
        
        # Storage for BM25
        self.documents = []
        self.bm25 = None
        
        print("✅ Hybrid search system initialized")
    
    def load_and_index_data(self, csv_path: str):
        """Load CSV and create both BM25 and vector indices"""
        print(f"\n📄 Loading data from {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Create documents
        self.documents = []
        for idx, row in df.iterrows():
            # Create searchable text
            text = f"""
            Claim ID: {row.get('claim_id', 'N/A')}
            Date: {row.get('date_filed', 'N/A')}
            Type: {row.get('claim_type', 'N/A')}
            Amount: ${row.get('claim_amount', 0):,.2f}
            Status: {row.get('status', 'N/A')}
            Description: {row.get('description', 'N/A')}
            Notes: {row.get('adjuster_notes', 'N/A')}
            """
            
            doc = Document(
                page_content=text.strip(),
                metadata={
                    "claim_id": str(row.get('claim_id', '')),
                    "amount": float(row.get('claim_amount', 0)),
                    "status": str(row.get('status', '')),
                    "type": str(row.get('claim_type', '')),
                    "row": idx
                }
            )
            self.documents.append(doc)
        
        # Create BM25 index
        print("🔍 Creating BM25 index...")
        tokenized_docs = [doc.page_content.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Create vector index
        print("🧮 Creating vector index...")
        from rag import SimpleInsuranceRAG
        rag = SimpleInsuranceRAG()
        rag.create_vector_store(self.documents)
        self.qdrant_client = rag.qdrant_client
        
        print(f"✅ Indexed {len(self.documents)} documents")
    
    def bm25_search(self, query: str, k: int = 3) -> List[Document]:
        """Pure keyword search using BM25"""
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k documents
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's a match
                doc = self.documents[idx]
                # Add score to metadata for transparency
                doc.metadata['bm25_score'] = float(scores[idx])
                results.append(doc)
        
        return results
    
    def vector_search(self, query: str, k: int = 3) -> List[Document]:
        """Semantic search using vectors"""
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        
        results = vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        docs = []
        for doc, score in results:
            doc.metadata['vector_score'] = float(score)
            docs.append(doc)
        
        return docs
    
    def hybrid_search(self, query: str, k: int = 3, bm25_weight: float = 0.5) -> List[Document]:
        """Combine BM25 and vector search results"""
        print(f"\n🔍 Searching for: '{query}'")
        
        # Get results from both methods
        bm25_results = self.bm25_search(query, k=k*2)
        vector_results = self.vector_search(query, k=k*2)
        
        # Combine and deduplicate
        all_docs = {}
        
        # Add BM25 results
        for doc in bm25_results:
            key = doc.metadata.get('row', id(doc))
            all_docs[key] = {
                'doc': doc,
                'bm25_score': doc.metadata.get('bm25_score', 0),
                'vector_score': 0
            }
        
        # Add vector results
        for doc in vector_results:
            key = doc.metadata.get('row', id(doc))
            if key in all_docs:
                all_docs[key]['vector_score'] = doc.metadata.get('vector_score', 0)
            else:
                all_docs[key] = {
                    'doc': doc,
                    'bm25_score': 0,
                    'vector_score': doc.metadata.get('vector_score', 0)
                }
        
        # Calculate combined scores
        for item in all_docs.values():
            # Normalize scores (simple approach)
            bm25_norm = item['bm25_score'] / max(1, max(d['bm25_score'] for d in all_docs.values()))
            vector_norm = item['vector_score'] / max(1, max(d['vector_score'] for d in all_docs.values()))
            
            # Weighted combination
            item['combined_score'] = (bm25_weight * bm25_norm) + ((1 - bm25_weight) * vector_norm)
        
        # Sort by combined score and return top k
        sorted_items = sorted(all_docs.values(), key=lambda x: x['combined_score'], reverse=True)
        return [item['doc'] for item in sorted_items[:k]]
    
    def query_with_answer(self, query: str, search_type: str = "hybrid") -> Dict:
        """Search and generate an answer"""
        # Choose search method
        if search_type == "bm25":
            docs = self.bm25_search(query)
        elif search_type == "vector":
            docs = self.vector_search(query)
        else:  # hybrid
            docs = self.hybrid_search(query)
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        prompt = f"""Based on the following insurance claims data, answer the question.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""
        
        answer = self.llm.invoke(prompt).content
        
        return {
            "query": query,
            "answer": answer,
            "sources": docs,
            "search_type": search_type
        }

def compare_search_methods(system: SimpleHybridSearch, query: str):
    """Compare all three search methods"""
    print(f"\n🔬 Comparing search methods for: '{query}'")
    print("=" * 60)
    
    for method in ["bm25", "vector", "hybrid"]:
        print(f"\n📊 {method.upper()} Search:")
        result = system.query_with_answer(query, search_type=method)
        
        print(f"Answer: {result['answer'][:150]}...")
        print(f"Sources found: {len(result['sources'])}")
        
        if result['sources']:
            doc = result['sources'][0]
            print(f"Top result: {doc.page_content[:100]}...")

def main():
    """Interactive hybrid search demo"""
    print("\n🚀 Hybrid Search System (BM25 + Vector)")
    print("=" * 60)
    
    # Initialize system
    system = SimpleHybridSearch()
    
    # Load data
    system.load_and_index_data("data/insurance_claims.csv")
    
    # Example queries that show BM25 vs Vector strengths
    print("\n📝 Example Queries:")
    print("1. 'CLM-001' - BM25 excels at exact matches")
    print("2. 'expensive car accidents' - Vector understands meaning")
    print("3. 'pending claims over $5000' - Hybrid combines both")
    
    # Demo comparison
    demo = input("\n🔬 Compare search methods? (y/n): ")
    if demo.lower() == 'y':
        test_queries = [
            "CLM-001",  # BM25 wins
            "expensive accidents",  # Vector wins
            "pending auto claims"  # Hybrid best
        ]
        for q in test_queries:
            compare_search_methods(system, q)
    
    # Interactive mode
    print("\n💬 Interactive Mode (type 'quit' to exit)")
    print("Commands: 'bm25:', 'vector:', or just ask (uses hybrid)")
    print("=" * 60)
    
    while True:
        query = input("\n❓ Your question: ")
        if query.lower() == 'quit':
            break
        
        # Check for specific search type
        if query.startswith("bm25:"):
            search_type = "bm25"
            query = query[5:].strip()
        elif query.startswith("vector:"):
            search_type = "vector"
            query = query[7:].strip()
        else:
            search_type = "hybrid"
        
        try:
            result = system.query_with_answer(query, search_type=search_type)
            
            print(f"\n📝 Answer ({search_type} search):")
            print(result['answer'])
            
            print(f"\n📚 Sources ({len(result['sources'])} found):")
            for i, doc in enumerate(result['sources'], 1):
                print(f"\n  Source {i}:")
                print(f"  {doc.page_content[:150]}...")
                if 'claim_id' in doc.metadata:
                    print(f"  Claim ID: {doc.metadata['claim_id']}")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()