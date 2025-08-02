# Simple Insurance RAG Application
# This file contains everything you need to get started

import os
import pandas as pd
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment variables
load_dotenv()


class SimpleInsuranceRAG:
    """A simple RAG system for insurance claims"""
    
    def __init__(self):
        # Get API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY in your .env file")
        
        # Set up OpenAI
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Use in-memory Qdrant - no Docker needed!
        self.qdrant_client = QdrantClient(":memory:")
        self.collection_name = "insurance_claims"
        
        print("✅ Initialized RAG system")
    
    def load_csv_data(self, csv_path: str) -> List[Document]:
        """Load insurance claims from CSV and convert to documents"""
        print(f"\n📄 Loading CSV file: {csv_path}")
        
        # Read CSV with explicit column names and error handling
        expected_columns = [
            'claim_id', 'claim_type', 'policy_number', 'policy_effective_date',
            'description', 'injury_type', 'loss_state', 'date_filed', 
            'insured_company', 'claim_amount', 'paid_dcc', 'outstanding_indemnity', 
            'outstanding_dcc', 'adjuster_notes'
        ]
        
        try:
            df = pd.read_csv(
                csv_path, 
                names=expected_columns,
                on_bad_lines='skip', 
                engine='python',
                quoting=3,  # Handle quotes properly
                skipinitialspace=True
            )
        except Exception as e:
            print(f"   Error reading CSV: {e}")
            print("   Trying alternative parsing...")
            df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
        
        # Filter out rows that are actually headers
        if not df.empty:
            df = df[df['claim_id'] != 'Claim Number']
        print(f"   Found {len(df)} claims")
        
        # Convert each row to a document
        documents = []
        for idx, row in df.iterrows():
            # Create readable text from the row
            # Convert amount to float if possible
            try:
                amount = float(row.get('claim_amount', 0))
                amount_str = f"${amount:,.2f}"
            except (ValueError, TypeError):
                amount_str = f"${row.get('claim_amount', 'N/A')}"
            
            text = f"""
            Claim ID: {row.get('claim_id', 'N/A')}
            Date Filed: {row.get('date_filed', 'N/A')}
            Claim Type: {row.get('claim_type', 'N/A')}
            Amount: {amount_str}
            Injury Type: {row.get('injury_type', 'N/A')}
            State: {row.get('loss_state', 'N/A')}
            Insured: {row.get('insured_company', 'N/A')}
            Description: {row.get('description', 'N/A')}
            Adjuster Notes: {row.get('adjuster_notes', 'N/A')}
            """
            
            # Create document with metadata
            doc = Document(
                page_content=text.strip(),
                metadata={
                    "claim_id": str(row.get('claim_id', '')),
                    "claim_type": str(row.get('claim_type', '')),
                    "amount": amount if 'amount' in locals() else 0,
                    "injury_type": str(row.get('injury_type', '')),
                    "loss_state": str(row.get('loss_state', '')),
                    "source": "csv",
                    "row": idx
                }
            )
            documents.append(doc)
        
        print(f"   Created {len(documents)} documents")
        return documents
    
    def create_vector_store(self, documents: List[Document]):
        """Create or update the vector store with documents"""
        print("\n🔧 Creating vector store...")
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for better retrieval
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        print(f"   Split into {len(chunks)} chunks")
        
        # Create collection in Qdrant
        try:
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # Size for text-embedding-3-small
                    distance=Distance.COSINE
                )
            )
        except:
            print("   Collection already exists, updating...")
        
        # Add documents to vector store
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        vector_store.add_documents(chunks)
        
        print("✅ Vector store created successfully!")
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant claims"""
        print(f"\n🔍 Searching for: '{query}'")
        
        # Connect to vector store
        vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        
        # Search
        results = vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            })
        
        return formatted_results
    
    def ingest_data(self, csv_path: str):
        """Complete ingestion pipeline"""
        print("\n🚀 Starting data ingestion...")
        
        # Load CSV data
        documents = self.load_csv_data(csv_path)
        
        # Create vector store
        self.create_vector_store(documents)
        
        print("\n✅ Data ingestion complete!")
        
        # Show sample searches
        print("\n📊 Testing with sample queries:")
        test_queries = [
            "auto accident claims",
            "claims over $5000",
            "pending claims"
        ]
        
        for query in test_queries:
            results = self.search(query, k=1)
            if results:
                print(f"\n❓ Query: '{query}'")
                print(f"✅ Found: {results[0]['content'][:150]}...")
                print(f"   Score: {results[0]['score']:.3f}")


# Main execution
if __name__ == "__main__":
    # Initialize the RAG system
    rag = SimpleInsuranceRAG()
    
    # Path to your CSV file
    csv_path = "data/insurance_claims.csv"  # Change this to your CSV path
    
    # Run ingestion
    rag.ingest_data(csv_path)
    
    # Interactive search loop
    print("\n💬 Interactive Search (type 'quit' to exit)")
    try:
        while True:
            query = input("\nEnter your question: ")
            if query.lower() == 'quit':
                break
            
            results = rag.search(query)
            
            print("\n📋 Results:")
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(result['content'][:300] + "...")
                print(f"Score: {result['score']:.3f}")
                print(f"Metadata: {result['metadata']}")
    except (EOFError, KeyboardInterrupt):
        print("\n\nGoodbye! 👋")