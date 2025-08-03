# query.py - Simple Query System for Insurance Claims
# This builds on your existing rag.py to add intelligent querying

import os
from typing import List, Dict
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
# Note: Cohere reranking disabled due to version compatibility issues
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

class SimpleQuerySystem:
    """Simple system to query your insurance claims intelligently"""
    
    def __init__(self, use_reranking: bool = False):
        # Check API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY in .env file")
        
        # Disable reranking due to compatibility issues
        self.use_reranking = False
        if use_reranking:
            print("   ⚠️  Reranking disabled due to package compatibility issues")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Using mini instead of nano (nano doesn't exist)
            temperature=0,
            max_tokens=500
        )
        
        # Connect to Qdrant (in-memory)
        self.qdrant_client = QdrantClient(":memory:")
        self.collection_name = "insurance_claims"
        
        print("✅ Query system initialized")
        print("   ℹ️  Using basic retrieval (reranking disabled)")
    
    def get_vector_store(self):
        """Get connection to vector store"""
        return Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
    
    def create_retriever(self, k: int = 3):
        """Create a basic retriever"""
        vector_store = self.get_vector_store()
        
        # Basic retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        
        return retriever
    
    def create_qa_chain(self):
        """Create a simple question-answering chain"""
        # Custom prompt for insurance claims
        prompt_template = """You are an insurance claims assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer based on the context, just say that you don't know.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer in a helpful and concise way:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the chain
        retriever = self.create_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Simple chain that stuffs all docs into prompt
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    
    def query(self, question: str, show_sources: bool = True) -> Dict:
        """Ask a question and get an answer with sources"""
        print(f"\n🔍 Processing query: '{question}'")
        
        # Create QA chain
        qa_chain = self.create_qa_chain()
        
        # Get answer
        result = qa_chain.invoke({"query": question})
        
        # Format response
        response = {
            "question": question,
            "answer": result["result"],
            "sources": []
        }
        
        # Add source information if requested
        if show_sources and "source_documents" in result:
            for i, doc in enumerate(result["source_documents"], 1):
                source_info = {
                    "chunk": i,
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                response["sources"].append(source_info)
        
        return response
    
    def format_response(self, response: Dict) -> str:
        """Format the response nicely for display"""
        formatted = f"\n💬 Question: {response['question']}\n"
        formatted += f"\n📝 Answer:\n{response['answer']}\n"
        
        if response['sources']:
            formatted += "\n📚 Sources:\n"
            for source in response['sources']:
                formatted += f"\n  Chunk {source['chunk']}:\n"
                formatted += f"  {source['content']}\n"
                if 'claim_id' in source['metadata']:
                    formatted += f"  Claim ID: {source['metadata']['claim_id']}\n"
        
        return formatted

# Example queries to demonstrate the system
def run_example_queries(query_system: SimpleQuerySystem):
    """Run some example queries"""
    example_queries = [
        "Summarize high-risk claims",
        "What are the pending claims?",
        "Show me claims over $10,000",
        "What types of auto accidents are in the database?",
        "Which claims need immediate attention?"
    ]
    
    print("\n🎯 Running Example Queries\n")
    print("=" * 50)
    
    for query in example_queries[:3]:  # Run first 3 examples
        response = query_system.query(query)
        print(query_system.format_response(response))
        print("=" * 50)

# Main function
def main():
    """Run the query system interactively"""
    print("\n🚀 Insurance Claims Query System")
    print("=" * 50)
    
    # Initialize system
    query_system = SimpleQuerySystem(use_reranking=False)  # Set True if you have Cohere API
    
    # Load data first
    print("\n📄 Loading claims data...")
    from rag import SimpleInsuranceRAG
    rag = SimpleInsuranceRAG()
    rag.ingest_data("data/insurance_claims.csv")
    
    # Transfer vector store to query system
    query_system.qdrant_client = rag.qdrant_client
    
    try:
        # Show example queries
        show_examples = input("\n📋 Show example queries? (y/n): ")
        if show_examples.lower() == 'y':
            run_example_queries(query_system)
        
        # Interactive mode
        print("\n💬 Interactive Query Mode (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            question = input("\n❓ Your question: ")
            if question.lower() == 'quit':
                break
            
            try:
                response = query_system.query(question)
                print(query_system.format_response(response))
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    except (EOFError, KeyboardInterrupt):
        print("\n\nGoodbye! 👋")

if __name__ == "__main__":
    main()