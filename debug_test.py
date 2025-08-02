from rag import SimpleInsuranceRAG

# Initialize
rag = SimpleInsuranceRAG()

# Ingest data
rag.ingest_data("data/insurance_claims.csv")

# Test searches with debug info
test_queries = ["auto accident", "slip and fall", "bodily injury", "GL-2024"]

for query in test_queries:
    print(f"\n🔍 Testing query: '{query}'")
    results = rag.search(query, k=3)
    print(f"Number of results: {len(results)}")
    
    if results:
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Score: {result['score']:.4f}")
            print(f"Content preview: {result['content'][:200]}...")
            print(f"Metadata: {result['metadata']}")
    else:
        print("❌ No results found!")

# Also test the vector store directly
print(f"\n🔧 Vector store info:")
try:
    from langchain_community.vectorstores import Qdrant
    vector_store = Qdrant(
        client=rag.qdrant_client, 
        collection_name=rag.collection_name,
        embeddings=rag.embeddings
    )
    
    # Check if collection exists and has data
    collections = rag.qdrant_client.get_collections()
    print(f"Collections: {collections}")
    
    collection_info = rag.qdrant_client.get_collection(rag.collection_name)
    print(f"Collection info: {collection_info}")
    
except Exception as e:
    print(f"Error checking vector store: {e}")