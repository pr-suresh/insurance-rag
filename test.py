from rag import SimpleInsuranceRAG

# Initialize
rag = SimpleInsuranceRAG()

# Ingest your data
rag.ingest_data("data/insurance_claims.csv")

# Test a search
results = rag.search("auto accident claims")
for r in results:
    print(r['content'][:200])
    print(f"Score: {r['score']}")
    print("---")

    