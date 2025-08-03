#!/usr/bin/env python3
"""
Debug script for claim number search issues
"""

import pandas as pd
from enhanced_rag import EnhancedInsuranceRAG

def debug_claim_search(target_claim="GL-2024-0025"):
    """Comprehensive debugging for claim number search."""
    
    print(f"🔍 DEBUGGING CLAIM SEARCH FOR: {target_claim}")
    print("="*60)
    
    # Step 1: Check raw CSV data
    print("\n1️⃣ CHECKING RAW CSV DATA")
    print("-" * 30)
    
    df = pd.read_csv('data/insurance_claims.csv')
    print(f"📊 Total rows in CSV: {len(df)}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check if target claim exists in CSV
    claim_column = 'Claim Number'
    if target_claim in df[claim_column].values:
        print(f"✅ Found {target_claim} in CSV!")
        
        # Get the row
        target_row = df[df[claim_column] == target_claim].iloc[0]
        print(f"📝 Full row data:")
        for col in df.columns[:8]:  # Show first 8 columns
            print(f"   {col}: {target_row[col]}")
    else:
        print(f"❌ {target_claim} NOT found in CSV")
        print("Available claim numbers (first 10):")
        for i, claim in enumerate(df[claim_column].head(10)):
            print(f"   {i+1}. {claim}")
        return
    
    # Step 2: Test RAG system loading
    print(f"\n2️⃣ TESTING RAG SYSTEM LOADING")
    print("-" * 30)
    
    rag = EnhancedInsuranceRAG(use_reranking=False)
    rag.ingest_data('data/insurance_claims.csv')
    
    print(f"📄 Total documents loaded: {len(rag.documents)}")
    
    # Step 3: Check if target claim is in loaded documents
    print(f"\n3️⃣ CHECKING LOADED DOCUMENTS")
    print("-" * 30)
    
    target_found_in_docs = False
    matching_docs = []
    
    for i, doc in enumerate(rag.documents):
        doc_claim_id = doc.metadata.get('claim_id', '')
        if doc_claim_id == target_claim:
            target_found_in_docs = True
            matching_docs.append((i, doc))
            
    if target_found_in_docs:
        print(f"✅ Found {len(matching_docs)} documents with claim {target_claim}")
        for i, (doc_idx, doc) in enumerate(matching_docs[:2]):
            print(f"\n📄 Document {i+1} (index {doc_idx}):")
            print(f"   Claim ID: '{doc.metadata.get('claim_id')}'")
            print(f"   Content preview: {doc.page_content[:150]}...")
    else:
        print(f"❌ {target_claim} NOT found in loaded documents")
        
        # Show what claim IDs we actually have
        unique_claims = set()
        for doc in rag.documents[:20]:  # Check first 20 docs
            claim_id = doc.metadata.get('claim_id', '')
            if claim_id:
                unique_claims.add(claim_id)
        
        print(f"📋 Unique claim IDs found (first 20 docs): {sorted(unique_claims)}")
    
    # Step 4: Test the search function step by step
    print(f"\n4️⃣ TESTING SEARCH FUNCTION")
    print("-" * 30)
    
    # Test detection
    detected = rag._detect_claim_number(target_claim)
    print(f"🔍 Detection result: '{detected}'")
    
    # Test normalization
    normalized = rag._normalize_claim_number(target_claim)
    print(f"📝 Normalization result: '{normalized}'")
    
    # Test exact matching manually
    print(f"\n🔍 Manual exact match test:")
    exact_matches = []
    for doc in rag.documents:
        doc_claim_id = doc.metadata.get('claim_id', '').upper().strip()
        if doc_claim_id == normalized:
            exact_matches.append(doc)
    
    print(f"📊 Manual exact matches found: {len(exact_matches)}")
    
    # Step 5: Run the actual search
    print(f"\n5️⃣ RUNNING ACTUAL SEARCH")
    print("-" * 30)
    
    try:
        results = rag.search_by_claim_number(target_claim)
        print(f"📊 Search results: {len(results)}")
        
        if results:
            for i, result in enumerate(results[:3]):
                print(f"\n📄 Result {i+1}:")
                print(f"   Claim ID: {result['metadata'].get('claim_id')}")
                print(f"   Score: {result['score']}")
                print(f"   Type: {result['type']}")
                print(f"   Content: {result['content'][:100]}...")
        else:
            print("❌ No results returned")
            
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with the problematic claim
    debug_claim_search("GL-2024-0025")
    
    print(f"\n" + "="*60)
    print("🔍 DEBUGGING COMPLETE")