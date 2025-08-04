# enhanced_rag.py - Enhanced RAG with Hybrid Search, Reranking, and Better Chunking
# Fixes the poor RAGAS scores by implementing all recommended improvements

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# BM25 for hybrid search
from rank_bm25 import BM25Okapi

# Cohere for reranking (optional)
try:
    from langchain_cohere import CohereRerank
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("⚠️  Cohere reranking disabled due to import issues")

load_dotenv()

class EnhancedInsuranceRAG:
    """Enhanced RAG with hybrid search, reranking, and better chunking."""
    
    def __init__(self, use_reranking: bool = True):
        # Use cheaper models
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # In-memory Qdrant
        self.qdrant_client = QdrantClient(":memory:")
        self.collection_name = "insurance_claims"
        
        # Check for Cohere reranking
        self.use_reranking = use_reranking and COHERE_AVAILABLE and os.getenv("COHERE_API_KEY")
        if self.use_reranking:
            self.reranker = CohereRerank(model="rerank-english-v2.0")
            print("✅ Cohere reranking enabled")
        else:
            print("ℹ️  Reranking disabled (add COHERE_API_KEY to enable)")
        
        # Storage for BM25
        self.documents = []
        self.bm25 = None
        
        print("✅ Enhanced RAG initialized")
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe to remove NaN values and fix data issues."""
        print("🧹 Cleaning data...")
        
        # Handle numeric columns properly (amounts in dollars)
        numeric_columns = ['claim_amount', 'paid_dcc', 'outstanding_indemnity', 'outstanding_dcc']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Handle date columns properly
        date_columns = ['date_filed', 'policy_effective_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Convert to string format for display, handling NaT
                df[col] = df[col].dt.strftime('%Y-%m-%d').fillna('Unknown Date')
        
        # Replace NaN with appropriate values after numeric/date conversion
        fill_values = {
            'claim_id': 'UNKNOWN',
            'claim_type': 'General Liability',
            'description': 'No description provided',
            'injury_type': 'Unknown',
            'loss_state': 'Unknown',
            'insured_company': 'Unknown Company',
            'adjuster_notes': 'No notes available'
        }
        
        for col, fill_val in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)
        
        # Clean string columns - remove extra whitespace and handle 'nan' strings
        string_columns = ['claim_id', 'claim_type', 'description', 'injury_type', 
                         'loss_state', 'insured_company', 'adjuster_notes']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # Replace various forms of 'nan' with proper values
                df[col] = df[col].replace(['nan', 'NaN', 'null', 'None', ''], fill_values.get(col, 'Unknown'))
        
        print(f"   ✓ Cleaned {len(df)} records")
        return df
    
    def create_documents_from_csv(self, csv_path: str) -> List[Document]:
        """Create clean documents from CSV with proper chunking."""
        print(f"\n📄 Loading and processing {csv_path}")
        
        # Load CSV with proper error handling for malformed rows
        # Use standard pandas CSV reading for better column alignment
        df = pd.read_csv(csv_path)
        
        # Map the actual CSV columns to our expected names
        column_mapping = {
            'Claim Number': 'claim_id',
            'Claim Feature': 'claim_type', 
            'Policy Number': 'policy_number',
            'Policy Effective Date': 'policy_effective_date',
            'Loss Description': 'description',
            'Type of injury': 'injury_type',
            'Loss state': 'loss_state',
            'Loss Date': 'date_filed',
            'Insured Company Name': 'insured_company',
            'Paid Indemnity': 'claim_amount',
            'Paid DCC': 'paid_dcc',
            'Outstanding Indemnity': 'outstanding_indemnity',
            'Outstanding DCC': 'outstanding_dcc',
            'Adjuster Notes': 'adjuster_notes'
        }
        
        # Rename columns to standard names
        df = df.rename(columns=column_mapping)
        
        # Clean the data
        df = self.clean_dataframe(df)
        
        documents = []
        
        # Create documents with clean text
        for idx, row in df.iterrows():
            # Build structured content
            content_parts = []
            
            # Add each field in a structured way with the actual data
            content_parts.append(f"Claim ID: {row.get('claim_id', 'Unknown')}")
            content_parts.append(f"Claim Type: {row.get('claim_type', 'Unknown')}")
            content_parts.append(f"Injury Type: {row.get('injury_type', 'Unknown')}")
            content_parts.append(f"Loss State: {row.get('loss_state', 'Unknown')}")
            content_parts.append(f"Date Filed: {row.get('date_filed', 'Unknown')}")
            content_parts.append(f"Insured Company: {row.get('insured_company', 'Unknown')}")
            
            # Handle all financial amounts with proper formatting
            if 'claim_amount' in row:
                paid_indemnity = float(row['claim_amount']) if row['claim_amount'] else 0
                paid_dcc = float(row.get('paid_dcc', 0)) if row.get('paid_dcc') else 0
                outstanding_indemnity = float(row.get('outstanding_indemnity', 0)) if row.get('outstanding_indemnity') else 0
                outstanding_dcc = float(row.get('outstanding_dcc', 0)) if row.get('outstanding_dcc') else 0
                
                content_parts.append(f"Paid Indemnity: ${paid_indemnity:,.2f}")
                content_parts.append(f"Paid DCC: ${paid_dcc:,.2f}")
                content_parts.append(f"Outstanding Indemnity: ${outstanding_indemnity:,.2f}")
                content_parts.append(f"Outstanding DCC: ${outstanding_dcc:,.2f}")
                content_parts.append(f"Total Exposure: ${(paid_indemnity + paid_dcc + outstanding_indemnity + outstanding_dcc):,.2f}")
            
            # Add other financial information
            if 'paid_dcc' in row:
                dcc = float(row['paid_dcc']) if row['paid_dcc'] else 0
                content_parts.append(f"Paid DCC: ${dcc:,.2f}")
            
            if 'outstanding_indemnity' in row:
                outstanding = float(row['outstanding_indemnity']) if row['outstanding_indemnity'] else 0
                if outstanding > 0:
                    content_parts.append(f"Outstanding Indemnity: ${outstanding:,.2f}")
            
            # Add description if available and meaningful
            if 'description' in row and str(row['description']) not in ['Unknown', 'No description provided', 'nan']:
                content_parts.append(f"Loss Description: {row['description']}")
            
            # Add adjuster notes if available and meaningful
            if 'adjuster_notes' in row and str(row['adjuster_notes']) not in ['Unknown', 'No notes available', 'nan']:
                content_parts.append(f"Adjuster Notes: {row['adjuster_notes']}")
            
            # Join with newlines
            content = "\n".join(content_parts)
            
            # Create document with metadata
            metadata = {
                'claim_id': str(row.get('claim_id', '')),
                'claim_type': str(row.get('claim_type', '')),
                'injury_type': str(row.get('injury_type', '')),
                'loss_state': str(row.get('loss_state', '')),
                'date_filed': str(row.get('date_filed', '')),
                'insured_company': str(row.get('insured_company', '')),
                'amount': float(row.get('claim_amount', 0)) + float(row.get('outstanding_indemnity', 0)),  # Total indemnity exposure
                'paid_indemnity': float(row.get('claim_amount', 0)),
                'paid_dcc': float(row.get('paid_dcc', 0)),
                'outstanding_indemnity': float(row.get('outstanding_indemnity', 0)),
                'outstanding_dcc': float(row.get('outstanding_dcc', 0)),
                'total_exposure': (float(row.get('claim_amount', 0)) + float(row.get('paid_dcc', 0)) + 
                                 float(row.get('outstanding_indemnity', 0)) + float(row.get('outstanding_dcc', 0))),
                'source': 'csv',
                'row_index': idx
            }
            
            # Clean metadata
            for key, value in metadata.items():
                if isinstance(value, str) and value.lower() == 'nan':
                    metadata[key] = ''
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        print(f"   ✓ Created {len(documents)} clean documents")
        return documents
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Split documents into optimal chunks."""
        print(f"📏 Chunking documents (size={chunk_size}, overlap={chunk_overlap})")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " "],
            length_function=len
        )
        
        chunks = []
        for doc in documents:
            # If document is small, keep it whole
            if len(doc.page_content) <= chunk_size:
                chunks.append(doc)
            else:
                # Split larger documents
                split_docs = text_splitter.split_documents([doc])
                chunks.extend(split_docs)
        
        print(f"   ✓ Created {len(chunks)} chunks")
        return chunks
    
    def create_bm25_index(self, documents: List[Document]):
        """Create BM25 index for keyword search."""
        print("🔍 Creating BM25 index...")
        
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in documents:
            # Simple tokenization - lowercase and split
            tokens = doc.page_content.lower().split()
            tokenized_docs.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
        print("   ✓ BM25 index created")
    
    def create_vector_store(self, documents: List[Document]):
        """Create vector store with metadata."""
        print("🧮 Creating vector store...")
        
        # Create collection
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=1536,  # text-embedding-3-small
                distance=Distance.COSINE
            )
        )
        
        # Add documents with metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Create embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add to Qdrant
        points = []
        for i, (embedding, text, metadata) in enumerate(zip(embeddings, texts, metadatas)):
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": text,
                    **metadata
                }
            )
            points.append(point)
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"   ✓ Added {len(points)} vectors to store")
    
    def ingest_data(self, csv_path: str):
        """Complete ingestion pipeline with all improvements."""
        print("\n🚀 Starting enhanced data ingestion...")
        
        # Create clean documents
        documents = self.create_documents_from_csv(csv_path)
        
        # Chunk documents with optimal size
        chunks = self.chunk_documents(documents, chunk_size=1000, chunk_overlap=200)
        
        # Create both BM25 and vector indices
        self.create_bm25_index(chunks)
        self.create_vector_store(chunks)
        
        print("\n✅ Enhanced ingestion complete!")
    
    def hybrid_search(self, query: str, k: int = 7) -> List[Dict]:
        """Hybrid search combining BM25 and vector search with claim number detection."""
        
        # Enhanced claim number detection
        detected_claim = self._detect_claim_number(query)
        if detected_claim:
            print(f"🔍 Detected claim number in query: {detected_claim}")
            return self.search_by_claim_number(detected_claim)
        
        # Get more candidates for reranking
        candidates_k = k * 2 if self.use_reranking else k
        
        # 1. BM25 Search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top BM25 results
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:candidates_k]
        bm25_results = []
        
        for idx in bm25_top_indices:
            if bm25_scores[idx] > 0:
                bm25_results.append({
                    'content': self.documents[idx].page_content,
                    'metadata': self.documents[idx].metadata,
                    'score': float(bm25_scores[idx]),
                    'type': 'bm25'
                })
        
        # 2. Vector Search
        query_embedding = self.embeddings.embed_query(query)
        
        vector_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=candidates_k
        )
        
        vector_search_results = []
        for result in vector_results:
            vector_search_results.append({
                'content': result.payload['text'],
                'metadata': {k: v for k, v in result.payload.items() if k != 'text'},
                'score': result.score,
                'type': 'vector'
            })
        
        # 3. Combine and deduplicate
        all_results = {}
        
        # Add BM25 results
        for result in bm25_results:
            key = result['content'][:100]  # Use first 100 chars as key
            all_results[key] = result
        
        # Add vector results (will override if duplicate)
        for result in vector_search_results:
            key = result['content'][:100]
            if key in all_results:
                # Combine scores
                all_results[key]['score'] = (all_results[key]['score'] + result['score']) / 2
                all_results[key]['type'] = 'hybrid'
            else:
                all_results[key] = result
        
        # Convert back to list
        combined_results = list(all_results.values())
        
        # 4. Rerank if available
        if self.use_reranking and len(combined_results) > 0:
            # Prepare documents for reranking
            docs_to_rerank = [result['content'] for result in combined_results]
            
            # Rerank
            reranked = self.reranker.rerank(query=query, documents=docs_to_rerank, top_n=k)
            
            # Reorder results based on reranking
            final_results = []
            for item in reranked:
                idx = item['index']
                result = combined_results[idx]
                result['rerank_score'] = item['relevance_score']
                final_results.append(result)
            
            return final_results[:k]
        else:
            # Sort by score and return top k
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            return combined_results[:k]
    
    def search_by_claim_number(self, claim_number: str) -> List[Dict]:
        """Enhanced search by claim number with fuzzy matching and validation."""
        print(f"🔍 Searching for claim: {claim_number}")
        
        # Check if documents are loaded
        if not self.documents:
            print("   ⚠️ No documents loaded")
            return []
        
        # Clean and normalize the claim number
        normalized_claim = self._normalize_claim_number(claim_number)
        print(f"   📝 Normalized to: {normalized_claim}")
        
        # 1. Try exact match first
        exact_matches = self._find_exact_matches(normalized_claim)
        if exact_matches:
            print(f"   ✅ Found {len(exact_matches)} exact matches")
            return exact_matches
        
        # 2. Try fuzzy/partial matching
        fuzzy_matches = self._find_fuzzy_matches(normalized_claim)
        if fuzzy_matches:
            print(f"   🎯 Found {len(fuzzy_matches)} fuzzy matches")
            return fuzzy_matches
        
        # 3. Fall back to regular search if no matches
        print(f"   🔍 No direct matches, trying content search...")
        return self._regular_hybrid_search(claim_number, k=5)
    
    def _detect_claim_number(self, query: str) -> str:
        """Detect GL-YYYY-NNNN claim number pattern."""
        import re
        
        query_upper = query.upper().strip()
        
        # Primary pattern: GL-YYYY-NNNN (your exact format)
        gl_pattern = r'\bGL-\d{4}-\d{4}\b'
        match = re.search(gl_pattern, query_upper)
        if match:
            return match.group(0)
        
        # Flexible patterns for user input variations
        patterns = [
            r'\bGL\d{8}\b',           # GL20240024
            r'\b\d{4}-?\d{4}\b',      # 2024-0024 or 20240024
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_upper)
            if match:
                return match.group(0)
        
        # Check if entire query looks like a claim number
        if re.match(r'^GL-?\d{4}-?\d{4}$', query_upper):
            return query_upper
            
        return None
    
    def _normalize_claim_number(self, claim_number: str) -> str:
        """Normalize to GL-YYYY-NNNN format."""
        import re
        
        normalized = claim_number.upper().strip()
        
        # If already in correct format, return as-is
        if re.match(r'^GL-\d{4}-\d{4}$', normalized):
            return normalized
            
        # Handle GL20240024 -> GL-2024-0024
        if re.match(r'^GL\d{8}$', normalized):
            numbers = normalized[2:]  # Remove GL
            return f"GL-{numbers[:4]}-{numbers[4:]}"
            
        # Handle 2024-0024 or 20240024 -> GL-2024-0024
        if re.match(r'^\d{4}-?\d{4}$', normalized):
            if '-' not in normalized:
                numbers = normalized
                return f"GL-{numbers[:4]}-{numbers[4:]}"
            else:
                return f"GL-{normalized}"
        
        # Return as-is if no pattern matches
        return normalized
    
    def _find_exact_matches(self, claim_number: str) -> List[Dict]:
        """Find exact claim number matches."""
        exact_matches = []
        
        for doc in self.documents:
            doc_claim_id = doc.metadata.get('claim_id', '').upper().strip()
            
            # Exact match
            if doc_claim_id == claim_number:
                exact_matches.append({
                    'content': doc.page_content,
                    'score': 1.0,
                    'type': 'exact_match',
                    'metadata': doc.metadata
                })
        
        return exact_matches
    
    def _find_fuzzy_matches(self, claim_number: str) -> List[Dict]:
        """Find fuzzy/partial claim number matches."""
        fuzzy_matches = []
        
        # Fuzzy matching for claim numbers
        import re
        
        for doc in self.documents:
            doc_claim_id = doc.metadata.get('claim_id', '').upper().strip()
            
            if not doc_claim_id:
                continue
            
            # Calculate similarity score
            similarity_score = self._calculate_claim_similarity(claim_number, doc_claim_id)
            
            # Include if similarity is high enough
            if similarity_score > 0.7:
                fuzzy_matches.append({
                    'content': doc.page_content,
                    'score': similarity_score,
                    'type': 'fuzzy_match',
                    'metadata': doc.metadata
                })
        
        # Sort by similarity score
        fuzzy_matches.sort(key=lambda x: x['score'], reverse=True)
        return fuzzy_matches[:5]  # Return top 5 fuzzy matches
    
    def _calculate_claim_similarity(self, claim1: str, claim2: str) -> float:
        """Calculate similarity between two claim numbers."""
        # Remove hyphens for comparison
        clean1 = claim1.replace('-', '').replace('_', '')
        clean2 = claim2.replace('-', '').replace('_', '')
        
        # Exact match
        if clean1 == clean2:
            return 1.0
        
        # Check if one contains the other
        if clean1 in clean2 or clean2 in clean1:
            return 0.9
        
        # Check numeric part similarity (last 4-7 digits)
        import re
        nums1 = re.findall(r'\d+', clean1)
        nums2 = re.findall(r'\d+', clean2)
        
        if nums1 and nums2:
            # Compare last number sequences
            if nums1[-1] == nums2[-1]:  # Same number sequence
                return 0.8
            
            # Similar number sequences (off by 1-3)
            try:
                diff = abs(int(nums1[-1]) - int(nums2[-1]))
                if diff <= 3:
                    return 0.7
            except ValueError:
                pass
        
        return 0.0

    def _regular_hybrid_search(self, query: str, k: int = 7) -> List[Dict]:
        """Regular hybrid search without claim number detection (to avoid recursion)."""
        # Get more candidates for reranking
        candidates_k = k * 2 if self.use_reranking else k
        
        # 1. BM25 Search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top BM25 results
        bm25_indices = np.argsort(bm25_scores)[::-1][:candidates_k]
        bm25_search_results = []
        
        for idx in bm25_indices:
            if bm25_scores[idx] > 0:  # Only include positive scores
                bm25_search_results.append({
                    'content': self.documents[idx].page_content,
                    'score': float(bm25_scores[idx]),
                    'type': 'bm25',
                    'metadata': self.documents[idx].metadata
                })
        
        # 2. Vector Search
        query_vector = self.embeddings.embed_query(query)
        vector_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=candidates_k
        )
        
        vector_search_results = []
        for result in vector_results:
            doc_idx = result.payload.get('doc_index')
            if doc_idx is not None and doc_idx < len(self.documents):
                vector_search_results.append({
                    'content': self.documents[doc_idx].page_content,
                    'score': float(result.score),
                    'type': 'vector',
                    'metadata': self.documents[doc_idx].metadata
                })
        
        # 3. Combine results
        all_results = {}
        
        # Add BM25 results first
        for result in bm25_search_results:
            key = result['content'][:100]  # Use first 100 chars as key
            all_results[key] = result
        
        # Add vector results (will override if duplicate)
        for result in vector_search_results:
            key = result['content'][:100]
            if key in all_results:
                # Combine scores
                all_results[key]['score'] = (all_results[key]['score'] + result['score']) / 2
                all_results[key]['type'] = 'hybrid'
            else:
                all_results[key] = result
        
        # Convert back to list and sort
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:k]
    
    def search_with_metadata_filter(self, query: str, filters: Dict = None, k: int = 7) -> List[Dict]:
        """Search with optional metadata filtering."""
        if not filters:
            return self.hybrid_search(query, k)
        
        # For now, apply filters after search (can be optimized)
        results = self.hybrid_search(query, k * 2)
        
        filtered_results = []
        for result in results:
            match = True
            for key, value in filters.items():
                if key in result['metadata']:
                    if result['metadata'][key] != value:
                        match = False
                        break
            
            if match:
                filtered_results.append(result)
        
        return filtered_results[:k]

# Update evaluate_rag.py to use enhanced RAG
def evaluate_enhanced_rag():
    """Evaluate the enhanced RAG system."""
    from evaluate_rag import SimpleRAGASEvaluator
    import pandas as pd
    
    print("\n🎯 EVALUATING ENHANCED RAG SYSTEM")
    print("=" * 60)
    
    # Use enhanced RAG instead of simple RAG
    class EnhancedEvaluator(SimpleRAGASEvaluator):
        def __init__(self):
            super().__init__()
            # Replace simple RAG with enhanced RAG
            self.rag = EnhancedInsuranceRAG(use_reranking=True)
        
        def evaluate_naive_retriever(self, test_df: pd.DataFrame, csv_path: str) -> tuple:
            """Override to use enhanced search with k=7."""
            print(f"\n🔍 Evaluating Enhanced RAG on {len(test_df)} questions...")
            
            # Load data with enhanced ingestion
            print("   Loading data with enhanced ingestion...")
            self.rag.ingest_data(csv_path)
            
            # Prepare evaluation data
            eval_data = []
            
            for idx, row in test_df.iterrows():
                question = row['question']
                ground_truth = row.get('ground_truth', 'No ground truth provided')
                
                # Use enhanced hybrid search with k=7
                search_results = self.rag.hybrid_search(question, k=7)
                contexts = [result["content"] for result in search_results]
                
                # Generate answer with better prompt
                context_text = "\n\n".join(contexts)
                prompt = f"""You are an insurance claims expert. Based ONLY on the context provided below, answer the question accurately.
If the information is not in the context, say "I cannot find this information in the provided context."

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
            
            # Rest of evaluation remains the same
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            
            dataset = Dataset.from_list(eval_data)
            
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
    
    # Run evaluation
    csv_path = "data/insurance_claims.csv"
    evaluator = EnhancedEvaluator()
    
    # Generate or load test set
    test_df = evaluator.generate_test_set(csv_path, 20)
    test_df.to_csv("enhanced_test_questions.csv", index=False)
    
    # Evaluate
    results, eval_data = evaluator.evaluate_naive_retriever(test_df, csv_path)
    
    # Display results
    summary_df, metrics = evaluator.display_results(results, eval_data)
    
    # Analyze performance
    evaluator.analyze_performance(metrics)
    
    print("\n\n✅ Enhanced Evaluation Complete!")

if __name__ == "__main__":
    # Test the enhanced RAG
    print("\n🧪 Testing Enhanced RAG System")
    print("=" * 60)
    
    rag = EnhancedInsuranceRAG(use_reranking=True)
    rag.ingest_data("data/insurance_claims.csv")
    
    # Test queries including claim number searches
    test_queries = [
        "GL-2024-0024",  # Exact claim number test
        "What is the claim amount for CLM-001?",
        "Show me all pending claims", 
        "What are the high value property damage claims?",
        "GL-2024-0025",  # Another exact claim test
        "Find claims in Minnesota"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        results = rag.hybrid_search(query, k=5)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results[:2], 1):
            print(f"\n{i}. Type: {result['type']} | Score: {result.get('rerank_score', result['score']):.3f}")
            print(f"   {result['content'][:150]}...")
    
    # Run full evaluation
    print("\n" + "="*60)
    evaluate_enhanced_rag()