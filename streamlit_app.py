#!/usr/bin/env python3
"""
Streamlit UI for Insurance Claims Adjudication System
Compact interface with RAG + Tavily search integration
"""

import streamlit as st
from enhanced_rag import EnhancedInsuranceRAG
from adjuster_agent import ClaimsAdjusterAgent
import os

# Page configuration
st.set_page_config(
    page_title="Claims Intelligence System",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.adjuster_agent = None
    st.session_state.data_loaded = False

# Custom CSS for compact styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .search-result {
        border-left: 3px solid #667eea;
        background: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .claim-id {
        color: #667eea;
        font-weight: bold;
    }
    .amount {
        color: #28a745;
        font-weight: bold;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .source-rag { background: #e3f2fd; color: #1976d2; }
    .source-tavily { background: #f3e5f5; color: #7b1fa2; }
    .source-agent { background: #e8f5e9; color: #388e3c; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚖️ Claims Intelligence System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">RAG Search + AI Legal Research</p>', unsafe_allow_html=True)

# Initialize systems automatically
if not st.session_state.data_loaded:
    with st.spinner("🚀 Initializing systems..."):
        try:
            # Initialize RAG
            st.session_state.rag_system = EnhancedInsuranceRAG(use_reranking=False)
            st.session_state.rag_system.ingest_data("data/insurance_claims.csv")
            
            # Initialize Agent (only if Tavily key exists)
            if os.getenv("TAVILY_API_KEY"):
                st.session_state.adjuster_agent = ClaimsAdjusterAgent()
                st.success("✅ Systems ready: Claims Database + Legal Research")
            else:
                st.warning("⚠️ Claims Database ready. Add TAVILY_API_KEY for legal research.")
                
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"❌ Initialization error: {str(e)}")

# Main search interface
if st.session_state.data_loaded:
    # Search input with options
    query = st.text_input(
        "🔍 Search claims or ask legal questions:",
        placeholder="e.g., GL-2024-0024, slip and fall Texas liability, auto accident settlements",
        key="search_query"
    )
    
    # Search options
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        search_mode = st.selectbox(
            "Search Mode",
            ["Smart Search (RAG + Legal)", "Claims Only (RAG)", "Legal Research Only"],
            help="Smart Search combines internal claims with external legal research"
        )
    with col2:
        if "claim" in query.lower() or "-" in query:
            search_type = "exact"
        else:
            search_type = "hybrid"
    with col3:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Quick examples
    example_col1, example_col2, example_col3 = st.columns(3)
    with example_col1:
        if st.button("📋 GL-2024-0024", use_container_width=True):
            query = "GL-2024-0024"
            search_clicked = True
    with example_col2:
        if st.button("⚖️ Texas slip liability", use_container_width=True):
            query = "slip and fall liability rules Texas"
            search_clicked = True
    with example_col3:
        if st.button("🚗 Auto settlements", use_container_width=True):
            query = "auto accident soft tissue injury settlements"
            search_clicked = True
    
    # Perform search
    if search_clicked and query:
        all_results = []
        
        # 1. RAG Search (if enabled)
        if search_mode in ["Smart Search (RAG + Legal)", "Claims Only (RAG)"]:
            with st.spinner("🔍 Searching claims database..."):
                try:
                    if search_type == "exact":
                        rag_results = st.session_state.rag_system.search_by_claim_number(query)
                    else:
                        rag_results = st.session_state.rag_system.hybrid_search(query, k=5)
                    
                    for result in rag_results:
                        result['source'] = 'rag'
                        all_results.append(result)
                except Exception as e:
                    st.error(f"RAG search error: {str(e)}")
        
        # 2. Agent/Legal Search (if enabled and available)
        if search_mode in ["Smart Search (RAG + Legal)", "Legal Research Only"] and st.session_state.adjuster_agent:
            with st.spinner("⚖️ Researching legal requirements..."):
                try:
                    # Get agent analysis
                    agent_response = st.session_state.adjuster_agent.process_query(query)
                    
                    # Add as a special result
                    all_results.append({
                        'source': 'agent',
                        'content': agent_response,
                        'metadata': {
                            'claim_id': 'Legal Analysis',
                            'claim_type': 'AI Research',
                            'amount': 0
                        },
                        'score': 1.0,
                        'type': 'legal_research'
                    })
                except Exception as e:
                    if "TAVILY_API_KEY" in str(e):
                        st.warning("⚠️ Legal research unavailable. Add TAVILY_API_KEY to .env file.")
                    else:
                        st.error(f"Agent error: {str(e)}")
        
        # Display results
        if all_results:
            st.success(f"✅ Found {len(all_results)} results")
            
            # Separate agent results from RAG results
            agent_results = [r for r in all_results if r['source'] == 'agent']
            rag_results = [r for r in all_results if r['source'] == 'rag']
            
            # Show agent/legal analysis first if available
            if agent_results:
                st.markdown("### ⚖️ Legal Analysis & Recommendations")
                for result in agent_results:
                    with st.container():
                        st.markdown(f'<span class="source-badge source-agent">AI LEGAL RESEARCH</span>', unsafe_allow_html=True)
                        st.markdown(result['content'])
                        st.markdown("---")
            
            # Show RAG results
            if rag_results:
                st.markdown("### 📋 Claims Database Results")
                for i, result in enumerate(rag_results[:5], 1):
                    metadata = result.get('metadata', {})
                    with st.container():
                        st.markdown(f"""
                        <div class="search-result">
                            <span class="source-badge source-rag">CLAIMS DB</span>
                            <span class="claim-id">Result {i}: {metadata.get('claim_id', 'Unknown')}</span>
                            <br>
                            <strong>Type:</strong> {metadata.get('claim_type', 'Unknown')} | 
                            <strong>Total Exposure:</strong> <span class="amount">${metadata.get('total_exposure', 0):,.2f}</span> | 
                            <strong>State:</strong> {metadata.get('loss_state', 'Unknown')}
                            <br>
                            <strong>Match Score:</strong> {result['score']:.3f} ({result['type']})
                            <p style="margin-top: 0.5rem;">{result['content'][:400]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show more results option
                if len(rag_results) > 5:
                    with st.expander(f"Show {len(rag_results) - 5} more results"):
                        for i, result in enumerate(rag_results[5:], 6):
                            metadata = result.get('metadata', {})
                            st.markdown(f"**Result {i}: {metadata.get('claim_id')}** - ${metadata.get('amount', 0):,.2f}")
                            st.text(result['content'][:200] + "...")
        else:
            st.warning("No results found. Try adjusting your search terms.")
    
    # Help section
    with st.expander("💡 Search Tips"):
        st.markdown("""
        **Search Modes:**
        - **Smart Search**: Combines claims database with AI legal research
        - **Claims Only**: Fast search in internal claims database
        - **Legal Research**: AI-powered state law and regulation search
        
        **Example Queries:**
        - Exact claim: `GL-2024-0024`
        - Legal question: `slip and fall liability Texas`
        - Pattern search: `auto accidents over $50000`
        - Compliance: `statute of limitations property damage Florida`
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9rem;'>
    Claims Intelligence: RAG + AI Legal Research | Powered by GPT-3.5 & Tavily
</div>
""", unsafe_allow_html=True)