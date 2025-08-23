#!/usr/bin/env python3
"""
Enhanced Streamlit UI for Insurance Claims Intelligence System
Includes new Pattern Analysis and Financial Analytics capabilities
"""

import streamlit as st
from enhanced_rag import EnhancedInsuranceRAG
from enhanced_adjuster_agent import EnhancedClaimsAdjusterAgent
from pattern_analysis_agent import PatternAnalysisAgent
from financial_agent import FinancialAnalysisAgent
import pandas as pd
import json
import os

# Helper functions (defined early to avoid NameError)
def format_rag_results(results, include_metadata=True):
    """Format RAG search results"""
    if not results:
        return "No relevant claims found."
    
    output = "### Search Results\n\n"
    for i, result in enumerate(results[:5], 1):
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        score = result.get('score', 0)
        search_type = result.get('type', 'unknown')
        
        if include_metadata and metadata:
            output += f"**Result {i}: {metadata.get('Claim Number', 'Unknown')}**\n"
            output += f"- Type: {metadata.get('Claim Feature', 'N/A')}\n"
            output += f"- State: {metadata.get('Loss state', 'N/A')}\n"
            output += f"- Amount: ${metadata.get('Paid Indemnity', 0):,.0f}\n"
            output += f"- Search: {search_type} (score: {score:.3f})\n"
            output += f"- Summary: {content[:200]}...\n\n"
        else:
            output += f"**Result {i}**: {content[:300]}...\n\n"
    
    return output

def generate_summary(json_data):
    """Generate a narrative summary from JSON data"""
    if isinstance(json_data, dict):
        if 'total_exposure' in json_data:
            return f"Total exposure: ${json_data['total_exposure']:,.0f} across {json_data.get('total_claims', 'N/A')} claims"
        elif 'predictions' in json_data:
            return f"Predicted outcome based on {json_data['predictions'].get('based_on_claims', 'N/A')} similar claims"
        elif 'risk_analysis' in json_data:
            return f"Identified {len(json_data['risk_analysis'].get('common_risk_factors', {}))} key risk factors"
    return "Analysis complete. See detailed results above."

# Page configuration
st.set_page_config(
    page_title="Enhanced Claims Intelligence System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.enhanced_agent = None
    st.session_state.pattern_agent = None
    st.session_state.financial_agent = None
    st.session_state.data_loaded = False

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .feature-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
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
    .pattern-badge {
        background: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .financial-badge {
        background: #d1ecf1;
        color: #0c5460;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚖️ Enhanced Claims Intelligence System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">AI-Powered Claims Analysis with Pattern Recognition & Financial Analytics</p>', unsafe_allow_html=True)

# Initialize systems automatically
if not st.session_state.data_loaded:
    with st.spinner("🚀 Initializing enhanced systems..."):
        try:
            # Initialize RAG
            st.session_state.rag_system = EnhancedInsuranceRAG(use_reranking=False)
            st.session_state.rag_system.ingest_data("data/insurance_claims.csv")
            
            # Initialize Enhanced Agent
            st.session_state.enhanced_agent = EnhancedClaimsAdjusterAgent()
            
            # Initialize specialized agents
            st.session_state.pattern_agent = PatternAnalysisAgent()
            st.session_state.financial_agent = FinancialAnalysisAgent()
            
            st.session_state.data_loaded = True
            st.success("✅ All systems ready: Claims Database + Pattern Analysis + Financial Analytics + Legal Research")
        except Exception as e:
            st.error(f"❌ Initialization error: {str(e)}")

# Sidebar with feature selection
with st.sidebar:
    st.markdown("## 🎯 Analysis Mode")
    analysis_mode = st.radio(
        "Select analysis type:",
        ["🔍 Smart Search (All Capabilities)",
         "📊 Pattern Analysis",
         "💰 Financial Analytics",
         "📚 Claims Database Only",
         "⚖️ Legal Research Only"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("## 📌 Quick Actions")
    
    # Quick action buttons
    if st.button("📈 Generate Financial Summary"):
        with st.spinner("Generating financial summary..."):
            response = st.session_state.enhanced_agent.process_query("Generate a comprehensive financial summary for all claims")
            st.session_state.last_response = response
    
    if st.button("🎯 Identify Top Risk Patterns"):
        with st.spinner("Analyzing risk patterns..."):
            response = st.session_state.enhanced_agent.process_query("Identify the top risk indicators and patterns across all claims")
            st.session_state.last_response = response
    
    if st.button("💡 Settlement Efficiency Report"):
        with st.spinner("Analyzing settlement efficiency..."):
            response = st.session_state.enhanced_agent.process_query("Analyze settlement efficiency and identify opportunities for improvement")
            st.session_state.last_response = response
    
    st.markdown("---")
    st.markdown("## 🛠️ New Features")
    st.markdown("""
    **Pattern Analysis:**
    - Claim pattern detection
    - Risk indicator identification
    - Outcome prediction
    - Trend analysis
    
    **Financial Analytics:**
    - Exposure calculation
    - Reserve adequacy
    - Cost driver analysis
    - Settlement efficiency
    """)

# Main content area
if st.session_state.data_loaded:
    # Feature cards for new capabilities
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Pattern Analysis Engine</h3>
            <p>Discover hidden patterns in your claims data:</p>
            <ul>
                <li>Identify high-risk claim patterns</li>
                <li>Predict claim outcomes</li>
                <li>Analyze settlement trends</li>
                <li>Detect fraud indicators</li>
            </ul>
            <span class="pattern-badge">AI-Powered</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💰 Financial Analytics Suite</h3>
            <p>Comprehensive financial insights:</p>
            <ul>
                <li>Calculate total exposure</li>
                <li>Analyze reserve adequacy</li>
                <li>Identify cost drivers</li>
                <li>Track settlement efficiency</li>
            </ul>
            <span class="financial-badge">Real-time Analysis</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Search interface
    st.markdown("---")
    st.markdown("## 🔍 Intelligent Query Interface")
    
    # Example queries based on mode
    example_queries = {
        "🔍 Smart Search (All Capabilities)": [
            "What patterns exist in slip and fall claims and what's our financial exposure?",
            "Analyze product liability claims in Texas and predict outcomes",
            "Generate a comprehensive analysis of construction defect claims"
        ],
        "📊 Pattern Analysis": [
            "What patterns do you see in slip and fall claims?",
            "Identify risk indicators for retail locations",
            "Predict outcome for a bodily injury claim in California"
        ],
        "💰 Financial Analytics": [
            "Calculate total exposure in California",
            "Generate cost driver report for top 10 drivers",
            "Analyze reserve adequacy for open claims"
        ],
        "📚 Claims Database Only": [
            "Find claims similar to GL-2024-0024",
            "Search for slip and fall settlements over $50,000",
            "Show recent product liability claims"
        ],
        "⚖️ Legal Research Only": [
            "Texas premises liability requirements",
            "California product liability statute of limitations",
            "Florida insurance bad faith laws"
        ]
    }
    
    # Show relevant examples
    with st.expander("💡 Example Queries"):
        examples = example_queries.get(analysis_mode, [])
        for ex in examples:
            st.code(ex)
    
    # Main query input
    query = st.text_area(
        "Enter your query:",
        placeholder="Ask about claims, patterns, financial metrics, or legal requirements...",
        height=100,
        key="main_query"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            search_k = st.slider("Number of results", 1, 10, 5)
        with col2:
            include_metadata = st.checkbox("Show metadata", value=True)
        with col3:
            export_results = st.checkbox("Enable export", value=False)
    
    # Search button
    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if query:
            with st.spinner(f"Processing with {analysis_mode}..."):
                try:
                    # Route to appropriate handler based on mode
                    if analysis_mode == "🔍 Smart Search (All Capabilities)":
                        response = st.session_state.enhanced_agent.process_query(query)
                    
                    elif analysis_mode == "📊 Pattern Analysis":
                        # Use pattern analysis tools directly
                        if "pattern" in query.lower():
                            tool = st.session_state.pattern_agent.tools[0]  # analyze_claim_patterns_by_type
                        elif "risk" in query.lower():
                            tool = st.session_state.pattern_agent.tools[1]  # identify_risk_indicators
                        elif "predict" in query.lower():
                            tool = st.session_state.pattern_agent.tools[2]  # predict_claim_outcome
                        else:
                            tool = st.session_state.pattern_agent.tools[3]  # analyze_settlement_trends
                        
                        # Extract parameters from query (simplified)
                        response = tool.invoke({"query": query})
                    
                    elif analysis_mode == "💰 Financial Analytics":
                        # Use financial analysis tools directly
                        if "exposure" in query.lower():
                            tool = st.session_state.financial_agent.tools[0]  # calculate_total_exposure
                        elif "reserve" in query.lower():
                            tool = st.session_state.financial_agent.tools[1]  # analyze_reserve_adequacy
                        elif "cost driver" in query.lower():
                            tool = st.session_state.financial_agent.tools[2]  # generate_cost_driver_report
                        elif "efficiency" in query.lower():
                            tool = st.session_state.financial_agent.tools[3]  # calculate_settlement_efficiency
                        else:
                            tool = st.session_state.financial_agent.tools[4]  # generate_financial_summary
                        
                        response = tool.invoke({})
                    
                    elif analysis_mode == "📚 Claims Database Only":
                        results = st.session_state.rag_system.hybrid_search(query, k=search_k)
                        response = format_rag_results(results, include_metadata)
                    
                    elif analysis_mode == "⚖️ Legal Research Only":
                        # Extract state from query if possible
                        states = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]
                        detected_state = None
                        for state in states:
                            if state in query.upper():
                                detected_state = state
                                break
                        
                        if detected_state:
                            response = f"Legal research for {detected_state}: {query}\n\n[Tavily search would be performed here]"
                        else:
                            response = "Please specify a state for legal research (e.g., 'California premises liability')"
                    
                    # Display results
                    st.markdown("### 📋 Analysis Results")
                    
                    # Format JSON responses nicely
                    if response.startswith('{'):
                        try:
                            json_data = json.loads(response)
                            st.json(json_data)
                            
                            # Also provide narrative summary
                            st.markdown("### 📝 Summary")
                            st.info(generate_summary(json_data))
                        except:
                            st.markdown(response)
                    else:
                        st.markdown(response)
                    
                    # Export option
                    if export_results:
                        st.download_button(
                            label="📥 Download Results",
                            data=response,
                            file_name="analysis_results.txt",
                            mime="text/plain"
                        )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query")
    
    # Display last response from quick actions
    if hasattr(st.session_state, 'last_response'):
        st.markdown("### 📋 Quick Action Results")
        st.markdown(st.session_state.last_response)


# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #888; font-size: 0.9rem;">
Enhanced Claims Intelligence System v2.0 | Powered by AI Pattern Recognition & Financial Analytics
</p>
""", unsafe_allow_html=True)