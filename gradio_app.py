#!/usr/bin/env python3
"""
Gradio UI for Insurance Claims Adjudication System
Simple, fast interface for claims adjusters
"""

import gradio as gr
import pandas as pd
from enhanced_rag import EnhancedInsuranceRAG

# Initialize RAG system
rag_system = None

def initialize_system():
    """Initialize the RAG system."""
    global rag_system
    try:
        rag_system = EnhancedInsuranceRAG(use_reranking=False)
        rag_system.ingest_data("data/insurance_claims.csv")
        return "✅ System initialized successfully! You can now search claims."
    except Exception as e:
        return f"❌ Error initializing system: {str(e)}"

def search_claims(query, search_type="Hybrid Search", max_results=5):
    """Search claims based on user input."""
    if not rag_system:
        return "❌ Please initialize the system first.", ""
    
    if not query.strip():
        return "❌ Please enter a search query.", ""
    
    try:
        # Perform search
        if search_type == "Exact Claim Number":
            results = rag_system.search_by_claim_number(query)
        else:
            results = rag_system.hybrid_search(query, k=max_results)
        
        if not results:
            return "No results found. Try adjusting your search terms.", ""
        
        # Format results
        formatted_results = []
        summary_data = []
        
        for i, result in enumerate(results[:max_results], 1):
            metadata = result.get('metadata', {})
            
            # Format individual result
            result_text = f"""
**Result {i}: {metadata.get('claim_id', 'Unknown')}**
- **Type:** {metadata.get('claim_type', 'Unknown')}
- **Amount:** ${metadata.get('amount', 0):,.2f}
- **State:** {metadata.get('loss_state', 'Unknown')}
- **Score:** {result['score']:.3f} ({result['type']})

{result['content'][:400]}...

---
"""
            formatted_results.append(result_text)
            
            # Collect data for summary
            summary_data.append({
                'Claim ID': metadata.get('claim_id', 'Unknown'),
                'Type': metadata.get('claim_type', 'Unknown'),
                'Amount': f"${metadata.get('amount', 0):,.2f}",
                'State': metadata.get('loss_state', 'Unknown'),
                'Score': f"{result['score']:.3f}"
            })
        
        # Create summary table
        summary_df = pd.DataFrame(summary_data)
        summary_html = summary_df.to_html(index=False, escape=False, classes="table table-striped")
        
        return "\n".join(formatted_results), summary_html
        
    except Exception as e:
        return f"❌ Search error: {str(e)}", ""

def get_quick_search(search_type):
    """Return predefined search queries."""
    quick_searches = {
        "Auto Claims": "auto accident claims",
        "Property Claims": "property damage claims", 
        "High Value Claims": "claims over $100000",
        "Pending Claims": "pending claims status",
        "Slip and Fall": "slip and fall incidents",
        "Medical Claims": "bodily injury medical claims"
    }
    return quick_searches.get(search_type, "")

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.claim-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.table {
    border-collapse: collapse;
    width: 100%;
}
.table th, .table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
.table th {
    background-color: #f2f2f2;
    font-weight: bold;
}
.table-striped tr:nth-child(even) {
    background-color: #f9f9f9;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Claims Adjuster Intelligence") as app:
    
    # Header
    gr.HTML("""
    <div class="claim-header">
        <h1>⚖️ Claims Adjuster Intelligence System</h1>
        <p>Professional claims search and analysis powered by AI</p>
    </div>
    """)
    
    # System initialization
    with gr.Row():
        with gr.Column(scale=2):
            init_btn = gr.Button("🚀 Initialize System", variant="primary")
            init_status = gr.Textbox(label="System Status", interactive=False)
    
    # Main search interface
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter claim number (e.g., GL-2024-0024) or search terms (e.g., auto accident claims)",
                lines=2
            )
            
            with gr.Row():
                search_type = gr.Dropdown(
                    choices=["Hybrid Search", "Exact Claim Number"],
                    value="Hybrid Search",
                    label="Search Type"
                )
                max_results = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Max Results"
                )
            
            search_btn = gr.Button("🔍 Search Claims", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Quick Searches")
            quick_search_type = gr.Dropdown(
                choices=["Auto Claims", "Property Claims", "High Value Claims", 
                        "Pending Claims", "Slip and Fall", "Medical Claims"],
                label="Quick Search"
            )
            quick_search_btn = gr.Button("🚀 Quick Search")
    
    # Results display
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Search Results")
            results_output = gr.Markdown()
        
        with gr.Column(scale=1):
            gr.Markdown("### Results Summary")
            summary_output = gr.HTML()
    
    # Event handlers
    init_btn.click(
        fn=initialize_system,
        outputs=init_status
    )
    
    search_btn.click(
        fn=search_claims,
        inputs=[query_input, search_type, max_results],
        outputs=[results_output, summary_output]
    )
    
    quick_search_btn.click(
        fn=get_quick_search,
        inputs=quick_search_type,
        outputs=query_input
    )
    
    # Example queries
    gr.Examples(
        examples=[
            ["GL-2024-0024", "Exact Claim Number"],
            ["auto accident claims", "Hybrid Search"],
            ["slip and fall incidents", "Hybrid Search"],
            ["claims over $50000", "Hybrid Search"],
            ["property damage Minnesota", "Hybrid Search"]
        ],
        inputs=[query_input, search_type],
        label="Example Searches"
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )