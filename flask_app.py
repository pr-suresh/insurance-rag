#!/usr/bin/env python3
"""
Flask Web Dashboard for Insurance Claims Adjudication System
Lightweight, customizable interface
"""

from flask import Flask, render_template, request, jsonify
import json
from enhanced_rag import EnhancedInsuranceRAG

app = Flask(__name__)

# Global RAG system
rag_system = None

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the RAG system."""
    global rag_system
    try:
        rag_system = EnhancedInsuranceRAG(use_reranking=False)
        rag_system.ingest_data("data/insurance_claims.csv")
        return jsonify({'status': 'success', 'message': 'System initialized successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/search', methods=['POST'])
def search():
    """Search claims endpoint."""
    if not rag_system:
        return jsonify({'status': 'error', 'message': 'System not initialized'})
    
    data = request.get_json()
    query = data.get('query', '')
    search_type = data.get('search_type', 'hybrid')
    max_results = data.get('max_results', 5)
    
    if not query:
        return jsonify({'status': 'error', 'message': 'Query cannot be empty'})
    
    try:
        if search_type == 'exact':
            results = rag_system.search_by_claim_number(query)
        else:
            results = rag_system.hybrid_search(query, k=max_results)
        
        # Format results for JSON response
        formatted_results = []
        for result in results:
            formatted_results.append({
                'claim_id': result['metadata'].get('claim_id', 'Unknown'),
                'claim_type': result['metadata'].get('claim_type', 'Unknown'),
                'amount': result['metadata'].get('amount', 0),
                'state': result['metadata'].get('loss_state', 'Unknown'),
                'score': result['score'],
                'search_type': result['type'],
                'content': result['content'][:500] + '...' if len(result['content']) > 500 else result['content']
            })
        
        return jsonify({
            'status': 'success',
            'results': formatted_results,
            'total': len(formatted_results)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)