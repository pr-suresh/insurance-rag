# smart_query.py - Automatically chooses the best retriever for each query
# No more manual selection - the system decides!

import os
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Import your existing hybrid search
from hybrid_query import SimpleHybridSearch

load_dotenv()

class SmartRetrieverRouter:
    """Automatically routes queries to the best retriever"""
    
    def __init__(self):
        self.hybrid_search = SimpleHybridSearch()
        
        # Define patterns for each retriever type
        self.patterns = {
            'bm25': {
                'claim_id': r'CLM[-\s]?\d+',
                'exact_amount': r'\$[\d,]+\.?\d*|\d+\s*dollars?',
                'status_keywords': ['pending', 'approved', 'denied', 'closed'],
                'exact_dates': r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}',
                'short_query': lambda q: len(q.split()) <= 2
            },
            'vector': {
                'question_words': ['what', 'why', 'how', 'which', 'when', 'who'],
                'conceptual_words': ['similar', 'like', 'risk', 'suspicious', 'unusual', 
                                   'pattern', 'trend', 'analyze', 'summarize', 'explain'],
                'comparative': ['high', 'low', 'expensive', 'cheap', 'large', 'small'],
                'long_query': lambda q: len(q.split()) > 6
            },
            'hybrid': {
                'mixed_indicators': ['and', 'with', 'over', 'under', 'between'],
                'complex_queries': lambda q: len(q.split()) >= 3 and len(q.split()) <= 6
            }
        }
        
        print("✅ Smart Router initialized - it decides which retriever to use!")
    
    def analyze_query(self, query: str) -> Tuple[str, Dict[str, float]]:
        """Analyze query and return best retriever with confidence scores"""
        query_lower = query.lower()
        
        scores = {
            'bm25': 0.0,
            'vector': 0.0,
            'hybrid': 0.0
        }
        
        # Check BM25 indicators
        if re.search(self.patterns['bm25']['claim_id'], query, re.IGNORECASE):
            scores['bm25'] += 0.9  # Very strong signal
            
        if re.search(self.patterns['bm25']['exact_amount'], query):
            scores['bm25'] += 0.7
            
        if any(status in query_lower for status in self.patterns['bm25']['status_keywords']):
            scores['bm25'] += 0.5
            
        if re.search(self.patterns['bm25']['exact_dates'], query):
            scores['bm25'] += 0.6
            
        if self.patterns['bm25']['short_query'](query):
            scores['bm25'] += 0.3
        
        # Check Vector indicators
        if any(q_word in query_lower for q_word in self.patterns['vector']['question_words']):
            scores['vector'] += 0.6
            
        if any(concept in query_lower for concept in self.patterns['vector']['conceptual_words']):
            scores['vector'] += 0.8
            
        if any(comp in query_lower for comp in self.patterns['vector']['comparative']):
            scores['vector'] += 0.5
            
        if self.patterns['vector']['long_query'](query):
            scores['vector'] += 0.4
        
        # Check Hybrid indicators
        if any(mixed in query_lower for mixed in self.patterns['hybrid']['mixed_indicators']):
            scores['hybrid'] += 0.5
            
        if self.patterns['hybrid']['complex_queries'](query):
            scores['hybrid'] += 0.3
        
        # If both BM25 and Vector have good scores, prefer hybrid
        if scores['bm25'] > 0.5 and scores['vector'] > 0.5:
            scores['hybrid'] += 0.7
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores['hybrid'] = 1.0  # Default to hybrid if no clear signals
        
        # Choose best retriever
        best_retriever = max(scores, key=scores.get)
        
        return best_retriever, scores
    
    def explain_choice(self, query: str, retriever: str, scores: Dict[str, float]) -> str:
        """Explain why a particular retriever was chosen"""
        explanations = {
            'bm25': [
                "Using keyword search because:",
                "- Found exact claim ID pattern" if re.search(r'CLM[-\s]?\d+', query, re.I) else None,
                "- Contains specific amount" if re.search(r'\$[\d,]+', query) else None,
                "- Has exact status keyword" if any(s in query.lower() for s in ['pending', 'approved', 'denied']) else None,
                "- Short, specific query" if len(query.split()) <= 2 else None
            ],
            'vector': [
                "Using semantic search because:",
                "- Question requiring understanding" if query.lower().startswith(('what', 'why', 'how')) else None,
                "- Conceptual/analytical query" if any(c in query.lower() for c in ['analyze', 'summarize', 'pattern']) else None,
                "- Comparative language" if any(c in query.lower() for c in ['expensive', 'high', 'risk']) else None,
                "- Complex natural language query" if len(query.split()) > 6 else None
            ],
            'hybrid': [
                "Using hybrid search because:",
                "- Query has both keywords and concepts",
                "- Medium complexity query",
                "- Best of both approaches needed"
            ]
        }
        
        # Filter out None explanations
        reasons = [r for r in explanations[retriever] if r is not None]
        
        # Add confidence
        confidence = scores[retriever]
        if confidence > 0.7:
            conf_text = "High confidence"
        elif confidence > 0.4:
            conf_text = "Medium confidence"
        else:
            conf_text = "Low confidence"
            
        reasons.append(f"- {conf_text} ({confidence:.0%})")
        
        return "\n".join(reasons)
    
    def smart_query(self, query: str, explain: bool = True) -> Dict:
        """Automatically route query to best retriever"""
        # Analyze query
        best_retriever, scores = self.analyze_query(query)
        
        if explain:
            print(f"\n🤖 AI Decision: Using {best_retriever.upper()} search")
            explanation = self.explain_choice(query, best_retriever, scores)
            print(explanation)
            print(f"\nConfidence scores: BM25: {scores['bm25']:.0%}, "
                  f"Vector: {scores['vector']:.0%}, Hybrid: {scores['hybrid']:.0%}")
        
        # Execute query with chosen retriever
        result = self.hybrid_search.query_with_answer(query, search_type=best_retriever)
        result['chosen_retriever'] = best_retriever
        result['confidence_scores'] = scores
        
        return result

def demo_routing_decisions():
    """Show how the router makes decisions"""
    test_queries = [
        "CLM-12345",                              # Clear BM25
        "What are the riskiest claims?",          # Clear Vector
        "pending auto claims over $5000",         # Clear Hybrid
        "approved",                               # BM25
        "analyze claim patterns for fraud",       # Vector
        "CLM-001 CLM-002 CLM-003",               # BM25
        "which claims need immediate attention"   # Vector
    ]
    
    router = SmartRetrieverRouter()
    
    print("\n📊 Routing Decision Examples:")
    print("=" * 60)
    
    for query in test_queries:
        best, scores = router.analyze_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Decision: {best.upper()}")
        print(f"Scores: BM25={scores['bm25']:.0%}, Vector={scores['vector']:.0%}, Hybrid={scores['hybrid']:.0%}")

def main():
    """Interactive smart query system"""
    print("\n🤖 Smart Query System with Automatic Routing")
    print("=" * 60)
    print("I automatically choose the best search method for your query!")
    
    # Initialize
    router = SmartRetrieverRouter()
    
    # Load data
    print("\n📄 Loading insurance claims data...")
    router.hybrid_search.load_and_index_data("data/insurance_claims.csv")
    
    # Show demo?
    if input("\n📊 See routing examples? (y/n): ").lower() == 'y':
        demo_routing_decisions()
    
    # Interactive mode
    print("\n💬 Ask anything - I'll choose the best search method!")
    print("Commands: 'explain off' to hide explanations, 'explain on' to show")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    explain = True
    
    while True:
        query = input("\n❓ Your question: ")
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'explain off':
            explain = False
            print("Explanations disabled")
            continue
        elif query.lower() == 'explain on':
            explain = True
            print("Explanations enabled")
            continue
        
        try:
            result = router.smart_query(query, explain=explain)
            
            print(f"\n📝 Answer:")
            print(result['answer'])
            
            if not explain:
                print(f"\n(Used {result['chosen_retriever']} search)")
            
            # Show sources
            print(f"\n📚 Sources ({len(result['sources'])} found):")
            for i, doc in enumerate(result['sources'][:2], 1):  # Show top 2
                print(f"\n  Source {i}:")
                print(f"  {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()