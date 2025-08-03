# insurance_agent.py - Simple LangGraph Agent for Insurance Claims
# Combines local claims data with online legislation search

import os
from typing import TypedDict, Dict, List
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import your existing smart query system
from smart_query import SmartRetrieverRouter

load_dotenv()

# Define the state that flows through the graph
class InsuranceAgentState(TypedDict):
    query: str                    # User's question
    claims_data: str             # Results from Qdrant
    legislation_data: str        # Results from Tavily
    needs_legislation: bool      # Whether to search for laws
    final_answer: str           # Combined answer
    detected_state: str         # Which state's laws to search

class SimpleInsuranceAgent:
    """Simple agent that searches claims and relevant laws"""
    
    def __init__(self):
        # Initialize components
        self.router = SmartRetrieverRouter()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Initialize Tavily (will check for API key)
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if self.tavily_api_key:
            self.tavily = TavilySearchResults(max_results=3)
            print("✅ Tavily enabled for legislation search")
        else:
            self.tavily = None
            print("ℹ️  Tavily disabled (add TAVILY_API_KEY for legislation search)")
        
        print("✅ Insurance Agent initialized")
        
        # Load claims data
        print("📄 Loading claims data...")
        self.router.hybrid_search.load_and_index_data("data/insurance_claims.csv")
    
    def search_claims(self, state: InsuranceAgentState) -> InsuranceAgentState:
        """Step 1: Search Qdrant for relevant claims"""
        print("\n🔍 Step 1: Searching claims database...")
        
        # Use smart router to search claims
        result = self.router.smart_query(state["query"], explain=False)
        
        # Format claims data
        claims_text = f"Answer from claims database:\n{result['answer']}\n\n"
        
        if result['sources']:
            claims_text += "Relevant claims found:\n"
            for i, doc in enumerate(result['sources'][:3], 1):
                claims_text += f"\n{i}. {doc.page_content[:200]}...\n"
                if 'claim_id' in doc.metadata:
                    claims_text += f"   Claim ID: {doc.metadata['claim_id']}\n"
        
        state["claims_data"] = claims_text
        
        # Always search legislation if Tavily is available
        state["detected_state"] = self._detect_state(state["query"], claims_text)
        state["needs_legislation"] = bool(self.tavily)  # Always true if Tavily available
        
        print(f"   ✓ Will search legislation for: {state['detected_state']}")
        
        return state
    
    def search_legislation(self, state: InsuranceAgentState) -> InsuranceAgentState:
        """Step 2: Always search Tavily for relevant state legislation"""
        if not self.tavily:
            state["legislation_data"] = "\n(Legislation search unavailable - add TAVILY_API_KEY to enable)"
            return state
        
        print("\n📚 Step 2: Searching for relevant legislation...")
        
        # Extract key terms from query and claims context
        query_terms = self._extract_legal_terms(state["query"])
        
        # Search for adjuster-relevant laws and regulations
        search_query = f"{state['detected_state']} insurance {query_terms} claims adjuster requirements deadlines statute limitations liability 2024"
        print(f"   Searching: {search_query}")
        
        try:
            # Search for legislation
            results = self.tavily.run(search_query)
            
            # Format results for adjusters
            legislation_text = f"\n⚖️ {state['detected_state']} Insurance Law & Compliance Requirements:\n"
            for i, result in enumerate(results[:3], 1):
                legislation_text += f"\n{i}. REGULATORY REQUIREMENT:\n"
                legislation_text += f"   {result['content'][:400]}...\n"
                legislation_text += f"   📌 Reference: {result['url']}\n"
            
            state["legislation_data"] = legislation_text
            print("   ✓ Found relevant legislation")
            
        except Exception as e:
            print(f"   ⚠️  Legislation search failed: {str(e)}")
            state["legislation_data"] = "\nNo legislation data available."
        
        return state
    
    def combine_and_respond(self, state: InsuranceAgentState) -> InsuranceAgentState:
        """Step 3: Combine results and generate final answer"""
        print("\n🤖 Step 3: Generating adjuster guidance...")
        
        # Build prompt
        prompt = f"""You are a senior insurance claims expert providing guidance to a claims adjuster.
        Based on the claims data and relevant state legislation, provide professional guidance for claim adjudication.

        Adjuster's Query: {state['query']}

        Historical Claims Data:
        {state['claims_data']}

        Relevant State Legislation & Compliance Requirements:
        {state['legislation_data']}

        Provide professional guidance that includes:
        1. CLAIM ASSESSMENT: Direct answer to the adjuster's query with specific claim references
        2. LEGAL COMPLIANCE: Relevant statutory requirements and deadlines the adjuster must follow
        3. LIABILITY ANALYSIS: Key factors to consider based on similar claims and precedents
        4. RECOMMENDED ACTIONS: Specific next steps for proper claim adjudication
        5. RISK FACTORS: Any red flags or special considerations based on the legislation

        Format your response as a professional memo to the claims adjuster with clear sections.
        """
        
        # Generate answer
        response = self.llm.invoke(prompt)
        state["final_answer"] = response.content
        
        print("   ✓ Answer generated")
        return state
    
    def _detect_state(self, query: str, claims_text: str) -> str:
        """Detect which state's laws to search for"""
        # Common state abbreviations and names
        states = {
            'CA': 'California', 'TX': 'Texas', 'FL': 'Florida', 
            'NY': 'New York', 'PA': 'Pennsylvania', 'IL': 'Illinois',
            'OH': 'Ohio', 'GA': 'Georgia', 'NC': 'North Carolina',
            'MI': 'Michigan', 'NJ': 'New Jersey', 'VA': 'Virginia'
        }
        
        # Check query and claims for state mentions
        combined_text = (query + " " + claims_text).upper()
        
        for abbr, name in states.items():
            if abbr in combined_text or name.upper() in combined_text:
                return name
        
        # Default to a common state if none detected
        return "California"
    
    def _extract_legal_terms(self, query: str) -> str:
        """Extract key terms from query for legislation search"""
        # Start with the main topic from the query
        query_lower = query.lower()
        
        # Identify claim type
        if 'auto' in query_lower or 'car' in query_lower or 'vehicle' in query_lower:
            claim_type = 'auto automobile vehicle'
        elif 'property' in query_lower or 'home' in query_lower:
            claim_type = 'property homeowner'
        elif 'medical' in query_lower or 'health' in query_lower:
            claim_type = 'medical health'
        else:
            claim_type = 'general'
        
        # Always search for common insurance topics
        base_terms = f"{claim_type} claim filing deadlines coverage requirements"
        
        return base_terms
    
    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        # Initialize workflow
        workflow = StateGraph(InsuranceAgentState)
        
        # Add nodes
        workflow.add_node("search_claims", self.search_claims)
        workflow.add_node("search_legislation", self.search_legislation)
        workflow.add_node("combine_respond", self.combine_and_respond)
        
        # Define flow - always go through all steps
        workflow.set_entry_point("search_claims")
        workflow.add_edge("search_claims", "search_legislation")
        workflow.add_edge("search_legislation", "combine_respond")
        workflow.add_edge("combine_respond", END)
        
        return workflow.compile()
    
    def run(self, query: str) -> str:
        """Run the agent with a query"""
        # Create workflow
        app = self.create_workflow()
        
        # Initial state
        initial_state = {
            "query": query,
            "claims_data": "",
            "legislation_data": "",
            "needs_legislation": False,
            "final_answer": "",
            "detected_state": ""
        }
        
        # Run workflow
        result = app.invoke(initial_state)
        
        return result["final_answer"]

def visualize_workflow():
    """Show how the workflow works"""
    print("""
    🔄 Insurance Agent Workflow (Always 3 Steps):
    
    User Query
        ↓
    [Step 1: Search Claims]
        ↓
    [Step 2: Search Legislation]
        ↓
    [Step 3: Combine & Answer]
        ↓
    Final Answer
    """)

def main():
    """Interactive agent for claims adjusters"""
    print("\n⚖️ Claims Adjuster Intelligence System")
    print("=" * 60)
    print("Professional guidance combining historical claims data with current state regulations")
    
    # Show workflow
    print("""
    📋 Adjudication Workflow:
    
    Your Query
        ↓
    [1. Search Historical Claims Database]
        ↓
    [2. Research Applicable State Laws]
        ↓
    [3. Generate Adjudication Guidance]
        ↓
    Professional Recommendation
    """)
    
    # Initialize agent
    agent = SimpleInsuranceAgent()
    
    # Example queries
    print("\n📝 Example adjuster queries:")
    print("1. 'What is the typical settlement range for slip and fall claims in retail stores?'")
    print("2. 'Are there specific deadlines for auto accident claim notifications in California?'")
    print("3. 'What factors should I consider for a premises liability claim with inadequate lighting?'")
    print("4. 'Show me similar food poisoning claims and applicable health code violations'")
    
    try:
        # Demo mode
        if input("\n📊 Run adjuster demo? (y/n): ").lower() == 'y':
            demo_query = "What factors should I consider when adjudicating a slip and fall claim in a California retail store?"
            print(f"\n📋 Processing query: '{demo_query}'")
            answer = agent.run(demo_query)
            print(f"\n📄 ADJUDICATION GUIDANCE:\n{answer}")
        
        # Interactive mode
        print("\n⚖️ Adjuster Query Mode (type 'quit' to exit)")
        print("=" * 60)
        print("Enter your claim-related questions for professional guidance")
        
        while True:
            query = input("\n🔍 Adjuster query: ")
            if query.lower() == 'quit':
                break
            
            try:
                print("\n⏳ Analyzing claims database and regulations...")
                answer = agent.run(query)
                print(f"\n📄 ADJUDICATION GUIDANCE:\n{answer}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    except (EOFError, KeyboardInterrupt):
        print("\n\nGoodbye! 👋")

if __name__ == "__main__":
    main()