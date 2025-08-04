# adjuster_agent.py - Simple Tavily Tool Integration for Claims Adjusters
# Uses tool nodes for cleaner integration and cheaper models

import os
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# Import your existing query system
from enhanced_rag import EnhancedInsuranceRAG

load_dotenv()

# Define the state
class AdjusterState(TypedDict):
    messages: Annotated[List, add_messages]
    claim_context: str
    detected_state: str

class ClaimsAdjusterAgent:
    """Simple agent with Tavily tool for insurance legislation searches"""
    
    def __init__(self):
        # Use cheaper model for cost savings
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Cheaper than gpt-4
            temperature=0,
            streaming=True
        )
        
        # Initialize Enhanced RAG for claims search
        print("📄 Loading claims database...")
        self.rag = EnhancedInsuranceRAG(use_reranking=False)
        self.rag.ingest_data("data/insurance_claims.csv")
        
        # Check for Tavily API key
        if not os.getenv("TAVILY_API_KEY"):
            print("⚠️  Warning: No TAVILY_API_KEY found. Add to .env for legislation search.")
        else:
            print("✅ Tavily legislation search enabled")
        
        # Create tools
        self.tools = self._create_tools()
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        print("✅ Adjuster Agent initialized with tools")
    
    def _create_tools(self):
        """Create the tools available to the agent"""
        
        # Store rag in local variable for closure
        rag_system = self.rag
        
        # Store last searched states for the legal search tool
        self.last_searched_states = []
        
        @tool
        def search_claims_database(query: str) -> str:
            """Search the internal claims database for similar cases and precedents.
            Use this to find historical claim data, patterns, and adjudication outcomes.
            
            Args:
                query: What to search for in the claims database
            """
            print(f"🔍 Searching claims database for: {query}")
            
            # Use the pre-initialized RAG system
            results = rag_system.hybrid_search(query, k=5)
            
            if not results:
                return f"CLAIMS DATABASE RESULTS:\nNo results found for query: {query}"
            
            # Format for adjuster and collect states
            output = f"CLAIMS DATABASE RESULTS:\nFound {len(results)} relevant claims:\n\n"
            states_found = []
            
            for i, result in enumerate(results[:3], 1):
                metadata = result.get('metadata', {})
                claim_id = metadata.get('claim_id', 'Unknown')
                claim_type = metadata.get('claim_type', 'Unknown')
                total_exposure = metadata.get('total_exposure', 0)
                state = metadata.get('loss_state', 'Unknown')
                
                # Collect states for legal search
                if state != 'Unknown' and state not in states_found:
                    states_found.append(state)
                
                output += f"{i}. CLAIM {claim_id}:\n"
                output += f"   Type: {claim_type}\n"
                output += f"   Total Exposure: ${total_exposure:,.2f}\n"
                output += f"   State: {state}\n"
                output += f"   Details: {result['content'][:300]}...\n\n"
            
            # Store found states for legal search
            self.last_searched_states = states_found
            
            if states_found:
                output += f"\n📍 Claims found in: {', '.join(states_found)}\n"
                output += f"💡 Tip: Use search_state_insurance_laws to check {states_found[0]} regulations\n"
            
            return output
        
        @tool
        def search_state_insurance_laws(query: str, state: str = None) -> str:
            """Search for specific state insurance laws, regulations, and legal requirements.
            Automatically uses the Loss State from recently searched claims when available.
            
            Args:
                query: The legal question or regulation to search for
                state: Optional - The US state whose laws to search. If not provided, uses states from recent claim searches.
            """
            # Auto-detect state from recent claim searches if not provided
            if state is None:
                if self.last_searched_states:
                    state = self.last_searched_states[0]  # Use first state from recent searches
                    print(f"🗺️ Auto-detected state from claims: {state}")
                else:
                    state = "California"  # Default fallback
                    print(f"🗺️ No state detected, using default: {state}")
            
            print(f"⚖️  Searching {state} insurance laws for: {query}")
            
            # Initialize Tavily Search
            tavily = TavilySearch(max_results=3)
            
            # Build targeted search query based on the claim context
            search_query = f"{state} insurance law regulation {query} statute requirement 2024 liability claims adjuster"
            
            try:
                # Search for legislation
                results = tavily.invoke({"query": search_query})
                
                # Format for adjusters
                output = f"\n{state.upper()} INSURANCE LAW RESEARCH:\n"
                output += f"Query: {query}\n"
                output += f"State: {state} {'(auto-detected from claims)' if state in self.last_searched_states else ''}\n\n"
                
                for i, result in enumerate(results, 1):
                    output += f"{i}. LEGAL FINDING:\n"
                    output += f"   {result.get('content', '')[:400]}...\n"
                    output += f"   Source: {result.get('url', 'N/A')}\n\n"
                
                # If multiple states were found in claims, suggest checking them too
                if len(self.last_searched_states) > 1:
                    other_states = [s for s in self.last_searched_states if s != state]
                    output += f"\n💡 Also consider checking regulations for: {', '.join(other_states)}\n"
                
                return output
            except Exception as e:
                return f"Error searching legislation: {str(e)}"
        
        return [search_claims_database, search_state_insurance_laws]
    
    def analyze_claim(self, state: AdjusterState) -> AdjusterState:
        """Initial analysis node - agent decides what tools to use"""
        print("\n🤖 Analyzing adjuster query...")
        
        # Get the last user message
        user_query = state["messages"][-1].content
        
        # Detect state from query
        state["detected_state"] = self._detect_state(user_query)
        
        # Create system prompt for adjuster agent
        system_prompt = f"""You are an experienced insurance claims adjuster assistant.
        You help adjusters make informed decisions by researching both historical claims and current state laws.
        
        Current query context:
        - Detected state: {state['detected_state']}
        - Query: {user_query}
        
        Available tools:
        1. search_claims_database - Find similar claims and precedents
        2. search_state_insurance_laws - Check state-specific regulations
        
        For queries about liability, compliance, or legal requirements, ALWAYS use both tools.
        First search claims for precedents, then verify against state laws.
        """
        
        # Get agent's response with tool calls
        messages = [
            {"role": "system", "content": system_prompt},
            state["messages"][-1]
        ]
        
        response = self.llm_with_tools.invoke(messages)
        state["messages"].append(response)
        
        return state
    
    def generate_guidance(self, state: AdjusterState) -> AdjusterState:
        """Generate final adjudication guidance"""
        print("\n📋 Generating adjudication guidance...")
        
        # Build comprehensive prompt
        guidance_prompt = """Based on the research conducted, provide professional adjudication guidance.
        
        Format your response as:
        
        ADJUDICATION ANALYSIS:
        1. CLAIM ASSESSMENT: Key findings from similar claims
        2. LEGAL COMPLIANCE: Relevant state laws and requirements  
        3. LIABILITY FACTORS: Important considerations
        4. RECOMMENDED ACTION: Specific next steps
        5. RISK ALERTS: Any red flags or special considerations
        
        Be specific and reference the actual findings from your research."""
        
        state["messages"].append(HumanMessage(content=guidance_prompt))
        
        # Generate final guidance
        final_response = self.llm.invoke(state["messages"])
        state["messages"].append(final_response)
        
        return state
    
    def _detect_state(self, query: str) -> str:
        """Simple state detection from query"""
        states = {
            'texas': 'Texas', 'tx': 'Texas',
            'california': 'California', 'ca': 'California', 
            'florida': 'Florida', 'fl': 'Florida',
            'new york': 'New York', 'ny': 'New York',
            'illinois': 'Illinois', 'il': 'Illinois'
        }
        
        query_lower = query.lower()
        for key, state in states.items():
            if key in query_lower:
                return state
        
        return "California"  # Default
    
    def create_workflow(self):
        """Create the LangGraph workflow with tool nodes"""
        workflow = StateGraph(AdjusterState)
        
        # Create tool node
        tool_node = ToolNode(self.tools)
        
        # Add nodes
        workflow.add_node("analyze", self.analyze_claim)
        workflow.add_node("tools", tool_node)
        workflow.add_node("guidance", self.generate_guidance)
        
        # Set entry point
        workflow.set_entry_point("analyze")
        
        # Add conditional edge - if tools are called, use them
        workflow.add_conditional_edges(
            "analyze",
            tools_condition,  # Built-in condition that checks for tool calls
            {
                "tools": "tools",
                "end": "guidance"
            }
        )
        
        # After tools, go to guidance
        workflow.add_edge("tools", "guidance")
        workflow.add_edge("guidance", END)
        
        return workflow.compile()
    
    def process_query(self, query: str) -> str:
        """Process an adjuster query"""
        # Create workflow
        app = self.create_workflow()
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "claim_context": "",
            "detected_state": ""
        }
        
        # Run workflow
        result = app.invoke(initial_state)
        
        # Return the last message (the guidance)
        return result["messages"][-1].content

def demo_queries():
    """Show example queries and how the agent handles them"""
    agent = ClaimsAdjusterAgent()
    
    print("\n" + "="*60)
    print("🎯 DEMO: How the Agent Decides to Use Tools")
    print("="*60)
    
    demo_cases = [
        {
            "query": "Does this slip and fall claim violate state liability rules in Texas?",
            "explanation": "This triggers BOTH tools because it asks about liability rules"
        },
        {
            "query": "What's the typical settlement for auto accidents with soft tissue injuries?",
            "explanation": "This mainly uses claims database for historical data"
        },
        {
            "query": "Is there a statute of limitations for property damage claims in Florida?",
            "explanation": "This triggers state law search for specific legal requirements"
        }
    ]
    
    for i, case in enumerate(demo_cases, 1):
        print(f"\n{'='*60}")
        print(f"Demo {i}: {case['query']}")
        print(f"Why: {case['explanation']}")
        print("="*60)
        
        if input("\nRun this demo? (y/n): ").lower() == 'y':
            print("\n⏳ Processing...")
            result = agent.process_query(case['query'])
            print(f"\n📋 ADJUDICATION GUIDANCE:\n{result}")

def main():
    """Interactive interface for claims adjusters"""
    print("\n⚖️  CLAIMS ADJUSTER INTELLIGENCE SYSTEM")
    print("="*60)
    print("AI-Powered Tool for Professional Claim Adjudication")
    print("\n💡 This system uses:")
    print("   • GPT-3.5-turbo (cost-effective)")
    print("   • Smart tool selection")
    print("   • State-specific law search")
    
    agent = ClaimsAdjusterAgent()
    
    # Show demo?
    if input("\n📊 See demo queries first? (y/n): ").lower() == 'y':
        demo_queries()
    
    # Interactive mode
    print("\n⚖️  ADJUSTER QUERY MODE")
    print("="*60)
    print("Ask questions about claims, liability, regulations, etc.")
    print("Examples:")
    print("• Does this claim violate state liability rules in Texas?")
    print("• What are the requirements for UM/UIM claims in California?")
    print("• Show similar premises liability settlements")
    print("\nType 'quit' to exit")
    print("="*60)
    
    while True:
        query = input("\n🔍 Adjuster query: ")
        if query.lower() == 'quit':
            break
        
        try:
            print("\n⏳ Researching claims database and state laws...")
            result = agent.process_query(query)
            print(f"\n📋 ADJUDICATION GUIDANCE:\n{result}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if "TAVILY_API_KEY" in str(e):
                print("💡 Tip: Add TAVILY_API_KEY to your .env file")

if __name__ == "__main__":
    main()