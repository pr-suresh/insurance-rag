# enhanced_adjuster_agent.py - Enhanced Claims Adjuster Agent with Pattern & Financial Analysis
# Integrates the new pattern analysis and financial analysis capabilities

import os
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# Import existing components
from enhanced_rag import EnhancedInsuranceRAG
from pattern_analysis_agent import PatternAnalysisAgent
from financial_agent import FinancialAnalysisAgent

# Import Tavily for legal research
try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

load_dotenv()

# Define the state
class EnhancedAdjusterState(TypedDict):
    messages: Annotated[List, add_messages]
    claim_context: str
    detected_state: str
    analysis_mode: str  # 'standard', 'pattern', 'financial'

class EnhancedClaimsAdjusterAgent:
    """Enhanced agent with pattern analysis and financial analytics capabilities"""
    
    def __init__(self):
        # Use GPT-3.5 for cost efficiency
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            streaming=True
        )
        
        # Initialize all components
        print("📄 Loading claims database...")
        self.rag = EnhancedInsuranceRAG(use_reranking=False)
        self.rag.ingest_data("data/insurance_claims.csv")
        
        print("📊 Initializing pattern analysis agent...")
        self.pattern_agent = PatternAnalysisAgent()
        
        print("💰 Initializing financial analysis agent...")
        self.financial_agent = FinancialAnalysisAgent()
        
        # Check for Tavily
        if not os.getenv("TAVILY_API_KEY") or not TAVILY_AVAILABLE:
            print("⚠️  Warning: Tavily legal search unavailable")
            self.tavily_available = False
        else:
            print("✅ Tavily legal search enabled")
            self.tavily_available = True
        
        # Store for state detection
        self.last_searched_states = []
        
        # Setup all tools
        self.setup_tools()
        
    def setup_tools(self):
        """Setup all available tools including new pattern and financial tools"""
        
        # Original search tools
        @tool
        def search_claims_database(query: str) -> str:
            """Search the internal claims database for similar cases and precedents."""
            try:
                results = self.rag.hybrid_search(query, k=5)
                
                if not results:
                    return "No relevant claims found in the database."
                
                # Extract states from results
                states = []
                for result in results:
                    metadata = result.get('metadata', {})
                    if 'Loss state' in metadata:
                        states.append(metadata['Loss state'])
                
                if states:
                    self.last_searched_states = list(set(states))
                
                # Format output
                output = "Found relevant claims:\n\n"
                for i, result in enumerate(results[:5], 1):
                    content = result.get('content', '')
                    metadata = result.get('metadata', {})
                    
                    output += f"**Claim {i}**: {metadata.get('Claim Number', 'Unknown')}\n"
                    output += f"- Type: {metadata.get('Claim Feature', 'N/A')}\n"
                    output += f"- State: {metadata.get('Loss state', 'N/A')}\n"
                    output += f"- Loss: ${metadata.get('Paid Indemnity', 0):,.0f} paid, ${metadata.get('Outstanding Indemnity', 0):,.0f} outstanding\n"
                    output += f"- Context: {content[:200]}...\n\n"
                
                return output
            except Exception as e:
                return f"Error searching claims database: {str(e)}"
        
        @tool
        def search_state_insurance_laws(query: str, state: str = None) -> str:
            """Search for specific state insurance laws and regulations."""
            if not self.tavily_available:
                return "Legal search unavailable. Please add TAVILY_API_KEY to .env file."
            
            # Auto-detect state if not provided
            if not state and self.last_searched_states:
                state = self.last_searched_states[0]
                print(f"Auto-detected state: {state}")
            
            if not state:
                return "Please specify a state or search claims first to auto-detect the state."
            
            try:
                tavily = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))
                
                # Enhanced query with state
                enhanced_query = f"{state} state insurance law regulation {query}"
                
                results = tavily.invoke(enhanced_query)
                
                if not results:
                    return f"No legal information found for {state} regarding: {query}"
                
                # Format results
                output = f"## {state} Insurance Law Research\n\n"
                for i, result in enumerate(results[:3], 1):
                    output += f"**Source {i}**: {result.get('title', 'N/A')}\n"
                    output += f"URL: {result.get('url', 'N/A')}\n"
                    output += f"{result.get('content', 'No content available')[:500]}...\n\n"
                
                return output
            except Exception as e:
                return f"Error searching legal information: {str(e)}"
        
        # Get pattern analysis tools
        pattern_tools = self.pattern_agent.get_tools()
        
        # Get financial analysis tools
        financial_tools = self.financial_agent.get_tools()
        
        # Combine all tools
        all_tools = [
            search_claims_database,
            search_state_insurance_laws
        ] + pattern_tools + financial_tools
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(all_tools)
        self.tool_node = ToolNode(all_tools)
        self.tools = all_tools
        
    def create_graph(self):
        """Create the enhanced LangGraph workflow"""
        workflow = StateGraph(EnhancedAdjusterState)
        
        # Define the adjuster node
        def adjuster_node(state: EnhancedAdjusterState):
            """Main adjuster logic with mode awareness"""
            messages = state.get("messages", [])
            
            # Determine analysis mode from user query
            last_message = messages[-1].content if messages else ""
            
            # Add mode-specific prompting
            system_prompt = """You are an enhanced insurance claims adjuster assistant with advanced capabilities:

1. **Claims Search**: Search historical claims for precedents and similar cases
2. **Legal Research**: Find state-specific insurance laws and regulations
3. **Pattern Analysis**: Identify claim patterns, risk indicators, and predict outcomes
4. **Financial Analysis**: Calculate exposures, analyze reserves, and generate cost reports

Available tools:
- search_claims_database: Find similar historical claims
- search_state_insurance_laws: Research state-specific regulations
- analyze_claim_patterns_by_type: Analyze patterns by claim type
- identify_risk_indicators: Identify risk factors from historical data
- predict_claim_outcome: Predict likely outcomes based on similar cases
- analyze_settlement_trends: Analyze settlement trends over time
- calculate_total_exposure: Calculate financial exposure with filters
- analyze_reserve_adequacy: Analyze if reserves are adequate
- generate_cost_driver_report: Identify primary cost drivers
- calculate_settlement_efficiency: Analyze settlement efficiency metrics
- generate_financial_summary: Generate comprehensive financial summary

Use multiple tools when appropriate to provide comprehensive analysis."""
            
            # Prepend system message
            full_messages = [AIMessage(content=system_prompt)] + messages
            
            # Get response from LLM
            response = self.llm_with_tools.invoke(full_messages)
            
            return {"messages": [response]}
        
        # Add nodes
        workflow.add_node("adjuster", adjuster_node)
        workflow.add_node("tools", self.tool_node)
        
        # Add edges
        workflow.set_entry_point("adjuster")
        workflow.add_conditional_edges(
            "adjuster",
            tools_condition,  # Built-in condition that checks for tool calls
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_edge("tools", "adjuster")
        
        # Compile the graph
        self.app = workflow.compile()
        
    def process_query(self, query: str) -> str:
        """Process a user query through the enhanced agent"""
        if not hasattr(self, 'app'):
            self.create_graph()
        
        # Determine if this is a specialized query
        query_lower = query.lower()
        
        # Add hints for better tool selection
        enhanced_query = query
        if 'pattern' in query_lower or 'trend' in query_lower:
            enhanced_query = f"{query} (Consider using pattern analysis tools)"
        elif 'financial' in query_lower or 'cost' in query_lower or 'reserve' in query_lower:
            enhanced_query = f"{query} (Consider using financial analysis tools)"
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=enhanced_query)],
            "claim_context": "",
            "detected_state": "",
            "analysis_mode": "enhanced"
        }
        
        # Run the graph
        try:
            result = self.app.invoke(initial_state)
            
            # Extract the final response
            final_message = result["messages"][-1]
            
            if hasattr(final_message, 'content'):
                return final_message.content
            else:
                return str(final_message)
                
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def get_available_capabilities(self) -> str:
        """Return a formatted list of available capabilities"""
        capabilities = """
## Enhanced Claims Adjuster Capabilities

### 🔍 Core Search
- **Claims Database Search**: Find similar historical claims and precedents
- **State Legal Research**: Research state-specific insurance laws and regulations

### 📊 Pattern Analysis (NEW)
- **Claim Pattern Analysis**: Analyze patterns by claim type, frequency, and costs
- **Risk Indicator Identification**: Identify risk factors from historical data
- **Outcome Prediction**: Predict likely claim outcomes based on similar cases
- **Settlement Trend Analysis**: Analyze how settlements change over time

### 💰 Financial Analytics (NEW)
- **Exposure Calculation**: Calculate total financial exposure with filters
- **Reserve Adequacy Analysis**: Determine if reserves are adequate
- **Cost Driver Reports**: Identify what drives claim costs
- **Settlement Efficiency**: Analyze efficiency of claim settlements
- **Financial Summaries**: Generate executive-level financial reports

### Example Queries:
1. "What patterns do you see in slip and fall claims?"
2. "Calculate our total exposure in California"
3. "Predict the outcome for a product liability claim in Texas"
4. "Generate a financial summary for 2023"
5. "Identify risk indicators for our retail clients"
6. "Analyze reserve adequacy for construction defect claims"
"""
        return capabilities

# Convenience function for testing
def test_enhanced_agent():
    """Test the enhanced agent with sample queries"""
    agent = EnhancedClaimsAdjusterAgent()
    
    test_queries = [
        "What patterns do you see in slip and fall claims?",
        "Calculate our total financial exposure in California",
        "Predict the outcome for a products liability bodily injury claim in Texas",
        "Generate a cost driver report for the top 5 cost drivers",
        "Analyze settlement efficiency for closed claims"
    ]
    
    print("Testing Enhanced Claims Adjuster Agent\n" + "="*50)
    print(agent.get_available_capabilities())
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        response = agent.process_query(query)
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    test_enhanced_agent()