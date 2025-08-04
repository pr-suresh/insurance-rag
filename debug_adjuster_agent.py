#!/usr/bin/env python3
"""
Debug-Enhanced Adjuster Agent with Comprehensive Logging
Includes step-by-step LangGraph logging and performance monitoring
"""

import os
import logging
import time
from typing import TypedDict, List, Annotated, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.callbacks import BaseCallbackHandler

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# Import your existing query system
from enhanced_rag import EnhancedInsuranceRAG

load_dotenv()

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_adjuster_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DebugCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for detailed LLM call logging."""
    
    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0
        self.start_time = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        self.call_count += 1
        self.start_time = time.time()
        logger.info(f"🤖 LLM Call #{self.call_count} Started")
        logger.debug(f"Prompt: {prompts[0][:200]}..." if prompts else "No prompt")
    
    def on_llm_end(self, response, **kwargs) -> None:
        duration = time.time() - self.start_time if self.start_time else 0
        token_usage = getattr(response, 'llm_output', {}).get('token_usage', {})
        
        if token_usage:
            total = token_usage.get('total_tokens', 0)
            self.total_tokens += total
            logger.info(f"✅ LLM Call completed in {duration:.2f}s | Tokens: {total} | Total: {self.total_tokens}")
        else:
            logger.info(f"✅ LLM Call completed in {duration:.2f}s")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        tool_name = serialized.get('name', 'Unknown Tool')
        logger.info(f"🔧 Tool '{tool_name}' started with input: {input_str[:100]}...")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        logger.info(f"🔧 Tool completed. Output length: {len(output)} chars")

# Define the state with debugging info
class DebugAdjusterState(TypedDict):
    messages: Annotated[List, add_messages]
    claim_context: str
    detected_state: str
    step_logs: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]

class DebugClaimsAdjusterAgent:
    """Enhanced Claims Adjuster Agent with comprehensive debugging."""
    
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"🚀 Initializing Debug Adjuster Agent - Session: {self.session_id}")
        
        # Initialize callback handler
        self.callback_handler = DebugCallbackHandler()
        
        # Use cheaper model for cost savings with debugging
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            streaming=True,
            callbacks=[self.callback_handler] if debug_mode else []
        )
        
        # Initialize query router for claims search
        logger.info("📄 Initializing Enhanced RAG system...")
        self.rag = EnhancedInsuranceRAG(use_reranking=False)
        self.rag.ingest_data("data/insurance_claims.csv")
        logger.info(f"✅ RAG system loaded with {len(self.rag.documents)} documents")
        
        # Check for Tavily API key
        if not os.getenv("TAVILY_API_KEY"):
            logger.warning("⚠️ No TAVILY_API_KEY found. Legal search will be limited.")
        else:
            logger.info("✅ Tavily legislation search enabled")
        
        # Create tools
        self.tools = self._create_tools()
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        logger.info("✅ Debug Adjuster Agent initialized")
    
    def _log_step(self, step_name: str, data: Dict[str, Any], state: DebugAdjusterState):
        """Log a step in the LangGraph execution."""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'data': data,
            'message_count': len(state.get('messages', [])),
            'session_id': self.session_id
        }
        
        if 'step_logs' not in state:
            state['step_logs'] = []
        
        state['step_logs'].append(log_entry)
        
        if self.debug_mode:
            logger.info(f"📝 Step: {step_name} | Messages: {len(state.get('messages', []))}")
            logger.debug(f"Step data: {data}")
    
    def _create_tools(self):
        """Create debugging-enhanced tools."""
        
        @tool
        def search_claims_database(query: str) -> str:
            """Search the internal claims database for similar cases and precedents."""
            
            start_time = time.time()
            logger.info(f"🔍 Claims database search: '{query}'")
            
            try:
                # Use enhanced RAG for better claim search
                results = self.rag.hybrid_search(query, k=5)
                duration = time.time() - start_time
                
                logger.info(f"📊 Found {len(results)} results in {duration:.2f}s")
                
                if not results:
                    return f"CLAIMS DATABASE RESULTS:\nNo results found for query: {query}"
                
                # Format for adjuster with debugging info
                output = f"CLAIMS DATABASE RESULTS:\nFound {len(results)} relevant claims (search time: {duration:.2f}s):\n\n"
                
                for i, result in enumerate(results[:3], 1):
                    metadata = result.get('metadata', {})
                    claim_id = metadata.get('claim_id', 'Unknown')
                    claim_type = metadata.get('claim_type', 'Unknown')
                    total_exposure = metadata.get('total_exposure', 0)
                    state = metadata.get('loss_state', 'Unknown')
                    score = result.get('score', 0)
                    search_type = result.get('type', 'unknown')
                    
                    output += f"{i}. CLAIM {claim_id} (Score: {score:.3f}, Type: {search_type}):\n"
                    output += f"   Type: {claim_type}\n"
                    output += f"   Total Exposure: ${total_exposure:,.2f}\n"
                    output += f"   State: {state}\n"
                    output += f"   Details: {result['content'][:300]}...\n\n"
                
                logger.debug(f"Claims search output length: {len(output)} characters")
                return output
                
            except Exception as e:
                logger.error(f"❌ Claims database search failed: {e}")
                return f"Error searching claims database: {str(e)}"
        
        @tool
        def search_state_insurance_laws(query: str, state: str = "California") -> str:
            """Search for specific state insurance laws and regulations."""
            
            start_time = time.time()
            logger.info(f"⚖️ Legal search: '{query}' in {state}")
            
            if not os.getenv("TAVILY_API_KEY"):
                logger.warning("⚠️ Tavily API key not available")
                return f"Legal search unavailable - add TAVILY_API_KEY to .env file"
            
            try:
                # Initialize Tavily Search
                tavily = TavilySearch(max_results=3)
                
                # Build targeted search query
                search_query = f"{state} insurance law regulation {query} statute requirement 2024 liability claims adjuster"
                
                # Search for legislation
                results = tavily.invoke({"query": search_query})
                duration = time.time() - start_time
                
                logger.info(f"📚 Legal search completed in {duration:.2f}s, found {len(results)} results")
                
                # Format for adjusters
                output = f"\n{state.upper()} INSURANCE LAW RESEARCH (search time: {duration:.2f}s):\n"
                for i, result in enumerate(results, 1):
                    output += f"\n{i}. LEGAL FINDING:\n"
                    output += f"   {result.get('content', '')[:400]}...\n"
                    output += f"   Source: {result.get('url', 'N/A')}\n"
                
                logger.debug(f"Legal search output length: {len(output)} characters")
                return output
                
            except Exception as e:
                logger.error(f"❌ Legal search failed: {e}")
                return f"Error searching legislation: {str(e)}"
        
        return [search_claims_database, search_state_insurance_laws]
    
    def analyze_claim(self, state: DebugAdjusterState) -> DebugAdjusterState:
        """Initial analysis node with detailed logging."""
        
        step_start = time.time()
        logger.info("🤖 Starting claim analysis...")
        
        # Get the last user message
        user_query = state["messages"][-1].content
        
        # Detect state from query
        detected_state = self._detect_state(user_query)
        state["detected_state"] = detected_state
        
        # Log step details
        step_data = {
            'user_query': user_query,
            'detected_state': detected_state,
            'query_length': len(user_query)
        }
        
        self._log_step('analyze_claim_start', step_data, state)
        
        # Create system prompt for adjuster agent
        system_prompt = f"""You are an experienced insurance claims adjuster assistant with debugging enabled.
        You help adjusters make informed decisions by researching both historical claims and current state laws.
        
        Session ID: {self.session_id}
        Current query context:
        - Detected state: {detected_state}
        - Query: {user_query}
        - Query type: {'Specific claim' if any(char in user_query for char in ['-', 'GL', 'CLM']) else 'General inquiry'}
        
        Available tools:
        1. search_claims_database - Find similar claims and precedents
        2. search_state_insurance_laws - Check state-specific regulations
        
        For queries about liability, compliance, or legal requirements, ALWAYS use both tools.
        First search claims for precedents, then verify against state laws.
        Be thorough and provide specific details including claim numbers, amounts, and legal citations.
        """
        
        # Get agent's response with tool calls
        messages = [
            {"role": "system", "content": system_prompt},
            state["messages"][-1]
        ]
        
        try:
            response = self.llm_with_tools.invoke(messages)
            state["messages"].append(response)
            
            step_duration = time.time() - step_start
            
            # Log completion
            completion_data = {
                'step_duration': step_duration,
                'response_type': type(response).__name__,
                'tool_calls': len(getattr(response, 'tool_calls', [])),
                'has_tool_calls': bool(getattr(response, 'tool_calls', []))
            }
            
            self._log_step('analyze_claim_complete', completion_data, state)
            logger.info(f"✅ Claim analysis completed in {step_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Claim analysis failed: {e}")
            # Add error message to state
            error_msg = AIMessage(content=f"Analysis failed: {str(e)}")
            state["messages"].append(error_msg)
        
        return state
    
    def generate_guidance(self, state: DebugAdjusterState) -> DebugAdjusterState:
        """Generate final adjudication guidance with logging."""
        
        step_start = time.time()
        logger.info("📋 Generating adjudication guidance...")
        
        # Build comprehensive prompt
        guidance_prompt = f"""Based on the research conducted in session {self.session_id}, provide professional adjudication guidance.
        
        Format your response as:
        
        ADJUDICATION ANALYSIS:
        1. CLAIM ASSESSMENT: Key findings from similar claims
        2. LEGAL COMPLIANCE: Relevant state laws and requirements  
        3. LIABILITY FACTORS: Important considerations
        4. RECOMMENDED ACTION: Specific next steps
        5. RISK ALERTS: Any red flags or special considerations
        6. DEBUG INFO: Session {self.session_id} - Analysis based on {len(state.get('step_logs', []))} logged steps
        
        Be specific and reference the actual findings from your research.
        Include claim numbers, amounts, and legal citations where available.
        """
        
        state["messages"].append(HumanMessage(content=guidance_prompt))
        
        try:
            # Generate final guidance
            final_response = self.llm.invoke(state["messages"])
            state["messages"].append(final_response)
            
            step_duration = time.time() - step_start
            
            # Log completion with performance metrics
            state["performance_metrics"] = {
                'total_duration': sum(log.get('data', {}).get('step_duration', 0) for log in state.get('step_logs', [])),
                'total_steps': len(state.get('step_logs', [])),
                'total_messages': len(state.get('messages', [])),
                'session_id': self.session_id,
                'guidance_generation_time': step_duration,
                'total_llm_calls': self.callback_handler.call_count,
                'total_tokens': self.callback_handler.total_tokens
            }
            
            completion_data = {
                'step_duration': step_duration,
                'final_response_length': len(final_response.content),
                'total_session_time': state["performance_metrics"]['total_duration']
            }
            
            self._log_step('generate_guidance_complete', completion_data, state)
            logger.info(f"✅ Guidance generated in {step_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Guidance generation failed: {e}")
            error_msg = AIMessage(content=f"Guidance generation failed: {str(e)}")
            state["messages"].append(error_msg)
        
        return state
    
    def _detect_state(self, query: str) -> str:
        """Enhanced state detection with logging."""
        
        states = {
            'texas': 'Texas', 'tx': 'Texas',
            'california': 'California', 'ca': 'California', 
            'florida': 'Florida', 'fl': 'Florida',
            'new york': 'New York', 'ny': 'New York',
            'illinois': 'Illinois', 'il': 'Illinois',
            'virginia': 'Virginia', 'va': 'Virginia',
            'minnesota': 'Minnesota', 'mn': 'Minnesota'
        }
        
        query_lower = query.lower()
        for key, state in states.items():
            if key in query_lower:
                logger.debug(f"🗺️ Detected state: {state} (keyword: {key})")
                return state
        
        logger.debug("🗺️ No state detected, defaulting to California")
        return "California"  # Default
    
    def create_workflow(self):
        """Create the LangGraph workflow with enhanced debugging."""
        
        logger.info("🔧 Creating LangGraph workflow...")
        
        workflow = StateGraph(DebugAdjusterState)
        
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
        
        logger.info("✅ LangGraph workflow created")
        
        return workflow.compile()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process an adjuster query with comprehensive debugging."""
        
        session_start = time.time()
        logger.info(f"🎯 Processing query in session {self.session_id}: '{query[:100]}...'")
        
        # Create workflow
        app = self.create_workflow()
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "claim_context": "",
            "detected_state": "",
            "step_logs": [],
            "performance_metrics": {}
        }
        
        try:
            # Run workflow
            result = app.invoke(initial_state)
            
            total_duration = time.time() - session_start
            
            # Enhance performance metrics
            result["performance_metrics"]["session_total_time"] = total_duration
            result["performance_metrics"]["query"] = query
            result["performance_metrics"]["timestamp"] = datetime.now().isoformat()
            
            logger.info(f"🎉 Query processed successfully in {total_duration:.2f}s")
            logger.info(f"📊 Performance: {result['performance_metrics']['total_llm_calls']} LLM calls, {result['performance_metrics']['total_tokens']} tokens")
            
            # Save debug log
            self._save_debug_log(result)
            
            return {
                "answer": result["messages"][-1].content,
                "performance_metrics": result["performance_metrics"],
                "step_logs": result["step_logs"],
                "session_id": self.session_id
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "performance_metrics": {"error": str(e)},
                "step_logs": [],
                "session_id": self.session_id
            }
    
    def _save_debug_log(self, result: Dict[str, Any]):
        """Save detailed debug log to file."""
        
        log_filename = f"debug_session_{self.session_id}.json"
        
        try:
            import json
            
            debug_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": result.get("performance_metrics", {}),
                "step_logs": result.get("step_logs", []),
                "message_count": len(result.get("messages", [])),
                "final_answer_length": len(result["messages"][-1].content) if result.get("messages") else 0
            }
            
            with open(log_filename, 'w') as f:
                json.dump(debug_data, f, indent=2)
            
            logger.info(f"💾 Debug log saved to {log_filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save debug log: {e}")

def main():
    """Demo the debug adjuster agent."""
    
    print("🚀 DEBUG ADJUSTER AGENT DEMO")
    print("="*60)
    
    # Initialize debug agent
    agent = DebugClaimsAdjusterAgent(debug_mode=True)
    
    # Test queries
    test_queries = [
        "What is the total exposure for claim GL-2024-0025?",
        "Does the slip and fall claim violate Virginia liability rules?",
        "How should I handle the carbon monoxide case from a compliance perspective?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST QUERY {i}: {query}")
        print("="*60)
        
        result = agent.process_query(query)
        
        print(f"\n📋 RESULT:")
        print(result["answer"][:500] + "..." if len(result["answer"]) > 500 else result["answer"])
        
        print(f"\n📊 PERFORMANCE:")
        metrics = result["performance_metrics"]
        print(f"  Session Time: {metrics.get('session_total_time', 0):.2f}s")
        print(f"  LLM Calls: {metrics.get('total_llm_calls', 0)}")
        print(f"  Tokens: {metrics.get('total_tokens', 0)}")
        print(f"  Steps: {len(result['step_logs'])}")

if __name__ == "__main__":
    main()