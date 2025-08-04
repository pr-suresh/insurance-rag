#!/usr/bin/env python3
"""
Test LangSmith Integration
Verifies that LangSmith is properly configured and working
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
import time

load_dotenv()

def check_langsmith_config():
    """Check if LangSmith is properly configured."""
    
    print("🔍 Checking LangSmith Configuration...")
    print("-" * 50)
    
    # Check API key
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if api_key:
        print("✅ LangSmith API key found")
        print(f"   Key prefix: {api_key[:10]}...")
    else:
        print("❌ No LangSmith API key found")
        print("   Please set LANGCHAIN_API_KEY in your .env file")
        return False
    
    # Check tracing settings
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    
    print(f"📊 Tracing enabled: {tracing_enabled}")
    print(f"📁 Project name: {project_name}")
    
    # Try to connect to LangSmith
    try:
        client = Client()
        print("✅ Successfully connected to LangSmith")
        
        # Try to create/access project
        try:
            client.create_project(project_name=project_name, if_exists="ignore")
            print(f"✅ Project '{project_name}' is accessible")
        except Exception as e:
            print(f"⚠️ Could not create/access project: {e}")
            
    except Exception as e:
        print(f"❌ Failed to connect to LangSmith: {e}")
        return False
    
    print("-" * 50)
    return True

def test_simple_chain():
    """Test a simple LangChain chain with LangSmith tracing."""
    
    print("\n🧪 Testing Simple Chain with LangSmith...")
    print("-" * 50)
    
    # Enable tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "insurance-rag-test"
    
    try:
        # Create a simple chain
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful insurance claims assistant."),
            ("user", "{input}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        # Test query
        test_query = "What are the key factors in evaluating a liability claim?"
        
        print(f"📝 Running test query: '{test_query}'")
        print("⏳ Executing chain (check LangSmith for trace)...")
        
        start_time = time.time()
        result = chain.invoke({"input": test_query})
        duration = time.time() - start_time
        
        print(f"✅ Chain executed successfully in {duration:.2f}s")
        print(f"📊 Response length: {len(result)} characters")
        print(f"💬 Response preview: {result[:200]}...")
        
        # Get trace URL
        project_name = os.getenv("LANGCHAIN_PROJECT", "insurance-rag-test")
        print(f"\n🌐 View trace at: https://smith.langchain.com/o/default/{project_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chain execution failed: {e}")
        return False

def test_complex_workflow():
    """Test a more complex workflow with multiple steps."""
    
    print("\n🧪 Testing Complex Workflow with LangSmith...")
    print("-" * 50)
    
    try:
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        
        # Create tools
        @tool
        def search_claims(query: str) -> str:
            """Search the claims database."""
            return f"Found 5 claims related to: {query}"
        
        @tool
        def check_regulations(state: str, claim_type: str) -> str:
            """Check state regulations."""
            return f"{state} regulations for {claim_type}: Standard liability limits apply"
        
        # Create agent with tools
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        tools = [search_claims, check_regulations]
        llm_with_tools = llm.bind_tools(tools)
        
        # Test complex query
        complex_query = "Find similar slip and fall claims in California and check applicable regulations"
        
        print(f"📝 Running complex query: '{complex_query}'")
        print("⏳ Executing workflow with tools...")
        
        messages = [
            HumanMessage(content=complex_query)
        ]
        
        start_time = time.time()
        
        # Step 1: Initial response with tool calls
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Step 2: Execute tools if called
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔧 Executing {len(response.tool_calls)} tool calls...")
            
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'search_claims':
                    result = search_claims.invoke(tool_call['args'])
                elif tool_call['name'] == 'check_regulations':
                    result = check_regulations.invoke(tool_call['args'])
                else:
                    result = "Unknown tool"
                
                print(f"   Tool '{tool_call['name']}' executed")
        
        duration = time.time() - start_time
        
        print(f"✅ Workflow completed in {duration:.2f}s")
        print(f"📊 Total messages: {len(messages)}")
        
        project_name = os.getenv("LANGCHAIN_PROJECT", "insurance-rag-test")
        print(f"\n🌐 View workflow trace at: https://smith.langchain.com/o/default/{project_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        return False

def test_adjuster_agent_with_langsmith():
    """Test the actual adjuster agent with LangSmith monitoring."""
    
    print("\n🧪 Testing Adjuster Agent with LangSmith...")
    print("-" * 50)
    
    try:
        # Import and initialize the debug agent
        from debug_adjuster_agent import DebugClaimsAdjusterAgent
        
        # Enable LangSmith tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "insurance-rag-adjuster"
        
        print("🤖 Initializing Debug Adjuster Agent with LangSmith...")
        agent = DebugClaimsAdjusterAgent(debug_mode=True)
        
        # Test query
        test_query = "What is the total exposure for claim GL-2024-0025?"
        
        print(f"📝 Running test query: '{test_query}'")
        print("⏳ Processing with full agent workflow...")
        
        start_time = time.time()
        result = agent.process_query(test_query)
        duration = time.time() - start_time
        
        print(f"✅ Agent query processed in {duration:.2f}s")
        print(f"📊 Session ID: {result.get('session_id', 'N/A')}")
        print(f"📊 Steps executed: {len(result.get('step_logs', []))}")
        print(f"💬 Answer preview: {result['answer'][:200]}...")
        
        # Performance metrics
        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            print(f"\n⚡ Performance Metrics:")
            print(f"   LLM Calls: {metrics.get('total_llm_calls', 0)}")
            print(f"   Tokens Used: {metrics.get('total_tokens', 0)}")
            print(f"   Total Time: {metrics.get('session_total_time', 0):.2f}s")
        
        project_name = os.getenv("LANGCHAIN_PROJECT", "insurance-rag-adjuster")
        print(f"\n🌐 View full agent trace at: https://smith.langchain.com/o/default/{project_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to test LangSmith integration."""
    
    print("🚀 LANGSMITH INTEGRATION TEST")
    print("=" * 50)
    
    # Step 1: Check configuration
    config_ok = check_langsmith_config()
    
    if not config_ok:
        print("\n⚠️ LangSmith configuration issues detected!")
        print("Please ensure:")
        print("1. LANGCHAIN_API_KEY is set in .env file")
        print("2. You have an active LangSmith account")
        print("3. Get your API key from: https://smith.langchain.com/settings")
        return
    
    # Step 2: Test simple chain
    simple_ok = test_simple_chain()
    
    # Step 3: Test complex workflow
    complex_ok = test_complex_workflow()
    
    # Step 4: Test actual agent
    agent_ok = test_adjuster_agent_with_langsmith()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("-" * 50)
    print(f"Configuration: {'✅' if config_ok else '❌'}")
    print(f"Simple Chain: {'✅' if simple_ok else '❌'}")
    print(f"Complex Workflow: {'✅' if complex_ok else '❌'}")
    print(f"Adjuster Agent: {'✅' if agent_ok else '❌'}")
    
    if all([config_ok, simple_ok, complex_ok, agent_ok]):
        print("\n🎉 All tests passed! LangSmith is working correctly.")
        print("\n📋 Next steps:")
        print("1. Check your traces at https://smith.langchain.com")
        print("2. Run automated_test_runner.py with monitoring enabled")
        print("3. Use debug_adjuster_agent.py for detailed debugging")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()