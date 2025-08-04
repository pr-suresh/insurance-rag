#!/usr/bin/env python3
"""
LangSmith Local Monitoring Setup for Agentic RAG
Provides comprehensive tracking of LangGraph workflows locally
"""

import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.run_helpers import traceable
from typing import Dict, Any, List
import json
from datetime import datetime

load_dotenv()

class LangSmithLocalMonitor:
    """Local LangSmith monitoring setup with fallback to local logging."""
    
    def __init__(self, project_name="insurance-rag-debug"):
        self.project_name = project_name
        self.local_mode = False
        
        # Try to initialize LangSmith client
        try:
            if os.getenv("LANGCHAIN_API_KEY"):
                self.client = Client()
                self.client.create_project(project_name=project_name, if_exists="ignore")
                print(f"✅ LangSmith connected - Project: {project_name}")
                print(f"🌐 View traces at: https://smith.langchain.com/projects/{project_name}")
            else:
                raise ValueError("No LANGCHAIN_API_KEY found")
                
        except Exception as e:
            print(f"⚠️ LangSmith unavailable ({e}), using local monitoring")
            self.local_mode = True
            self._setup_local_monitoring()
    
    def _setup_local_monitoring(self):
        """Set up local trace logging as fallback."""
        
        self.local_traces = []
        self.trace_file = f"langsmith_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        print(f"📁 Local traces will be saved to: {self.trace_file}")
    
    def configure_environment(self):
        """Configure environment variables for LangSmith tracing."""
        
        if not self.local_mode:
            # Set LangSmith environment variables
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            
            print("🔧 LangSmith environment configured:")
            print(f"  - LANGCHAIN_TRACING_V2: true")
            print(f"  - LANGCHAIN_PROJECT: {self.project_name}")
        else:
            # Disable LangSmith tracing for local mode
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            print("🔧 Local monitoring mode - LangSmith tracing disabled")
    
    @traceable(name="rag_search")
    def trace_rag_search(self, query: str, results: List[Dict], metadata: Dict = None):
        """Trace RAG search operations."""
        
        trace_data = {
            "operation": "rag_search",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results_count": len(results),
            "metadata": metadata or {}
        }
        
        if self.local_mode:
            self.local_traces.append(trace_data)
        
        return trace_data
    
    @traceable(name="legal_research")
    def trace_legal_research(self, query: str, state: str, results: Dict, metadata: Dict = None):
        """Trace legal research operations."""
        
        trace_data = {
            "operation": "legal_research",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "state": state,
            "results": results,
            "metadata": metadata or {}
        }
        
        if self.local_mode:
            self.local_traces.append(trace_data)
        
        return trace_data
    
    @traceable(name="agent_workflow")
    def trace_agent_workflow(self, session_id: str, step: str, input_data: Dict, output_data: Dict):
        """Trace complete agent workflow steps."""
        
        trace_data = {
            "operation": "agent_workflow",
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "step": step,
            "input": input_data,
            "output": output_data
        }
        
        if self.local_mode:
            self.local_traces.append(trace_data)
        
        return trace_data
    
    def save_local_traces(self):
        """Save local traces to file."""
        
        if self.local_mode and self.local_traces:
            try:
                with open(self.trace_file, 'w') as f:
                    json.dump(self.local_traces, f, indent=2)
                
                print(f"💾 Saved {len(self.local_traces)} traces to {self.trace_file}")
                
            except Exception as e:
                print(f"❌ Failed to save traces: {e}")
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of traces for analysis."""
        
        if self.local_mode:
            traces = self.local_traces
        else:
            # For LangSmith, we'd query the API here
            traces = []
        
        operations = {}
        for trace in traces:
            op = trace.get("operation", "unknown")
            operations[op] = operations.get(op, 0) + 1
        
        return {
            "total_traces": len(traces),
            "operations": operations,
            "start_time": traces[0]["timestamp"] if traces else None,
            "end_time": traces[-1]["timestamp"] if traces else None
        }

def setup_langsmith_monitoring(project_name="insurance-rag-debug") -> LangSmithLocalMonitor:
    """Initialize LangSmith monitoring for the project."""
    
    print("🔍 Setting up LangSmith monitoring...")
    
    monitor = LangSmithLocalMonitor(project_name)
    monitor.configure_environment()
    
    return monitor

def create_langsmith_config():
    """Create .env configuration for LangSmith."""
    
    config_lines = [
        "",
        "# LangSmith Configuration for Local Monitoring",
        "# Get your API key from: https://smith.langchain.com/settings",
        "LANGCHAIN_API_KEY=your_langsmith_api_key_here",
        "LANGCHAIN_TRACING_V2=true",
        "LANGCHAIN_PROJECT=insurance-rag-debug",
        ""
    ]
    
    # Check if .env exists and add config
    env_path = ".env"
    
    try:
        # Read existing .env
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                existing_content = f.read()
        else:
            existing_content = ""
        
        # Check if LangSmith config already exists
        if "LANGCHAIN_API_KEY" not in existing_content:
            with open(env_path, 'a') as f:
                f.write('\n'.join(config_lines))
            
            print(f"📝 Added LangSmith configuration to {env_path}")
            print("📋 Next steps:")
            print("  1. Get API key from https://smith.langchain.com/settings")
            print("  2. Replace 'your_langsmith_api_key_here' in .env file")
            print("  3. Restart your application to enable tracing")
        else:
            print("✅ LangSmith configuration already exists in .env")
    
    except Exception as e:
        print(f"❌ Failed to create LangSmith config: {e}")

def test_langsmith_setup():
    """Test LangSmith monitoring setup."""
    
    print("🧪 Testing LangSmith monitoring setup...")
    
    monitor = setup_langsmith_monitoring("test-project")
    
    # Test trace operations
    monitor.trace_rag_search(
        query="test query",
        results=[{"content": "test result"}],
        metadata={"test": True}
    )
    
    monitor.trace_legal_research(
        query="test legal query",
        state="California",
        results={"found": True},
        metadata={"test": True}
    )
    
    monitor.trace_agent_workflow(
        session_id="test_session",
        step="test_step",
        input_data={"query": "test"},
        output_data={"result": "success"}
    )
    
    # Get summary
    summary = monitor.get_trace_summary()
    print(f"📊 Trace Summary: {summary}")
    
    # Save traces if in local mode
    monitor.save_local_traces()
    
    print("✅ LangSmith setup test complete!")

if __name__ == "__main__":
    print("🚀 LANGSMITH LOCAL MONITORING SETUP")
    print("=" * 50)
    
    # Create configuration
    create_langsmith_config()
    
    # Test setup
    test_langsmith_setup()
    
    print("\n📋 USAGE INSTRUCTIONS:")
    print("-" * 30)
    print("1. Import: from langsmith_setup import setup_langsmith_monitoring")
    print("2. Initialize: monitor = setup_langsmith_monitoring()")
    print("3. Use @traceable decorator on functions to trace")
    print("4. View traces in LangSmith dashboard or local JSON files")