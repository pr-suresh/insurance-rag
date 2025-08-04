#!/usr/bin/env python3
"""
Automated Test Query Runner for Agentic RAG System
Runs comprehensive test queries and validates outputs automatically
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from enhanced_rag import EnhancedInsuranceRAG
from debug_adjuster_agent import DebugClaimsAdjusterAgent
from langsmith_setup import setup_langsmith_monitoring
from dotenv import load_dotenv

load_dotenv()

class AutomatedTestRunner:
    """Automated test runner for comprehensive RAG system validation."""
    
    def __init__(self, use_debug_agent=True, enable_monitoring=True):
        print("🚀 Initializing Automated Test Runner...")
        
        self.use_debug_agent = use_debug_agent
        self.test_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up monitoring
        if enable_monitoring:
            self.monitor = setup_langsmith_monitoring(f"automated-tests-{self.test_session_id}")
        else:
            self.monitor = None
        
        # Initialize systems
        if use_debug_agent:
            print("🤖 Loading Debug Adjuster Agent...")
            self.agent = DebugClaimsAdjusterAgent(debug_mode=True)
            self.rag = None
        else:
            print("📚 Loading Basic RAG System...")
            self.rag = EnhancedInsuranceRAG(use_reranking=False)
            self.rag.ingest_data("data/insurance_claims.csv")
            self.agent = None
        
        self.test_results = []
        print("✅ Test runner initialized")
    
    def load_test_queries(self) -> List[Dict]:
        """Load comprehensive test queries for validation."""
        
        test_queries = [
            # Basic functionality tests
            {
                "category": "basic_search",
                "query": "What is the total exposure for claim GL-2024-0025?",
                "expected_elements": ["GL-2024-0025", "$350,000", "cyber liability"],
                "timeout": 30
            },
            {
                "category": "basic_search", 
                "query": "Which company was involved in the carbon monoxide case?",
                "expected_elements": ["Superior Mechanical Systems", "carbon monoxide", "GL-2024-0024"],
                "timeout": 30
            },
            
            # Reasoning and analysis tests
            {
                "category": "reasoning",
                "query": "What factors contributed to the high liability in the carbon monoxide case?",
                "expected_elements": ["95%", "installation", "negligence", "health"],
                "timeout": 60
            },
            {
                "category": "reasoning",
                "query": "Does the slip and fall claim violate Virginia liability rules?",
                "expected_elements": ["Virginia", "liability", "premises", "regulations"],
                "timeout": 60
            },
            
            # Complex multi-step tests
            {
                "category": "complex",
                "query": "What are the common patterns in high-value claims across different types?",
                "expected_elements": ["professional", "cyber", "premises", "patterns"],
                "timeout": 90
            },
            {
                "category": "complex",
                "query": "How should I handle the carbon monoxide case from a compliance perspective?",
                "expected_elements": ["compliance", "carbon monoxide", "regulations", "safety"],
                "timeout": 90
            },
            
            # Edge cases and error handling
            {
                "category": "edge_case",
                "query": "What is the status of claim XYZ-9999-0000?",
                "expected_elements": ["no results", "not found", "unavailable"],
                "timeout": 30
            },
            {
                "category": "edge_case",
                "query": "",  # Empty query
                "expected_elements": ["error", "empty", "invalid"],
                "timeout": 15
            },
            
            # State-specific legal research
            {
                "category": "legal",
                "query": "What are the Texas insurance requirements for cyber liability claims?",
                "expected_elements": ["Texas", "cyber", "insurance", "requirements"],
                "timeout": 90
            },
            {
                "category": "legal",
                "query": "California premises liability standards for slip and fall cases",
                "expected_elements": ["California", "premises", "liability", "slip"],
                "timeout": 90
            }
        ]
        
        print(f"📋 Loaded {len(test_queries)} test queries across {len(set(q['category'] for q in test_queries))} categories")
        return test_queries
    
    def run_single_test(self, test_case: Dict) -> Dict[str, Any]:
        """Run a single test case and validate results."""
        
        query = test_case["query"]
        category = test_case["category"]
        expected = test_case["expected_elements"]
        timeout = test_case.get("timeout", 60)
        
        print(f"🧪 Testing [{category}]: {query[:50]}...")
        
        start_time = time.time()
        test_result = {
            "query": query,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.test_session_id
        }
        
        try:
            # Run the query with timeout
            if self.use_debug_agent and query.strip():  # Skip empty queries for agent
                result = self.agent.process_query(query)
                answer = result.get("answer", "")
                performance_metrics = result.get("performance_metrics", {})
                step_logs = result.get("step_logs", [])
            elif self.rag and query.strip():
                search_results = self.rag.hybrid_search(query, k=5)
                answer = f"Found {len(search_results)} results: " + (search_results[0]['content'][:200] if search_results else "No results")
                performance_metrics = {}
                step_logs = []
            else:
                answer = "Error: Empty query or system unavailable"
                performance_metrics = {}
                step_logs = []
            
            duration = time.time() - start_time
            
            # Validate answer contains expected elements
            validation_results = self._validate_answer(answer, expected)
            
            test_result.update({
                "status": "passed" if validation_results["passed"] else "failed",
                "duration": duration,
                "answer": answer,
                "answer_length": len(answer),
                "validation": validation_results,
                "performance_metrics": performance_metrics,
                "step_count": len(step_logs),
                "timeout_exceeded": duration > timeout
            })
            
            # Log to monitoring if available
            if self.monitor:
                self.monitor.trace_agent_workflow(
                    session_id=self.test_session_id,
                    step=f"test_{category}",
                    input_data={"query": query},
                    output_data={"status": test_result["status"], "duration": duration}
                )
            
            status_emoji = "✅" if test_result["status"] == "passed" else "❌"
            print(f"  {status_emoji} {test_result['status'].upper()} in {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result.update({
                "status": "error",
                "duration": duration,
                "error": str(e),
                "answer": "",
                "validation": {"passed": False, "error": str(e)}
            })
            
            print(f"  ❌ ERROR: {str(e)}")
        
        return test_result
    
    def _validate_answer(self, answer: str, expected_elements: List[str]) -> Dict[str, Any]:
        """Validate that answer contains expected elements."""
        
        answer_lower = answer.lower()
        validation_results = {
            "expected_elements": expected_elements,
            "found_elements": [],
            "missing_elements": [],
            "passed": False
        }
        
        for element in expected_elements:
            if element.lower() in answer_lower:
                validation_results["found_elements"].append(element)
            else:
                validation_results["missing_elements"].append(element)
        
        # Pass if at least 50% of expected elements are found
        found_ratio = len(validation_results["found_elements"]) / len(expected_elements) if expected_elements else 0
        validation_results["found_ratio"] = found_ratio
        validation_results["passed"] = found_ratio >= 0.5
        
        return validation_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all automated tests and generate comprehensive report."""
        
        print(f"\n🎯 STARTING AUTOMATED TEST SUITE")
        print("=" * 60)
        print(f"Session ID: {self.test_session_id}")
        print(f"System: {'Debug Agent' if self.use_debug_agent else 'Basic RAG'}")
        print("=" * 60)
        
        test_queries = self.load_test_queries()
        start_time = time.time()
        
        # Run all tests
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}]", end=" ")
            result = self.run_single_test(test_case)
            self.test_results.append(result)
        
        total_duration = time.time() - start_time
        
        # Generate report
        report = self._generate_test_report(total_duration)
        
        # Save results
        self._save_test_results()
        
        # Save monitoring traces
        if self.monitor:
            self.monitor.save_local_traces()
        
        return report
    
    def _generate_test_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.get("status") == "passed"])
        failed_tests = len([r for r in self.test_results if r.get("status") == "failed"])
        error_tests = len([r for r in self.test_results if r.get("status") == "error"])
        
        # Category analysis
        category_stats = {}
        for result in self.test_results:
            cat = result.get("category", "unknown")
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0, "failed": 0, "errors": 0}
            
            category_stats[cat]["total"] += 1
            category_stats[cat][result.get("status", "error") + ("d" if result.get("status") != "error" else "s")] += 1
        
        # Performance analysis
        durations = [r.get("duration", 0) for r in self.test_results if r.get("duration")]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        report = {
            "session_id": self.test_session_id,
            "timestamp": datetime.now().isoformat(),
            "system_type": "debug_agent" if self.use_debug_agent else "basic_rag",
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "total_duration": total_duration,
                "avg_test_duration": avg_duration
            },
            "category_breakdown": category_stats,
            "detailed_results": self.test_results
        }
        
        self._print_test_report(report)
        return report
    
    def _print_test_report(self, report: Dict[str, Any]):
        """Print formatted test report to console."""
        
        print(f"\n\n📊 TEST EXECUTION REPORT")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"📋 SUMMARY")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']} ✅")
        print(f"  Failed: {summary['failed']} ❌")
        print(f"  Errors: {summary['errors']} 💥")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Total Duration: {summary['total_duration']:.2f}s")
        print(f"  Avg Test Duration: {summary['avg_test_duration']:.2f}s")
        
        print(f"\n📊 CATEGORY BREAKDOWN")
        for category, stats in report["category_breakdown"].items():
            success_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"  {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        print(f"\n🔍 FAILED/ERROR TESTS:")
        for result in self.test_results:
            if result.get("status") in ["failed", "error"]:
                query = result.get("query", "")[:50]
                status = result.get("status", "unknown")
                category = result.get("category", "unknown")
                print(f"  ❌ [{category}] {query}... ({status})")
        
        print("\n" + "=" * 60)
    
    def _save_test_results(self):
        """Save test results to files."""
        
        # Save JSON report
        json_filename = f"test_results_{self.test_session_id}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Save CSV summary
        csv_filename = f"test_summary_{self.test_session_id}.csv"
        df_data = []
        for result in self.test_results:
            df_data.append({
                "query": result.get("query", "")[:100],
                "category": result.get("category", ""),
                "status": result.get("status", ""),
                "duration": result.get("duration", 0),
                "validation_passed": result.get("validation", {}).get("passed", False),
                "answer_length": result.get("answer_length", 0)
            })
        
        pd.DataFrame(df_data).to_csv(csv_filename, index=False)
        
        print(f"💾 Results saved:")
        print(f"  JSON: {json_filename}")
        print(f"  CSV: {csv_filename}")

def main():
    """Main function to run automated tests."""
    
    print("🚀 AUTOMATED AGENTIC RAG TEST SUITE")
    print("=" * 50)
    
    # Choose system to test
    use_agent = input("Test Debug Agent? (y/n, default=y): ").lower() != 'n'
    enable_monitoring = input("Enable LangSmith monitoring? (y/n, default=y): ").lower() != 'n'
    
    # Initialize and run tests
    runner = AutomatedTestRunner(
        use_debug_agent=use_agent,
        enable_monitoring=enable_monitoring
    )
    
    report = runner.run_all_tests()
    
    print(f"\n🎉 Automated testing complete!")
    print(f"📊 Success Rate: {report['summary']['success_rate']:.1f}%")
    
    if report['summary']['success_rate'] >= 80:
        print("✅ EXCELLENT: System performing well across test categories")
    elif report['summary']['success_rate'] >= 60:
        print("⚠️ MODERATE: Some issues detected, review failed tests")
    else:
        print("❌ CRITICAL: Significant issues detected, immediate attention needed")

if __name__ == "__main__":
    main()