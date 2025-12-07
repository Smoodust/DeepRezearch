import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coding_agent import CodingAgent
from core.orchestrator import WorkflowOrchestrator
from core.workflows.coding_wf import CodeWorkflow
from core.workflows.research_wf import ResearchWorkflow


class TestOrchestrator:
    def __init__(self):
        self.orchestrator = WorkflowOrchestrator()

    async def setup_workflows(self):
        """Initialize all workflows"""

        # Initialize coding workflow
        coding_agent = CodingAgent(name="python_coder", model="llama3.1:8b")
        coding_workflow = CodeWorkflow(coding_agent)
        self.orchestrator.register_workflow("coding", coding_workflow)

        # Initialize research workflow (stub)
        research_workflow = ResearchWorkflow(None)
        self.orchestrator.register_workflow("research", research_workflow)

    async def run_test_cases(self):
        """Run test scenarios"""

        test_cases = [
            # "Hi! How are you?",
            "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0. Write Python code to solve the task",
            # "Research modern approaches to machine learning",
            # "Explain what polymorphism is in OOP",
            # "Create a class for working with SQLite database",
            # "Analyze the advantages and disadvantages of microservices architecture",
            # "Write tests for calculate_average function",
            # "How does garbage collection work in Python?",
            # "Implement a REST API endpoint for user authentication",
        ]

        for i, user_input in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {user_input}")
            print(f"{'='*60}")

            try:
                result = await self.orchestrator.process_request(user_input)

                print(f"ORCHESTRATOR DECISION:")
                print(f"  Workflow: {result['decision'].workflow_type}")
                print(f"  Reasoning: {result['decision'].reasoning}")
                print(f"  Confidence: {result['decision'].confidence:.2f}")
                print(f"  Result: {result['result']}")

            except Exception as e:
                print(f"ERROR: {str(e)}")

    async def interactive_mode(self):
        """Interactive mode for testing"""
        print("\n🎯 ORCHESTRATOR INTERACTIVE MODE")
        print("Type 'quit' to exit")

        while True:
            try:
                user_input = input("\n🧠 Your request: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    break

                if not user_input:
                    continue

                print("🤔 Analyzing request...")
                result = await self.orchestrator.process_request(user_input)

                print(f"\n📋 RESULT:")
                print(f"   🔧 Workflow: {result['decision'].workflow_type}")
                print(f"   💭 Reasoning: {result['decision'].reasoning}")
                print(f"   ✅ Confidence: {result['decision'].confidence:.2f}")
                print(f"\n   📝 Response:")
                print(f"   {result['result']}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


async def main():
    """Main testing function"""
    tester = TestOrchestrator()

    print("🔄 Initializing workflows...")
    await tester.setup_workflows()

    print("🚀 Running test scenarios...")
    await tester.run_test_cases()

    print("\n🎮 Starting interactive mode...")
    await tester.interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
