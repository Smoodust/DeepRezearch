import asyncio
import os
import sys

from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coding_agent import CodingAgent
from core.orchestrator import WorkflowOrchestrator


class TestOrchestrator:
    def __init__(self):
        self.orchestrator = WorkflowOrchestrator()

    async def setup_workflows(self):
        """Initialize all workflows"""

        # Initialize coding workflow
        coding_agent = CodingAgent(name="python_coder", model_name="llama3.1:8b")
        self.orchestrator.register_workflow("coding", coding_agent)

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
            logger.debug(f"\n{'='*60}")
            logger.debug(f"TEST {i}: {user_input}")
            logger.debug(f"{'='*60}")

            try:
                result = await self.orchestrator.process_request(user_input)

                logger.success(f"ORCHESTRATOR DECISION:")
                logger.success(f"  Workflow: {result['decision'].workflow_type}")
                logger.success(f"  Reasoning: {result['decision'].reasoning}")
                logger.success(f"  Confidence: {result['decision'].confidence:.2f}")
                logger.success(f"  Result: {result['result']}")

            except Exception as e:
                logger.error(f"ERROR: {str(e)}")

    async def interactive_mode(self):
        """Interactive mode for testing"""
        logger.debug("\n🎯 ORCHESTRATOR INTERACTIVE MODE")
        logger.debug("Type 'quit' to exit")

        while True:
            try:
                user_input = input("\n🧠 Your request: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    break

                if not user_input:
                    continue

                logger.debug("🤔 Analyzing request...")
                result = await self.orchestrator.process_request(user_input)

                logger.success(f"\n📋 RESULT:")
                logger.success(f"   🔧 Workflow: {result['decision'].workflow_type}")
                logger.success(f"   💭 Reasoning: {result['decision'].reasoning}")
                logger.success(f"   ✅ Confidence: {result['decision'].confidence:.2f}")
                logger.success(f"\n   📝 Response:")
                logger.success(f"   {result['result']}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Error: {str(e)}")


async def main():
    """Main testing function"""
    tester = TestOrchestrator()

    logger.info("🔄 Initializing workflows...")
    await tester.setup_workflows()

    logger.info("🚀 Running test scenarios...")
    await tester.run_test_cases()

    logger.info("\n🎮 Starting interactive mode...")
    await tester.interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
