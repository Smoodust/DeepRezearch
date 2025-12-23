import asyncio
import os
import sys

from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.coding_agent import CodingAgent
from agents.research_agent import ResearchAgent
from agents.synthesis_agent import SynthesisAgent
from core.orchestrator import WorkflowOrchestrator

logger.add("agents.log")


class Orchestrator:
    def __init__(self):
        self.orchestrator = WorkflowOrchestrator(model_name="llama3.1:8b")

    async def setup_workflows(self):
        """Initialize all workflows"""

        # Initialize coding workflow
        coding_agent = CodingAgent(model_name="llama3.1:8b")
        search_agent = ResearchAgent(
            model_name="llama3.1:8b", max_result=5, n_queries=2
        )
        synthesis_agent = SynthesisAgent(model_name="llama3.1:8b")
        self.orchestrator.register_workflow(coding_agent)
        self.orchestrator.register_workflow(search_agent)
        self.orchestrator.register_workflow(synthesis_agent)

    async def run_test_cases(self):
        """Run test scenarios"""

        test_cases = [
            (
                "Research the average annual inflation rate in the United States for the last three full "
                "calendar years (e.g., 2022, 2023, 2024) using reliable public sources. "
                "Then assume an employee with a starting salary of $60,000 receives a nominal salary increase "
                "of 4% per year. Using the researched inflation data, calculate the real (inflation-adjusted) "
                "salary at the end of each year and the real wage growth or loss for each year. "
                "Explain the formula used and present the results in a comparison table including: "
                "year, nominal salary, inflation rate, real salary, and real wage growth percentage. "
                "If final data for the most recent year is unavailable, clearly state any assumptions made."
            ),
            # "What's the highway speed limit in Germany in km/h? Convert it to miles per hour for comparison with US limits."
            # "Find the number of Nobel laureates in 2024 and 2025. What is the difference in percentage?",
            # """Research the current 10-year US Treasury bond yield and S&P 500 average dividend yield. Then calculate the equity risk premium and create a comparison table for the last 5 years if you can find the historical data.""",
            # "Hi! How are you?",
            # "Calculate 2**222 in python REPL. Provide me an output",
            # "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0. Write Python code to solve the task",
            # "What's the standard GPA scale (A=4.0, etc.)? Calculate the GPA for someone with 3 A's, 2 B's, and 1 C in a semester."
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
                print(f"{result}")

            except Exception as e:
                logger.error(f"ERROR: {str(e)}")


async def main():
    """Main testing function"""
    tester = Orchestrator()

    logger.info("🔄 Initializing workflows...")
    await tester.setup_workflows()

    logger.info("🚀 Running test scenarios...")
    await tester.run_test_cases()


if __name__ == "__main__":
    asyncio.run(main())
