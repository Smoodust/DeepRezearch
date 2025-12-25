import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from agents.coding_agent.code_builder import CodingAgentBuilder
from agents.orchestrator.orchestrator_builder import OrchestratorAgentBuilder
from agents.research_agent.research_builder import ResearchAgentBuilder
from agents.synthesis_agent.synthesis_builder import SynthesisAgentBuilder

from agents.base_agent import BaseAgentOutput

from loguru import logger


agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent

    logger.info("Aplication startup")

    try:
        search_agent = ResearchAgentBuilder(model_name="llama3.1:8b")
        coding_agent = CodingAgentBuilder(model_name="llama3.1:8b")
        synthesis_agent = SynthesisAgentBuilder(model_name="llama3.1:8b")

        agent = OrchestratorAgentBuilder(
            model_name="llama3.1:8b",
            agents_to_build=[search_agent, coding_agent, synthesis_agent],
        ).build()

    except Exception as e:
        logger.error(f"Startup error: {e}")
        agent = None

    yield

    logger.info("Shut down")


app = FastAPI(
    title="DeepRezearch API",
    description="API for Perplexity na minimalkax",
    version="1.0.0",
    openapi_tags=[{"name": "DeepRezearch", "description": "Perplexity na minimalkax"}],
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether processing was successful")
    input: str = Field(..., description="Original user query")
    output: str = Field(..., description="Processed response from agent system")
    timestamp: float = Field(..., description="Processing completion timestamp")
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional metadata about the processing"
    )


@app.post("/query",
          status_code=200,
          response_model=QueryResponse,
          summary="Process user query",
          description="""Process a user query through the multi-agent system.
          
          The query is routed through an orchestrator that determines which specialized agents 
          (research, coding, or synthesis) should handle different aspects of the request.
          
          Returns a structured response containing the processed output.""", 
          tags=["DeepRezearch"])
async def process_query(
    request: QueryRequest = Body(
        ...,
        description="User query to be processed by the agent system",
        example={
            "message": "Hi! How are you?"
        }
    )
) -> QueryResponse:
    try:
        result: BaseAgentOutput = await agent.run({"workflow_input": request.message})

        return {
            "success": True,
            "input": request.message,
            "output": result.get("output", ""),
            "timestamp": asyncio.get_event_loop().time(),
        }

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
