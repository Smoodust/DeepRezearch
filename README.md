# DeepRezearch

DeepRezearch a simple multi agent system for **deep rezearch🥴**

## Overview
DeepRezearch is multi-agent system that helps you complete simple, multi-step tasks requiring information lookup, calculations, or a combination of both. System coordinates a set of specialized agents through an orchestrator agent. The orchestrator routes queries to the right agents and composes results

## Agents
- Orchestrator agent routes requests and composes outputs
- Research agent performs information retrieval and summarization
- Coding agent generates and verifies code snippets
- Synthesis agent combines results and produces final responses

## Architecture
```mermaid
flowchart TD
  Orchestrator[Orchestrator Agent]
  Research[Research Agent]
  Coding[Coding Agent]
  Synthesis[Synthesis Agent]

  Orchestrator --> Research
  Orchestrator --> Coding
  Orchestrator --> Synthesis

  Research --> Orchestrator
  Coding --> Orchestrator
  Synthesis --> Orchestrator
```

## How to run
```bash
docker compose up -d
```

The compose file starts an Ollama model service and the FastAPI app

## Notes
- The project expects an Ollama model named llama3.1 to be available when using docker compose
- Adjust model urls and names in environment variables or in main.py if needed

Developed with ❤️, ☕️ and a little bit with 🤬 for mr. Pudikov