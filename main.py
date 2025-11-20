from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chat_models import init_chat_model
from langchain_core.tools import Tool

# Initialize components
model = init_chat_model("qwen3:0.6b", model_provider="ollama")
search_tool = DuckDuckGoSearchRun()

def research_assistant():
    """Enhanced research agent with specialized tools"""
    
    tools = [
        Tool(
            name="web_search",
            func=search_tool.run,
            description="Search the web for current information, facts, and multiple perspectives on any topic"
        )
    ]
    
    # Create research-focused agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""You are an expert research analyst. Follow this methodology:

RESEARCH PROTOCOL:
1. TOPIC ANALYSIS: Break down the main topic into key subtopics
2. MULTI-ANGLE SEARCH: Search each subtopic from different perspectives
3. SOURCE SYNTHESIS: Combine information from multiple searches
4. CRITICAL ANALYSIS: Identify patterns, contradictions, and gaps
5. KNOWLEDGE INTEGRATION: Create comprehensive understanding

Always perform multiple searches to ensure depth and breadth of coverage."""
    )
    return agent

def deep_research(topic, subtopics=None):
    """Perform structured deep research"""
    
    researcher = research_assistant()
    
    if subtopics is None:
        # Auto-generate subtopics based on main topic
        subtopic_prompt = f"""
        Break down the topic '{topic}' into 3-5 key subtopics that should be researched.
        Provide only the subtopic names as a bullet list.
        """
        
        subtopics_response = researcher.invoke(
            {"messages": [{"role": "user", "content": subtopic_prompt}]}
        )
        print("Generated subtopics for research...")
    
    research_prompt = f"""
    Conduct DEEP RESEARCH on: {topic}
    
    RESEARCH INSTRUCTIONS:
    - Search for comprehensive overview and definition
    - Search for current developments and recent news
    - Search for different viewpoints or debates
    - Search for data, statistics, and evidence
    - Search for future trends and predictions
    
    OUTPUT REQUIREMENTS:
    • Executive Summary
    • Key Findings (with evidence)
    • Different Perspectives
    • Current Challenges
    • Future Implications
    • Recommended Further Research Areas
    
    Use multiple search queries to ensure comprehensive coverage.
    """
    
    print(f"🧠 Conducting deep research: {topic}")
    result = researcher.invoke(
        {"messages": [{"role": "user", "content": research_prompt}]}
    )
    
    return result

# Example implementation
if __name__ == "__main__":
    # Research topics
    topics = [
        "renewable energy storage technologies 2024",
        "neural networks in medical diagnosis",
        "sustainable agriculture innovations"
    ]
    
    for topic in topics:
        print(f"\n{'='*80}")
        print(f"RESEARCH TOPIC: {topic}")
        print(f"{'='*80}")
        
        try:
            research = deep_research(topic)
            final_output = research["messages"][-1].content
            print(f"\n📊 RESEARCH COMPLETED:")
            print(final_output)
            print(f"\n{'-'*80}")
            
            # Save to file
            filename = f"research_{topic.replace(' ', '_')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Research Report: {topic}\n")
                f.write("="*50 + "\n")
                f.write(final_output)
            print(f"💾 Research saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Research failed for '{topic}': {str(e)}")