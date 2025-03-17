import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools


load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")



search_agent=Agent(
    tools=[
        DuckDuckGoTools(
        news=True,
        search=True,
        fixed_max_results=10
        )
        ],
    model = Gemini("gemini-2.0-flash"),
    markdown=True,
    show_tool_calls=True,
)



search_agent.print_response("What are the latest news related to AI")