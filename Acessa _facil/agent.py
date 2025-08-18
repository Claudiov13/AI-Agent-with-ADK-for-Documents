from google.adk import Agent
from google.adk.tools import VertexAiSearchTool


search_tool = VertexAiSearchTool(data_store_id="copyID google cloud APP")

# Cria o agente principal
root_agent = Agent(
    model="gemini-2.5-pro",
    name="xxxxxx",
    instruction="""
   Any instruction as you want
    """,
    tools=[search_tool]
)
 


