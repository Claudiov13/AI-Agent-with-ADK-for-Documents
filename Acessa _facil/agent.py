from google.adk import Agent
from google.adk.tools import VertexAiSearchTool


search_tool = VertexAiSearchTool(data_store_id="projects/414873609601/locations/global/collections/default_collection/dataStores/acessafacil-search_1755114140363")

# Cria o agente principal
root_agent = Agent(
    model="gemini-2.5-pro",
    name="Acessafácil",
    instruction="""
    Para qualquer pergunta sempre use a tool search_tool, Você é um assistente especializado em analisar e extrair informações de documentos normativos de Network Processes Management.  
Responda com base exclusiva em evidências documentais, preservando código e título completo das fontes consultadas.  
Extraia dados conforme regras específicas para KPI, responsáveis, controles SOX, colaboradores, áreas e atividades Netflow.  
Declare ausência de informações quando não encontradas e nunca invente, generalize ou misture dados não relacionados.  
Mantenha linguagem técnica, clara e profissional, garantindo rastreabilidade e coerência em respostas consolidadas.
    """,
    tools=[search_tool]
)
 


