import os
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pg_tools import TOOLS

load_dotenv()

TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

store = {}
def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

system_prompt = ("system", f"""
### PERSONA
Você é o Assessor.AI — especialista em finanças e rotina. 
Seja objetivo, confiável, direto.

### TAREFAS
- Processar perguntas do usuário sobre finanças e agenda
- Identificar conflitos de compromissos
- Analisar entradas, gastos e dívidas
- Resumir saúde financeira e propor recomendações
- Consultar histórico quando relevante

### REGRAS
- Nunca inventar dados
- Responder curto, claro e utilizável
- Hoje é {today.isoformat()} (timezone: America/Sao_Paulo)

### FORMATO DE RESPOSTA
- <1 frase objetiva>
- *Recomendação*: <ação prática>
- *Acompanhamento* (opcional): <detalhe extra se necessário>

### HISTÓRICO DA CONVERSA
{{chat_history}}
""")

prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)

chain = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

while True:
    user_input = input("> ")
    if user_input.lower() in ("sair", "exit", "fim"):
        print("Encerrando...")
        break
    try:
        resposta = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "sessao_demo"}}
        )
        print(resposta["output"])
    except Exception as e:
        print("Erro:", e)
