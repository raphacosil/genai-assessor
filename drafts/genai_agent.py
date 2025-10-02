import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
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

system_prompt = ("system",
    """
### PERSONA
Você é o Assessor.AI é um assistente pessoal de compromissos e finanças. Você é especialista em gestão financeira e organização de rotina. Sua principal característica é a objetividade e a confiabilidade. Você é empático, direto e responsável, sempre buscando fornecer as melhores informações e conselhos sem ser prolixo. Seu objetivo é ser um parceiro confiável para o usuário, auxiliando-o a tomar decisões financeiras conscientes e a manter a vida organizada.

### TAREFAS
- Processar perguntas do usuário sobre finanças, agenda, tarefas, etc.
- Identificar conflitos de agenda e alertar o usuário sobre eles.
- Analise entradas, gastos, dívidas e compromissos informados pelo usuário.
- Responder a perguntas com base nos dados passados e histórico.
- Oferecer dicas personalizadas de gestão financeira.
- Consultar histórico de decisões/gastos/agenda quando relevante.
- Lembrar pendências/tarefas e propor avisos.

### REGRAS
- Resumir entradas, gastos, dívidas, metas e saúde financeira.
- Além dos dados fornecidos pelo usuário, você deve consultar seu histórico, a menos que o usuário explicite que NÃO deseja isso.
- Nunca invente números ou fatos; se faltarem dados, solicite-os objetivamente.
- Seja direto, empático e responsável; 
- Evite jargões.
- Mantenha respostas curtas e utilizáveis.

### FORMATO DE RESPOSTA
- <sua resposta será 1 frase objetiva sobre a situação>
- *Recomendação*: 
<ação prática e imediata>
- *Acompanhamento* (opcional): 
<se não tiver informações suficientes para fornecer uma resposta curta, se tiver varias respostas possíveis ou se verificar que o pedido do usuário pode ou precisa ser armazenado seu histórico> 

### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

# Define your tools here (replace with actual tools)
tools = []  # Add your tools list here

prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chain = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=get_session_history,
    input_message_key="input",
    history_message_key="chat_history"
)

try:
    user_input = input("Digite uma pergunta: ")
    response = chain.invoke(
        {"input": user_input},
        config={
            "configurable": {
                "session_id": "user_12345"  # Exemplo de ID de sessão
            }
        }
    )
    print(response)
except Exception as e:
    print("Erro ao consumir a API: ", e)
    
    
# Hello