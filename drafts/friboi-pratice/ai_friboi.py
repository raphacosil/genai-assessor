import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

CORTES_VALIDOS = [
    "acém", "alcatra", "capa do file", "contrafilé", "costela", "coxão duro", "coxão mole",
    "file mignon", "fraldinha", "lagarto", "maminha", "miúdos", "musculo", "paleta",
    "patinho", "peito", "picanha", "ponta do contrafile"
]
CORTES_VALIDOS_NORMALIZADOS = {c.lower().strip(): c for c in CORTES_VALIDOS}

# =========================
# CONEXÃO COM MONGODB (RAG)
# =========================
mongo_client = MongoClient("mongodb://localhost:27017")
db = mongo_client["friboi_pratica"]
receitas_collection = db["receitas_pratica"]

def buscar_no_mongo(corte: str):
    """
    implemente essa função para retornar pelo menos 3 receitas que tenham o corte que o agente_verificador encontrou
    """
    query = {"Corte": {"$regex": corte, "$options": "i"}}
    docs = list(receitas_collection.find(query).limit(3))
    return docs
    
# =========================
# MEMÓRIA (por sessão/agente)
# =========================
store = {}
def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# =========================
# AGENTE 1: Verificador de Corte
# =========================
prompt_verificador = ChatPromptTemplate.from_messages([
    ("system", """
Você é um chatbot da Friboi. Seu objetivo é conversar de forma amigável com o usuário e tentar identificar se ele menciona um corte de carne.

Se mencionar um dos seguintes cortes: {cortes}, retorne apenas o nome do corte identificado (sem comentários).
Se não mencionar nenhum corte, responda normalmente com uma saudação, comentário ou pergunta simpática que estimule a continuação da conversa.

Cortes válidos: {cortes}
"""),
    MessagesPlaceholder(variable_name="historico"),  # memória do agente
    ("human", "{input}")
])

cadeia_verificador = prompt_verificador.partial(cortes=", ".join(CORTES_VALIDOS)) | llm | StrOutputParser()

# Wrap com memória por sessão
agente_verificador = RunnableWithMessageHistory(
    cadeia_verificador,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="historico",
)

# =========================
# AGENTE 2: Gerador de Receita (com RAG)
# =========================
# Abaixo crie o system prompt para esse agente

prompt_gerador = ChatPromptTemplate.from_messages([
    (
        "system",
        """
            ### OBJETIVO
            Interpretar a PERGUNTA_ORIGINAL sobre receitas de carne e gerar uma resposta breve e completa 
            (ingredientes + modo de preparo).  
            Usar exclusivamente o contexto do MongoDB (contexto_rag).  
            Responder sempre como um chef simpático.

            ### TAREFAS
            1. Usar o contexto_rag para descrever a receita de forma clara e prática.  
            2. Não inventar informações que não estejam no contexto.
            3. Estruturar a resposta em duas seções: **Ingredientes** e **Modo de preparo**.  

            ### CONTEXTO
            - Corte atual: {corte_atual}
            - contexto_rag: {contexto_rag}
            - PERSONA: chef simpático e experiente  

            ### REGRAS
            - Ser breve, mas completo.  
            - Não criar ingredientes ou etapas inexistentes.  
            - Caso o contexto esteja vazio ou insuficiente, responda algo como:
            "Não encontrei receitas com esse corte agora, mas posso sugerir outras opções."
            - Tom profissional, cordial e seguro.
            - Não use emojis.

            ### SAÍDA
            Retorne apenas o texto da receita formatado como explicação de um chef simpático.
"""
    ),
    MessagesPlaceholder("historico"),
    ("human", "{input}")
])

# Abaixo crie a chain para esse agente
cadeia_receita = prompt_gerador | llm | StrOutputParser()

# Abaixo crie o agente_gerador, que vai buscar o corte e responder com a receita, o nome do agente tem que ser agente_gerador
agente_gerador = RunnableWithMessageHistory(
    cadeia_receita,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="historico",
)

# =========================
# EXECUÇÃO DA CONVERSA
# =========================
def formatar_contexto_rag(docs):
    if not docs:
        return "Não há receitas cadastradas no momento para este corte."
    d = docs[0]  
    return (
        f"# {d.get('Título','')}\n"
        f"## Introdução\n{d.get('Introdução','')}\n"
        f"## Ingredientes\n{d.get('Ingredientes','')}\n"
        f"## Modo de preparo\n{d.get('Modo de preparo','')}\n"
        f"## Marca\n{d.get('Marca','')}\n"
        f"## Corte\n{d.get('Corte','')}"
    )

def iniciar_chat():
    print("Digite 'sair' para encerrar.\n")
    corte_detectado = None
    session_id = "usuario_demo"  

    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("IA: Até a próxima!")
            break

        cfg = {"configurable": {"session_id": session_id}}

        resposta_verificador = agente_verificador.invoke({"input": user_input}, config=cfg).strip().lower()

        if resposta_verificador in CORTES_VALIDOS_NORMALIZADOS:
            corte_detectado = CORTES_VALIDOS_NORMALIZADOS[resposta_verificador]

            docs = buscar_no_mongo(corte_detectado)
            contexto_rag = formatar_contexto_rag(docs)

            resposta = agente_gerador.invoke(
                {
                    "input": f"O usuário pediu uma receita com {corte_detectado}.",
                    "corte_atual": corte_detectado,
                    "contexto_rag": contexto_rag,
                },
                config=cfg,
            )
        else:
            if corte_detectado is None:
                resposta = resposta_verificador
            else:
                docs = buscar_no_mongo(corte_detectado)
                contexto_rag = formatar_contexto_rag(docs)

                resposta = agente_gerador.invoke(
                    {
                        "input": user_input,
                        "corte_atual": corte_detectado,
                        "contexto_rag": contexto_rag,
                    },
                    config=cfg,
                )

        print("IA:", resposta)

iniciar_chat()