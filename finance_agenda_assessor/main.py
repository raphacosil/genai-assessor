import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pg_tools import TOOLS
from datetime import datetime
from zoneinfo import ZoneInfo
from operator import itemgetter
from faq_tools import get_faq_context


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()


store = {}
def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=api_key
)

fast_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0, 
    google_api_key=api_key
)


system_router_prompt = ("system",
    """
### PERSONA SISTEMA
Você é o Assessor.AI — um assistente pessoal de compromissos e finanças. É objetivo, responsável, confiável e empático, com foco em utilidade imediata. Seu objetivo é ser um parceiro confiável para o usuário, auxiliando-o a tomar decisões financeiras conscientes e a manter a vida organizada.
- Evite jargões.
- Evite ser prolixo.
- Não invente dados.
- Respostas sempre curtas e aplicáveis.
- Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.


### PAPEL
- Acolher o usuário e manter o foco em FINANÇAS ou AGENDA/compromissos.
- Decidir a rota: {{financeiro | agenda | faq}} ou se a pergunta é fora do escopo.
- Responder diretamente em:
  (a) saudações/small talk, ou 
  (b) fora de escopo (redirecionando para finanças/agenda).
- Seu objetivo é conversar de forma amigável com o usuário e tentar identificar se ele menciona algo sobre finanças ou agenda.
- Em fora_escopo: ofereça 1–2 sugestões práticas para voltar ao seu escopo (ex.: agendar algo, registrar/consultar um gasto).
- Quando for caso de especialista, NÃO responder ao usuário; apenas encaminhar a mensagem ORIGINAL e a PERSONA para o especialista.


### REGRAS
- Seja breve, educado e objetivo.
- Se faltar um dado absolutamente essencial para decidir a rota, faça UMA pergunta mínima (CLARIFY). Caso contrário, deixe CLARIFY vazio.
- Responda de forma textual.
- Se a men
- Se a mensagem do usuario for uma dúvida geral sobre o sistema, funcionalidades, regras ou politicas -> ROUTE-faq 
- Se for uma operação financeira, orçamento, transação -> ROUTE=financeiro 
- Se for sobre compromissos, eventos, lembretes > ROUTE-agenda 
- Se não se encaixar em nenhum desses casos continue conversa até 0 usuårio a conversar sobre finanças ou agenda/compromisso.


### PROTOCOLO DE ENCAMINHAMENTO (texto puro)
ROUTE=<financeiro|agenda>
PERGUNTA_ORIGINAL=<mensagem completa do usuário, sem edições>
PERSONA=<copie o bloco "PERSONA SISTEMA" daqui>
CLARIFY=<pergunta mínima se precisar; senão deixe vazio>


### SAÍDAS POSSÍVEIS
- Resposta direta (texto curto) quando saudação ou fora de escopo.
- Encaminhamento ao especialista usando exatamente o protocolo acima.


### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

example_prompt_base = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}"),
])

shots_router = [
    {
        "human": "Oi, tudo bem?",
        "ai": "Olá! Posso te ajudar com finanças ou agenda; por onde quer começar?"
    },
    {
        "human": "Me conta uma piada.",
        "ai": "Consigo ajudar apenas com finanças ou agenda. Prefere olhar seus gastos ou marcar um compromisso?"
    },
    {
        "human": "Quanto gastei com mercado no mês passado?",
        "ai": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quanto gastei com mercado no mês passado?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    {
        "human": "Agendar pagamento amanhã às 9h",
        "ai": "Você quer lançar uma transação (finanças) ou criar um compromisso no calendário (agenda)?"
    },
    {
        "human": "Tenho reunião amanhã às 9h?",
        "ai": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Tenho reunião amanhã às 9h?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    {
        "human": "Qual e-mail de suporte?",
        "ai": "ROUTE=faq\nPERGUNTA_ORIGINAL=Qual e-mail de suporte?\nPERSONA={{PERSONA_SISTEMA}}\nCLARIFY="
    },
]

fewshots_router = FewShotChatMessagePromptTemplate(
    examples=shots_router,
    example_prompt=example_prompt_base
)


system_prompt_finance = ("system",
    """
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL sobre finanças e operar as tools de `transactions` para responder. 
    A saída SEMPRE é JSON (contrato abaixo) para o Orquestrador.

    ### TAREFAS

    ### CONTEXTO
    - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
    - Entrada vem do Roteador via protocolo:
    - ROUTE=financeiro
    - PERGUNTA_ORIGINAL=...
    - PERSONA=...   (use como diretriz de concisão/objetividade)
    - CLARIFY=...   (se preenchido, priorize responder esta dúvida antes de prosseguir)


    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.

    ### SAÍDA (JSON)
    Campos mínimos para enviar para o orquestrador:
    # Obrigatórios:
     - dominio   : "financeiro"
     - intencao  : "consultar" | "inserir" | "atualizar" | "deletar" | "resumo"
     - resposta  : uma frase objetiva
     - recomendacao : ação prática (pode ser string vazia se não houver)
    # Opcionais (incluir só se necessário):
     - acompanhamento : texto curto de follow-up/próximo passo
     - esclarecer     : pergunta mínima de clarificação (usar OU 'acompanhamento')
     - escrita        : {{"operacao":"adicionar|atualizar|deletar","id":123}}
     - janela_tempo   : {{"de":"YYYY-MM-DD","ate":"YYYY-MM-DD","rotulo":'mês passado'}}
     - indicadores    : {{chaves livres e numéricas úteis ao log}}

    ### HISTÓRICO DA CONVERSA
    {chat_history}
    """
)

shots_finance = [
    {
        "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quanto gastei com mercado no mês passado?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"financeiro","intencao":"consultar","resposta":"Você gastou R$ 842,75 com 'comida' no mês passado.","recomendacao":"Quer detalhar por estabelecimento?","janela_tempo":{{"de":"2025-08-01","ate":"2025-08-31","rotulo":"mês passado (ago/2025)"}}}}"""
    },
    {
        "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Registrar almoço hoje R$ 45 no débito\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"financeiro","intencao":"inserir","resposta":"Lancei R$ 45,00 em 'comida' hoje (débito).","recomendacao":"Deseja adicionar uma observação?","escrita":{{"operacao":"adicionar","id":2045}}}}"""
    },
    {
        "human": "ROUTE=financeiro\nPERGUNTA_ORIGINAL=Quero um resumo dos gastos\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"financeiro","intencao":"resumo","resposta":"Preciso do período para seguir.","recomendacao":"","esclarecer":"Qual período considerar (ex.: hoje, esta semana, mês passado)?"}}"""
    },
]

fewshots_finance = FewShotChatMessagePromptTemplate(
    examples=shots_finance,
    example_prompt=example_prompt_base,
)

system_prompt_agenda = ("system",
    """
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL sobre agenda/compromissos e (quando houver tools) consultar/criar/atualizar/cancelar eventos. 
    A saída SEMPRE é JSON (contrato abaixo) para o Orquestrador.


    ### TAREFAS



    ### CONTEXTO
    - Hoje é {today_local} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
    - Entrada do Roteador:
    - ROUTE=agenda
    - PERGUNTA_ORIGINAL=...
    - PERSONA=...   (use como diretriz de concisão/objetividade)
    - CLARIFY=...   (se preenchido, responda primeiro)


    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.


    ### SAÍDA (JSON)
    # Obrigatórios:
     - dominio   : "agenda"
     - intencao  : "consultar" | "criar" | "atualizar" | "cancelar" | "listar" | "disponibilidade" | "conflitos"
     - resposta  : uma frase objetiva
     - recomendacao : ação prática (pode ser string vazia)
    # Opcionais (incluir só se necessário):
     - acompanhamento : texto curto de follow-up/próximo passo
     - esclarecer     : pergunta mínima de clarificação
     - janela_tempo   : {{"de":"YYYY-MM-DDTHH:MM","ate":"YYYY-MM-DDTHH:MM","rotulo":"ex.: 'amanhã 09:00–10:00'"}}
     - evento         : {{"titulo":"...","data":"YYYY-MM-DD","inicio":"HH:MM","fim":"HH:MM","local":"...","participantes":["..."]}}


     ### HISTÓRICO DA CONVERSA
    {chat_history}
    """
)

shots_agenda = [
    {
        "human": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Tenho janela amanhã à tarde?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"agenda","intencao":"disponibilidade","resposta":"Você está livre amanhã das 14:00 às 16:00.","recomendacao":"Quer reservar 15:00–16:00?","janela_tempo":{{"de":"2025-09-29T14:00","ate":"2025-09-29T16:00","rotulo":"amanhã 14:00–16:00"}}}}"""
    },
    {
        "human": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Marcar reunião com João amanhã às 9h por 1 hora\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"agenda","intencao":"criar","resposta":"Posso criar 'Reunião com João' amanhã 09:00–10:00.","recomendacao":"Confirmo o envio do convite?","janela_tempo":{{"de":"2025-09-29T09:00","ate":"2025-09-29T10:00","rotulo":"amanhã 09:00–10:00"}},"evento":{{"titulo":"Reunião com João","data":"2025-09-29","inicio":"09:00","fim":"10:00","local":"online"}}}}"""
    },
    {
        "human": "ROUTE=agenda\nPERGUNTA_ORIGINAL=Agendar revisão do orçamento na sexta\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"agenda","intencao":"criar","resposta":"Preciso do horário para agendar.","recomendacao":"","esclarecer":"Qual horário você prefere na sexta?"}}"""
    },
]

fewshots_agenda = FewShotChatMessagePromptTemplate(
    examples=shots_agenda,
    example_prompt=example_prompt_base,
)


system_prompt_faq = ("system",
"""
    ### PAPEL
    Você deve responder perguntas sobre dúvidas SOMENTE com base no documento normativo oficial (trechos fornecidos em CONTEXTO).
    Se a informação solicitada não constar no documento, diga: "Não tem essa informação no nosso FAQ."
    
    ### REGRAS
    - Seja breve, claro e educado.
    - Fale em linguagem simples, sem jargões técnicos ou referências a código/infra.
    - Quando fizer sentido, mencione a parte relevante (Ex.: "Seção 6.2.1") se isso estiver explícito no trecho.
    - Não prometa funcionalidades futuras. Se o documento falar em roadmap, informe de modo conservador.
    - Em tópicos sensíveis, reforce a informação normativa (ex.: LGPD, impossibilidade de exclusão de lançamentos, não substituição de profissionais, suporte).
    
    ### ENTRADA
    - ROUTE=faq
    - PERGUNTA_ORIGINAL=...
    - PERSONA=... (use como diretriz de concisão/objetividade)
    - CLARIFY=... (se preenchido, responda primeiro)
""")
 
prompt_faq = ChatPromptTemplate.from_messages([
    system_prompt_faq,
    "human",
    "Pergunta do usuário:\n{question}\n\nCONTEXTO (trechos do documento):\n{context}\n\nResponda com base APENAS no CONTEXTO."
])

system_prompt_orquestrador = ("system",
    """
### PAPEL
Você é o Agente Orquestrador do Assessor.AI. Sua função é entregar a resposta final ao usuário **somente** quando um Especialista retornar o JSON.


### ENTRADA
- ESPECIALISTA_JSON contendo chaves como:
  dominio, intencao, resposta, recomendacao (opcional), acompanhamento (opcional),
  esclarecer (opcional), janela_tempo (opcional), evento (opcional), escrita (opcional), indicadores (opcional).


### REGRAS
- Use **exatamente** `resposta` do especialista como a **primeira linha** do output.
- Se `recomendacao` existir e não for vazia, inclua a seção *Recomendação*; caso contrário, **omita**.
- Para *Acompanhamento*: se houver `esclarecer`, use-o; senão, se houver `acompanhamento`, use-o; caso contrário, **omita** a seção.
- Não reescreva números/datas se já vierem prontos. Não invente dados. Seja conciso.
- Não retorne JSON; **sempre** retorne no FORMATO DE SAÍDA.


### FORMATO DE SAÍDA (sempre ao usuário)
<sua resposta será 1 frase objetiva sobre a situação>
- *Recomendação*:
<ação prática e imediata>     # omita esta seção se não houver recomendação
- *Acompanhamento* (opcional):
<pergunta/minipróximo passo>  # omita se nada for necessário


### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

shots_orquestrador = [
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"financeiro","intencao":"consultar","resposta":"Você gastou R$ 842,75 com 'comida' no mês passado.","recomendacao":"Quer detalhar por estabelecimento?","janela_tempo":{{"de":"2025-08-01","ate":"2025-08-31","rotulo":"mês passado (ago/2025)"}}}}""",
        "ai": "Você gastou R$ 842,75 com 'comida' no mês passado.\n- *Recomendação*:\nQuer detalhar por estabelecimento?"
    },

    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"financeiro","intencao":"resumo","resposta":"Preciso do período para seguir.","recomendacao":"","esclarecer":"Qual período considerar (ex.: hoje, esta semana, mês passado)?"}}""",
        "ai": """Preciso do período para seguir.\n- *Acompanhamento* (opcional):\nQual período considerar (ex.: hoje, esta semana, mês passado)?"""
    },

    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"agenda","intencao":"criar","resposta":"Posso criar 'Reunião com João' amanhã 09:00–10:00.","recomendacao":"Confirmo o envio do convite?","janela_tempo":{{"de":"2025-09-29T09:00","ate":"2025-09-29T10:00","rotulo":"amanhã 09:00–10:00"}},"evento":{{"titulo":"Reunião com João","data":"2025-09-29","inicio":"09:00","fim":"10:00","local":"online"}}}}""",
        "ai": """Posso criar 'Reunião com João' amanhã 09:00–10:00.\n- *Recomendação*:\nConfirmo o envio do convite?"""
    },
]

fewshots_orquestrador = FewShotChatMessagePromptTemplate(
    examples=shots_orquestrador,
    example_prompt=example_prompt_base,
)


prompt_router = ChatPromptTemplate.from_messages([
    system_router_prompt,
    fewshots_router,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(today_local=today.isoformat())

prompt_orchestrator = ChatPromptTemplate.from_messages([
    system_prompt_orquestrador,
    fewshots_orquestrador,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
]).partial(today_local=today.isoformat())

prompt_schedule_agent = ChatPromptTemplate.from_messages([
    system_prompt_agenda,
    fewshots_agenda,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
]).partial(today_local=today.isoformat())

prompt_finance_agent = ChatPromptTemplate.from_messages([
    system_prompt_finance,
    fewshots_finance,
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"), 
]).partial(today_local=today.isoformat())


finance_agent = create_tool_calling_agent(llm, TOOLS, prompt_finance_agent)
finance_agent_executor = AgentExecutor(agent=finance_agent, tools=TOOLS, verbose=False)
finance_agent = RunnableWithMessageHistory(
    finance_agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    verbose=False,
    handle_parsing_errors=False,
    return_intermediate_steps=False,
)


schedule_agent = create_tool_calling_agent(llm, TOOLS, prompt_schedule_agent)
schedule_agent_executor = AgentExecutor(agent=schedule_agent, tools=TOOLS, verbose=False)
schedule_agent = RunnableWithMessageHistory(
    schedule_agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    verbose=False,
    handle_parsing_errors=False,
    return_intermediate_steps=False,
)

router_chain = RunnableWithMessageHistory(
    prompt_router | fast_llm,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

orchestrator_agent = RunnableWithMessageHistory(
    prompt_orchestrator | fast_llm | StrOutputParser(),
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

faq_chain_core = (
    RunnablePassthrough.assign(
        question = itemgetter("input"),
        context=lambda x: get_faq_context(x["input"])
    )
    | prompt_faq | fast_llm | StrOutputParser()
)

def execute_assessor_flow(user_question: str, session_id: str):
    """
    Função que controla o fluxo do assessor com base no retorno do router (se ele encaminhará para um dos agentes de acordo com a pergunta o usuário).
    """
    response_router = router_chain.invoke(input={"input": user_question},
                                        config={"configurable": {"session_id": session_id}})
    
    if not "ROUTE=" in response_router:
        return response_router
    else:
        
        if "ROUTE=financeiro" in response_router:
            resposta_finance = finance_agent.invoke(input={"input": response_router},
                                        config={"configurable": {"session_id": session_id}})
            output_orchestrator = orchestrator_agent.invoke(input={"input": resposta_finance["output"]},
                                        config={"configurable": {"session_id": session_id}})
            
            print(output_orchestrator)
            return output_orchestrator
        
        elif "ROUTE=agenda" in response_router:
            resposta_schedule = schedule_agent.invoke(input={"input": response_router},
                                        config={"configurable": {"session_id": session_id}})
            
            output_orchestrator = orchestrator_agent.invoke(input={"input": resposta_schedule["output"]},
                                        config={"configurable": {"session_id": session_id}})
            
            print(output_orchestrator)
            return output_orchestrator
        
        elif "ROUTE=faq" in response_router:
            response_faq = faq_chain_core.invoke(input={"input": user_question},
                                        config={"configurable": {"session_id": session_id}})
            
            print(response_faq)
            return response_faq

while True:
    try:
        user_input = input("> | ")
        if user_input.lower() in ("sair", "end", "fim", "tchau", "bye", "tchautchau"):
            print("Encerrando a conversa")
            break
        
        resposta = execute_assessor_flow(
            user_question=user_input,
            session_id="PRECISA_MAS_NÃO_IMPORTA"
        )
        
        print(resposta)
    except Exception as e:
        print("Erro ao consumir a API: ", e)
