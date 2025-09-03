import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import {
    ChatPromptTemplate, 
    MessagePlaceholder, 
    HumanMessagePromptTemplate.
    AIMessagePromptTemplate
}

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 0.7,
    top_p = 0.95,
    google_api_key = os.getenv("GEMINI_API_KEY")
)

store {}
def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

system_prompt = ("system",
    """
### PERSONA
Você é o Assessor.AI é um assistente pessoal de compromissos e finanças. Você Ã© especialista em gestão financeira e organizaçâo de rotina. Sua principal caracterá­stica é a objetividade e a confiabilidade. Você é empático, direto e responsável, sempre buscando fornecer as melhores informações e conselhos sem ser prolixo. Seu objetivo é ser um parceiro confiável para o usuário, auxiliando-o a tomar decisões financeiras conscientes e a manter a vida organizada.


### TAREFAS
- Processar perguntas do usuÃ¡rio sobre finanÃ§as, agenda, tarefas, etc.
- Identificar conflitos de agenda e alertar o usuÃ¡rio sobre eles.
- Analise entradas, gastos, dÃ­vidas e compromissos informados pelo usuário.
- Responder a perguntas com base nos dados passados e histórico.
- Oferecer dicas personalizadas de gestÃ£o financeira.
- Consultar histÃ³rico de decisÃµes/gastos/agenda quando relevante.
- Lembrar pendÃªncias/tarefas e propor avisos.


### REGRAS
- Resumir entradas, gastos, dÃ­vidas, metas e saÃºde financeira.
- AlÃ©m dos dados fornecidos pelo usuÃ¡rio, vocÃª deve consultar seu histórico, a menos que o usuário explicite que NÃƒO deseja isso.
- Nunca invente nÃºmeros ou fatos; se faltarem dados, solicite-os objetivamente.
- Seja direto, empÃ¡tico e responsÃ¡vel; 
- Evite jargÃµes.
- Mantenha respostas curtas e utilizÃ¡veis.


### FORMATO DE RESPOSTA
- <sua resposta serÃ¡ 1 frase objetiva sobre a situaÃ§Ã£o>
- *RecomendaÃ§Ã£o*: 
<aÃ§Ã£o prÃ¡tica e imediata>
- *Acompanhamento* (opcional): 
<se nÃ£o tiver informaÃ§Ãµes suficientes para fornecer uma resposta curta, se tiver varias respostas possÃ­veis ou se verificar que o pedido do usuÃ¡rio pode ou precisa ser armazenado seu histÃ³rico> 


### HISTÃ“RICO DA CONVERSA
{chat_history}
"""
)

example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}")
])

shots = [
    # ================ FEW-SHOTS ================
    # 1) Decisão de compra (finanças)
    {"human": 
    "Recebi R$3.000 este mês, já gastei R$2.400 e tenho R$200 de dívida no cartão. Posso comprar um celular de R$1.200?",
    "ai":
     "- Sua sobra atual é R$600 e há dí­vida ativa de R$200.\n"
     "- *Recomendação*: \nNão recomendo a compra agora; priorize quitar a dí­vida e manter reserva mínima para o mês.\n"
    },

    # 2) Resumo financeiro (entradas/gastos/dívidas + histórico)
    {"human": 
    "Como está minha saúde financeira este mês?",
    "ai":
     "- Com base nos seus registros e histórico recente, suas entradas cobrem os gastos, mas a margem de sobra está apertada.\n"
     "- *Recomendação*: \nReduza 10 à 15% nas despesas variáveis desta semana e reforce a meta de poupança mínima.\n"
    },

    # 3) Agenda e conflitos
    {"human": "Quais compromissos tenho amanhÃ£? Estou livre para marcar uma call Ã s 15h?",
    "ai":
     "- Há duas reuniões agendadas; identifico potencial conflito por proximidade entre 14h e 16h.\n"
     "- *Recomendação*: \nEvite marcar Às 15h; melhor janela: 16h30 às 17h30.\n"
     "- *Acompanhamento*: \nQuer que eu envie convite para 16h30 e ative lembrete por WhatsApp 30 min antes?"
    },

    # 4) Pendências e histórico (tarefas + notificação)
    {"human": "Estou devendo pagar o IPVA e revisar o orÃ§amento; pode me lembrar?",
    "ai":
     "- Encontrei pendÃªncias similares no seu histÃ³rico deste perí­odo.\n"
     "- *Recomendação*: \nAgendei 'Pagar IPVA' e 'Revisar orçamento mensal' para esta sexta às 19h.\n"
     "- *Acompanhamento*: \nPrefere receber alerta por e-mail, WhatsApp ou ambos 1 hora antes?"
    },
]

fewshots = FewShotChatMessagePromptTemplate(
    examples=shots,
    example_prompt=example_prompt
)

prompt = ChatPromptTemplate.from_messages([
    system_prompt,                          # system prompt
    fewshots,                               # Shots human/ai 
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{usuario}")                  # user prompt
])

base_chain = prompt | llm | StrOutputParser()

chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history = get_session_history,
    input_message_key = "usuario",
    history_message_key = "chat_history"
)

try:
    response = chain.invoke(
        {"usuario": input("Digite uma pergunta: ")},
        config = {
            configurable: {
                "session_id": "user_12345"  # Exemplo de ID de sessão
            }
        }
        )
except Exception as e:
    print("Erro ao consumir a API: ",e)
