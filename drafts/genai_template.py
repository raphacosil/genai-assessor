import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
 
load_dotenv()
genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
 
llm = genai.GenerativeModel(
    model_name = "gemini-2.5-flash",
    system_instruction = "",
    generation_config=genai.types.GenerationConfig(
        temperature=0.7,
        top_p=0.95
    )
)
try:
    response = llm.generate_content(input("Type your question: "))
    print(response.text)
except Exception as e :
    print("API ERROR: ", e)