import os
import logging
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
from config import GROQ_API_KEY, NVIDIA_API_KEY, AVAILABLE_MODELS
from retriever import hybrid_search_perfume
from langchain_community.tools import DuckDuckGoSearchRun
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

# Set up SQLite cache to save API limits for repeated questions
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Setup Logging
logger = logging.getLogger(__name__)

# 1. Pydantic Schema
class PerfumeSearchInput(BaseModel):
    query: str = Field(description="The main search keyword translated to English (e.g., fresh, sweet, elegant).")
    style_preference: str = Field(description="The target gender or style. MUST be exactly one of: 'male', 'female', or 'unisex'.")
    notes_or_vibes: str = Field(description="Specific fragrance notes or vibes requested by the user. Leave empty if not mentioned.")
    excluded_notes: str = Field(description="Fragrance notes the user explicitly dislikes. Leave empty if none.")

class InternetSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the internet. MUST be in English for best search results.")

# 2. Tools 
@tool("search_perfume", args_schema=PerfumeSearchInput)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_perfume(query: str, style_preference: str, notes_or_vibes: str, excluded_notes: str = "") -> str:
    """Use this tool EXCLUSIVELY to search for perfumes in the store's database"""
    logger.info(f"[System] Searching DB for: {query} | Style: {style_preference} | Notes: {notes_or_vibes} | Excluded: {excluded_notes}")
    return hybrid_search_perfume(query, style_preference, notes_or_vibes, excluded_notes)

@tool("websearch", args_schema=InternetSearchInput)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def websearch(query: str) -> str:
    """Use this tool to search the internet (via DuckDuckGo) for real-time information such as perfume prices."""
    logger.info(f"[System] Web Searching for: {query}")
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        logger.error(f"Error in websearch: {e}")
        raise e  

# 3. Create LLM 
def get_llm(model_id: str):
    """LLM Factory with Fallbacks"""
    model_info = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    provider = model_info["provider"] if model_info else "gemini"
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    # Define cross-provider fallbacks
    try:
        gemini_fallback = ChatVertexAI(
            model_name="gemini-2.5-flash", 
            temperature=0.3,
            safety_settings=safety_settings
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Vertex AI fallback: {e}")
        gemini_fallback = None
    
    groq_fallback = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=GROQ_API_KEY
    ) if GROQ_API_KEY else None
    

    
    if provider == "gemini":
        primary_llm = ChatVertexAI(
            model_name=model_id, 
            temperature=0.3,
            safety_settings=safety_settings
        )
        if groq_fallback:
            return primary_llm.with_fallbacks([groq_fallback])
        return primary_llm
        
    elif provider == "groq":
        primary_llm = ChatGroq(
            model=model_id, 
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        if gemini_fallback:
            return primary_llm.with_fallbacks([gemini_fallback])
        return primary_llm
        
    elif provider == "nvidia":
        if model_id in ["nvidia/llama-3.3-nemotron-super-49b-v1.5", "meta/llama-3.3-70b-instruct"]:
            primary_llm = ChatOpenAI(
                model=model_id, 
                temperature=0.3,
                api_key=NVIDIA_API_KEY,
                base_url="https://integrate.api.nvidia.com/v1"
            )
        else:
            primary_llm = ChatNVIDIA(
                model=model_id, 
                temperature=0.3,
                api_key=NVIDIA_API_KEY
            )
            
        if gemini_fallback:
            return primary_llm.with_fallbacks([gemini_fallback])
        return primary_llm

    # Fallback if provider is unrecognized
    logger.warning(f"[LLM] Unknown provider for model '{model_id}', defaulting to gemini-2.5-flash-lite")
    return ChatVertexAI(model_name="gemini-2.5-flash-lite", temperature=0.3, safety_settings=safety_settings)

# 4. Agent
tools = [search_perfume, websearch]
def load_system_prompt(language: str) -> str:
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "system_prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
        return template.format(language=language)
    return "You are a helpful assistant."

def chat_with_sommelier(user_message: str, session_id: str, model_id: str, language: str, chat_history: list =[]):
    llm = get_llm(model_id)
    dynamic_system_prompt = load_system_prompt(language)
    agent_executor = create_react_agent(
        model=llm, 
        tools=tools, 
        prompt=dynamic_system_prompt
    )

    recent_messages = chat_history[-5:] if len(chat_history) > 5 else chat_history
    langchain_history = []
    for msg in recent_messages:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))  
        else:
            langchain_history.append(AIMessage(content=msg["content"])) 

    langchain_history.append(HumanMessage(content=user_message))
    config = {"configurable": {"thread_id": session_id}}
    
    response = agent_executor.invoke({"messages": langchain_history}, config=config)
    
    content = response["messages"][-1].content
    if isinstance(content, list):
        return "".join([part.get("text", "") for part in content if isinstance(part, dict)])
    return str(content)