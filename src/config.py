import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')

# --- System Defaults ---
DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_LANGUAGE = "Thai"

# --- (Single Source of Truth) ---
AVAILABLE_MODELS = [
    # (Always Available - Recommend)
    {"id": "gemini-2.5-flash-lite", "provider": "gemini", "name": "Gemini 2.5 Flash-Lite", "description": "ใช้งานได้แน่นอน (เร็วและประหยัดสุด)"},
    {"id": "gemini-2.5-flash", "provider": "gemini", "name": "Gemini 2.5 Flash", "description": "ใช้งานได้แน่นอน (ฉลาดและครอบคลุม)"},
    {"id": "gemini-3.1-flash-lite-preview", "provider": "gemini", "name": "Gemini 3.1 Flash-Lite Preview", "description": "ใช้งานได้แน่นอน (ตัวอย่างรุ่นเบาและใหม่สุด)"},
    
    # Others (Might be slow/unavailable)
    # Groq 
    {"id": "llama-3.3-70b-versatile", "provider": "groq", "name": "Llama 3.3 70B (Groq)", "description": "Fast Llama (อาจจะช้า หรือใช้ไม่ได้ขึ้นอยู่กับเวลา)"},
    {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "provider": "groq", "name": "Llama 4 Scout (Groq)", "description": "New Llama 4 (อาจจะช้า หรือใช้ไม่ได้ขึ้นอยู่กับเวลา)"},
    {"id": "qwen/qwen3.6-27b", "provider": "groq", "name": "Qwen 3.6 27B (Groq)", "description": "Qwen model (อาจจะช้า หรือใช้ไม่ได้ขึ้นอยู่กับเวลา)"},
    
    # NVIDIA NIM 
    {"id": "nvidia/llama-3.3-nemotron-super-49b-v1.5", "provider": "nvidia", "name": "Nemotron 49B (NVIDIA)", "description": "Nvidia's model (อาจจะช้า หรือใช้ไม่ได้ขึ้นอยู่กับเวลา)"},
    {"id": "meta/llama-3.3-70b-instruct", "provider": "nvidia", "name": "Llama 3.3 70B (NVIDIA)", "description": "Llama 3.3 (อาจจะช้า หรือใช้ไม่ได้ขึ้นอยู่กับเวลา)"}
]
