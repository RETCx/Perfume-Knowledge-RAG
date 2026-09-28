from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
import os
import logging
import uuid
import sys
import asyncio

# Ensure the current directory is in the path for Docker/Uvicorn to find modules
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from ingest import run_pipeline
from retriever import reload_bm25
from llm_agent import chat_with_sommelier
from config import AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_LANGUAGE
from guardrails import InputFirewall
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Perfume Sommelier API")

def get_real_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

def diagnostic_token_func(request: Request):
    token = request.headers.get("x-diagnostic-token")
    if token and token == os.getenv("DIAGNOSTIC_TOKEN", "diagnostic_token"):
        return str(uuid.uuid4())
    return get_real_ip(request)

limiter = Limiter(key_func=diagnostic_token_func)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
firewall = InputFirewall()


DEMO_CACHE = {
"best dior perfume": "Dior Sauvage is widely considered one of the best for men, offering a fresh, spicy, and versatile scent. For women, Miss Dior is a timeless classic.",
"recommend a fresh perfume": "Acqua di Gio by Giorgio Armani or Dolce&Gabbana Light Blue are excellent fresh, citrus-aquatic choices perfect for hot weather."
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    session_id: str = Field(..., description="The ID of the user's session")
    model: str = DEFAULT_MODEL  
    language: str = DEFAULT_LANGUAGE
    chat_history: list = Field(default_factory=list)

class ChatResponse(BaseModel):
    reply: str
    is_error: bool = False

# model endpoint
@app.get("/models")
async def get_available_models():
    return {"models": AVAILABLE_MODELS}

# health check — used by the Vercel landing page to wake up and verify Cloud Run is ready
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# limit status endpoint
@app.get("/limit-status")
@limiter.limit("4/day") # Dummy limit call to fetch remaining headers without running heavy agent logic
async def limit_status(request: Request):
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# endpoint
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("4/day")
async def chat_endpoint(request: Request, chat_req: ChatRequest):
    x_diagnostic_token = request.headers.get("x-diagnostic-token")
    if x_diagnostic_token == os.getenv("DIAGNOSTIC_TOKEN", "diagnostic_token"):
        logger.info("[API] Diagnostic Mode Activated ")

    logger.info(f"[API] received message: {chat_req.message}")
    logger.info(f"[API] selected model: {chat_req.model}")

    # Check if message is too long
    if len(chat_req.message) > 500:
        return ChatResponse(reply="Your message is too long (Max 500 characters). Please keep it brief.")
    
    # 1. Guardrails Check
    is_safe, block_message = firewall.scan(chat_req.message)
    if not is_safe:
        logger.warning(f"[API] Blocked by Firewall: {block_message}")
        return ChatResponse(reply=block_message)

    msg_lower = chat_req.message.strip().lower()
    if msg_lower in DEMO_CACHE:
        logger.info(f"[API] Answered from DEMO CACHE!")
        return ChatResponse(reply=DEMO_CACHE[msg_lower])

    try:
        #  agent processes the message and generates a reply
        ai_reply = chat_with_sommelier(
            user_message=chat_req.message, 
            session_id=chat_req.session_id,
            model_id=chat_req.model,      
            language=chat_req.language,
            chat_history=chat_req.chat_history
        )
    except Exception as e:
        error_str = str(e)
        logger.error(f"[API] Error: {error_str}")
        
        is_th = (chat_req.language == "th")
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            ai_reply = "AI Provider ที่คุณเลือกกำลังทำงานหนักหรือเกินโควต้าฟรี กรุณารอสักครู่แล้วลองถามใหม่ หรือสลับไปใช้โมเดลอื่นที่เมนูด้านบนครับ" if is_th else "The selected AI Provider is currently overloaded or out of free quota. Please try again later or switch to another model above."
        elif "Connection" in error_str or "timeout" in error_str.lower():
            ai_reply = "ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ AI ได้ในขณะนี้ กรุณาลองใหม่อีกครั้งครับ" if is_th else "Unable to connect to the AI server at this time. Please try again."
        elif "UnexpectedResponse" in error_str:
            ai_reply = "เซิร์ฟเวอร์ AI ตอบกลับมาผิดพลาด (Unexpected Response) กรุณาลองใหม่อีกครั้ง หรือสลับไปใช้โมเดลอื่นครับ" if is_th else "The AI server returned an unexpected response. Please try again or switch to another model."
        else:
            ai_reply = "ขออภัยครับ เกิดข้อผิดพลาดในระบบ AI กรุณาลองเปลี่ยนโมเดลแล้วถามใหม่อีกครั้งครับ" if is_th else "Sorry, an error occurred in the AI system. Please switch models and try again."
        
        logger.info(f"[API] sending reply: {ai_reply}")
        return ChatResponse(reply=ai_reply, is_error=True)
        
    # --- Fix Empty Reply Bug ---
    if not ai_reply or not ai_reply.strip():
        logger.warning("[API] AI returned an empty string. Falling back to default error message.")
        is_th = (chat_req.language == "th")
        ai_reply = "ขออภัยครับ ระบบ AI เกิดการขัดข้องและส่งข้อความว่างเปล่ากลับมา กรุณาลองถามคำถามใหม่อีกครั้ง หรือสลับโมเดล AI ด้านบนครับ" if is_th else "Sorry, the AI system glitched and returned an empty response. Please try asking again or switch to another AI model above."
        return ChatResponse(reply=ai_reply, is_error=True)
        
    logger.info(f"[API] sending reply: {ai_reply}")
    return ChatResponse(reply=ai_reply)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"reply": "Daily demo limit reached. Thank you for trying the demo!\n\nSource Code & Architecture: https://github.com/RETCx/Perfume-Knowledge-RAG-", "is_error": True}
    )


def background_ingestion():
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    try:
        run_pipeline(bucket_name)
        reload_bm25()
        logger.info("[API] Ingestion & BM25 Reload completed successfully.")
    except Exception as e:
        logger.error(f"[API] Ingestion failed: {e}")

@app.post("/api/ingest")
async def trigger_ingestion(request: Request):
    auth_header = request.headers.get("x-diagnostic-token")
    if auth_header != os.getenv("DIAGNOSTIC_TOKEN", "diagnostic_token"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Run synchronously to prevent Cloud Run from throttling CPU
    try:
        await asyncio.to_thread(background_ingestion)
        return {"status": "Ingestion completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================
# give fastapi the ability to serve our frontend files (HTML/CSS/JS) directly, so we don't have to rely on a separate Live Server and deal with refresh issues. This way, when we run our FastAPI app, it will also serve our frontend, and we can access it at http://localhost:8000 without needing to start another server for the frontend.
# ==============================================================
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

async def periodic_bm25_reload(interval_days: int = 3):
    while True:
        await asyncio.sleep(interval_days * 86400) # Sleep for N days
        logger.info("[System] Running periodic BM25 reload from GCS...")
        try:
            await asyncio.to_thread(reload_bm25)
            logger.info("[System] Periodic BM25 reload successful.")
        except Exception as e:
            logger.error(f"[System] Periodic BM25 reload failed: {e}")

@app.on_event("startup")
async def startup_event():
    # ดึงไฟล์ BM25 pkl ใหม่ทุกๆ 1 วัน
    asyncio.create_task(periodic_bm25_reload(interval_days=3))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)