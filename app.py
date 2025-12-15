from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from RAG.rag import RAG_chatbot
import traceback  # <-- Add this import

bot = RAG_chatbot()
app = FastAPI()

templates = Jinja2Templates(directory="templates")

class UserQuery(BaseModel):
    user_query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(payload: UserQuery = Body(...)):
    try:
        response = await bot.get_response(payload.user_query)
        return {"answer": response}
    except Exception as e:
        # --- START TEMPORARY DEBUG CODE ---
        print("--- CRITICAL ERROR TRACEBACK ---")
        traceback.print_exc() 
        print("--------------------------------")
        # --- END TEMPORARY DEBUG CODE ---
        raise HTTPException(status_code=500, detail=str(e))