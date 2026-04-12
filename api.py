from fastapi import FastAPI
from pydantic import BaseModel
from router import handle_command

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Cozy TraderAgent API running"}

@app.post("/chat")
def chat(req: ChatRequest):
    result = handle_command(req.message)

    return {
        "input": req.message,
        "response": result
    }
