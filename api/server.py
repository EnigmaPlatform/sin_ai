# api/server.py

from fastapi import FastAPI, HTTPException
from core.network import SinNetwork
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any

app = FastAPI()
sin = SinNetwork()

class ChatRequest(BaseModel):
    message: str
    context: List[str] = []

@app.post("/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    try:
        response = sin.communicate(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn/text")
async def learn_text(text: str) -> Dict[str, str]:
    try:
        sin.learn_from_text(text)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
