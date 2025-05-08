from fastapi import FastAPI, HTTPException
from sin_ai.core.network import SinNetwork
from pydantic import BaseModel
import uvicorn

app = FastAPI()
sin = SinNetwork()

class ChatRequest(BaseModel):
    message: str
    context: list[str] = []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = sin.communicate(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn/text")
async def learn_text(text: str):
    try:
        sin.learn_from_text(text)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
