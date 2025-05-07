from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from pathlib import Path
from utils.logger import SinLogger
from utils.config import ConfigManager
from core.ai.model import SinModel
from utils.model_manager import ModelManager

app = FastAPI(title="SinAI API", version="1.0")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация компонентов
logger = SinLogger("API")
config = ConfigManager()
model = SinModel()
manager = ModelManager()

class ChatRequest(BaseModel):
    prompt: str
    context: Optional[List[str]] = None

class TrainRequest(BaseModel):
    file_type: str  # "text" или "code"
    description: Optional[str] = None

class ModelInfo(BaseModel):
    version: str
    description: str
    created: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Основной endpoint для диалога"""
    try:
        response = model.generate(request.prompt, request.context)
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(500, "Internal server error")

@app.post("/train")
async def train_endpoint(file: UploadFile, request: TrainRequest):
    """Обучение модели на предоставленных данных"""
    try:
        content = await file.read()
        content = content.decode("utf-8")
        
        result = model.learn(content, request.file_type)
        if result["status"] != "success":
            raise HTTPException(400, result.get("message", "Training failed"))
        
        version = manager.save(model, {"description": request.description})
        return {"version": version, "metrics": result}
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(500, str(e))

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Список доступных моделей"""
    models = manager.list_models()
    return [
        ModelInfo(
            version=version,
            description=meta.get("description", ""),
            created=meta.get("timestamp", "")
        )
        for version, meta in models.items()
    ]

@app.post("/models/{version}/load")
async def load_model(version: str):
    """Загрузка конкретной версии модели"""
    try:
        global model
        model = manager.load(version)
        return {"status": "success", "loaded_version": version}
    except Exception as e:
        raise HTTPException(404, str(e))

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Запуск сервера API"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
