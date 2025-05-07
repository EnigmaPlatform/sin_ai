from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from core.network.model import SinNetwork
from utils.model_manager import EnhancedModelManager
import logging
import os

class APIServer:
    def __init__(self, model: SinNetwork, manager: EnhancedModelManager):
        self.app = FastAPI(title="SinAI API", version="1.0")
        self.model = model
        self.manager = manager
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger("API")
        
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        @self.app.post("/chat")
        async def chat_endpoint(request: ChatRequest):
            """Основной endpoint для диалога"""
            try:
                response = await self.model.generate_response(
                    request.prompt, 
                    context=request.context
                )
                return {"response": response}
            except Exception as e:
                self.logger.error(f"Chat error: {str(e)}")
                raise HTTPException(500, "Internal server error")

        @self.app.post("/train")
        async def train_endpoint(request: TrainRequest):
            """Обучение модели на предоставленных данных"""
            try:
                if not os.path.exists(request.file_path):
                    raise HTTPException(400, "File not found")
                
                result = await self.model.learn_from_file(request.file_path)
                version = self.manager.save_model(self.model)
                return {
                    "status": "success",
                    "version": version,
                    "metrics": result
                }
            except Exception as e:
                self.logger.error(f"Training error: {str(e)}")
                raise HTTPException(500, str(e))

        @self.app.get("/models")
        async def list_models():
            """Список доступных моделей"""
            return self.manager.list_models()

        @self.app.get("/resources")
        async def get_resources():
            """Мониторинг ресурсов"""
            return self._get_system_resources()

    def _get_system_resources(self):
        import psutil
        import torch
        
        resources = {
            "cpu": {
                "usage": psutil.cpu_percent(),
                "cores": psutil.cpu_count()
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "used": psutil.virtual_memory().used
            }
        }
        
        if torch.cuda.is_available():
            resources["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory": {
                    "total": torch.cuda.get_device_properties(0).total_memory,
                    "allocated": torch.cuda.memory_allocated(0),
                    "cached": torch.cuda.memory_reserved(0)
                }
            }
        
        return resources

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)

class ChatRequest(BaseModel):
    prompt: str
    context: Optional[list] = None

class TrainRequest(BaseModel):
    file_path: str
    epochs: Optional[int] = 1
