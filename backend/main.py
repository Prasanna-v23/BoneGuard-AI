"""
BoneGuard AI - FastAPI Application Server
=========================================
Project: Advanced Bone Fracture Assessment and Clinical Decision Support
         Through Deep Learning-Based X-Ray Image Analysis
Architecture: React.js UI <-> FastAPI <-> DenseNet121 + DG-CBAM
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.pipeline import get_inference_service
from routes.predict import router as predict_router
from routes.info import router as info_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("boneguard.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize and cache model singleton on startup
    logger.info("Initializing BoneGuard AI Model Pipeline on startup...")
    service = get_inference_service()
    logger.info(f"Model {service.model_info['model_name']} ready on {service.device}")
    yield
    logger.info("BoneGuard AI server shutting down.")


app = FastAPI(
    title="BoneGuard AI Decision Support API",
    description="Deep Learning-Based Bone Fracture Assessment and Grad-CAM Attention System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for React frontend development and local access
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predict_router)
app.include_router(info_router)


@app.get("/health", tags=["System Health"])
async def health_check():
    """
    Health check and model status inspection endpoint.
    """
    try:
        service = get_inference_service()
        model_loaded = service.model is not None
        device_str = str(service.device)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "service": "BoneGuard AI",
                "model_loaded": False,
                "error": str(e)
            }
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "service": "BoneGuard AI",
            "model_loaded": model_loaded,
            "device": device_str,
            "model": "DenseNet121 + DG-CBAM",
            "version": "1.0.0"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
