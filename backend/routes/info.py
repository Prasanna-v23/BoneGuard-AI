"""
BoneGuard AI - Metadata & Validation Performance Routes
=======================================================
Exposes technical model specifications, parameter statistics,
and verified BFRMX internal test validation data.
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from inference.pipeline import get_inference_service

router = APIRouter(tags=["Model Specifications & Validation"])


@router.get("/model-info")
async def get_model_specifications():
    """
    Returns technical details regarding the deployed DenseNet121 + DG-CBAM model.
    """
    service = get_inference_service()
    info = service.get_model_info()
    return JSONResponse(status_code=status.HTTP_200_OK, content=info)


@router.get("/validation-info")
async def get_validation_performance():
    """
    Returns research performance statistics for both the internal BFRMX test dataset
    and the independent external FracAtlas benchmark.
    """
    service = get_inference_service()
    val_data = service.get_validation_info()
    return JSONResponse(status_code=status.HTTP_200_OK, content=val_data)
