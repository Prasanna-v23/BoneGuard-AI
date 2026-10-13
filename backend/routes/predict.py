"""
BoneGuard AI - Prediction & Grad-CAM Routes
===========================================
Handles image upload validation, single-image inference,
and Grad-CAM heatmap generation endpoints.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
import logging

from inference.pipeline import get_inference_service

logger = logging.getLogger("boneguard.routes.predict")
router = APIRouter(tags=["Inference & Prediction"])

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB limit


async def validate_uploaded_image(image: UploadFile) -> bytes:
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload an X-ray image."
        )

    # Check content type if provided
    if image.content_type and image.content_type.lower() not in ALLOWED_MIME_TYPES:
        # Also check file extension as fallback
        ext = image.filename.lower().split(".")[-1]
        if ext not in ["jpg", "jpeg", "png", "webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {image.content_type or ext}. Supported formats are JPG, JPEG, PNG, WEBP."
            )

    contents = await image.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty. Please select a valid X-ray image."
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file size exceeds the 15 MB limit."
        )

    return contents


@router.post("/predict")
async def predict_xray(image: UploadFile = File(...)):
    """
    Standard single-image fracture assessment endpoint.
    Accepts multipart/form-data with 'image' file.
    """
    contents = await validate_uploaded_image(image)
    service = get_inference_service()

    try:
        result = service.predict(contents)
        result["filename"] = image.filename
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except ValueError as ve:
        logger.warning(f"Image decode error for {image.filename}: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read this image. Please ensure it is a valid, uncorrupted X-ray image."
        )
    except Exception as e:
        logger.error(f"Prediction failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed due to an internal processing error. Please try again."
        )


@router.post("/predict-with-gradcam")
async def predict_xray_with_gradcam(image: UploadFile = File(...)):
    """
    Inference with Grad-CAM visual attention explainability.
    Returns prediction, probabilities, confidence, and Base64-encoded visualizations.
    """
    contents = await validate_uploaded_image(image)
    service = get_inference_service()

    try:
        result = service.predict_with_gradcam(contents)
        result["filename"] = image.filename
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except ValueError as ve:
        logger.warning(f"Image decode error for {image.filename}: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read this image. Please ensure it is a valid, uncorrupted X-ray image."
        )
    except Exception as e:
        logger.error(f"Grad-CAM inference failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed due to an internal processing error. Please try again."
        )
