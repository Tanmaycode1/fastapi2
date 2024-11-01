from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
from ..services.openai_service import OpenAIService
import logging
import time
import traceback
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/parse")
async def parse_medical_text(request: Request) -> JSONResponse:
    """
    Parse medical text into structured JSON format with detailed error handling.
    """
    start_time = time.time()

    try:
        # Log the incoming request
        logger.debug("Received parse request")

        # Get the request body
        body = await request.json()
        logger.debug(f"Request body received: {body}")

        if 'text' not in body or not body['text'].strip():
            logger.error("Missing or empty text in request")
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Initialize OpenAI service
        logger.debug("Initializing OpenAI service")
        service = OpenAIService()

        # Process the text
        logger.debug("Starting text processing")
        result = await service.process_full_text(body['text'])
        logger.debug("Text processing completed")

        # Add processing metadata
        processing_time = round(time.time() - start_time, 2)
        estimated_cost = service.estimate_cost(body['text'])

        result['metadata'] = {
            'processing_time': processing_time,
            'estimated_cost': estimated_cost,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'text_length': len(body['text'])
        }

        logger.debug(f"Successful processing in {processing_time} seconds")
        return JSONResponse(content=result)

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error("Unexpected error in parse_medical_text:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())

        # Return detailed error information
        error_detail = {
            "detail": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc().split('\n'),
            "processing_time": round(time.time() - start_time, 2)
        }

        return JSONResponse(
            status_code=500,
            content=error_detail
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Enhanced health check endpoint.
    """
    try:
        # Test OpenAI service initialization
        service = OpenAIService()
        api_key_status = "available" if service.get_api_key() else "missing"

        return {
            "status": "healthy",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "openai_api_key": api_key_status,
            "environment": {
                "python_version": sys.version,
                "debug_mode": logger.getEffectiveLevel() == logging.DEBUG
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }