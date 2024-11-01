from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
from app.services.openai_service import OpenAIService
import logging
from time import time

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/parse")
async def parse_medical_text(request: Dict[str, Any]) -> JSONResponse:
    """
    Parse medical text into structured JSON format.
    """
    try:
        # Validate input
        if 'text' not in request or not request['text'].strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        start_time = time()

        # Initialize OpenAI service
        service = OpenAIService()

        # Process the text
        logger.info("Processing medical text")
        result = await service.process_full_text(request['text'])

        # Add metadata
        processing_time = time() - start_time
        result['metadata'] = {
            'processing_time': round(processing_time, 2),
            'timestamp': str(time()),
            'text_length': len(request['text'])
        }

        if request.get('include_cost_estimate', False):
            result['metadata']['estimated_cost'] = service.estimate_cost(request['text'])

        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing medical text"
        )


@router.get("/test")
async def test_endpoint() -> Dict[str, str]:
    """
    Simple test endpoint to verify routing is working.
    """
    return {"status": "Router is functioning"}
