from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging
import os
import time
import json
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .core.openai_client import OpenAIClient, OpenAIError
from .prompts.config import get_prompt_config
from .utils.json_handler import JSONHandler, JSONProcessingError


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
try:
    load_dotenv()
    openai_client = OpenAIClient("os.getenv('OPENAI_API_KEY')")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise


class RequestModel:
    def __init__(self, body: Any, max_tokens: Optional[int] = 4000, temperature: Optional[float] = 0.7):
        self.body = body
        self.max_tokens = max_tokens
        self.temperature = temperature


@app.post("/api/convert-policy-to-json")
async def convert_policy_to_json(request: Dict[str, Any]):
    """Convert policy text into structured JSON format."""
    try:
        req = RequestModel(**request)
        config = get_prompt_config("policy_json_conversion")
        messages = [
            {"role": "system", "content": config.system_content},
            {"role": "user", "content": req.body}
        ]

        # Using your existing OpenAI client with chunking support
        response = await openai_client.get_complete_response(
            messages,
            config,
            req.max_tokens,
            req.temperature
        )

        return response

    except Exception as e:
        logger.error(f"Error in convert_policy_to_json: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-encoding")
async def generate_encoding(request: Dict[str, Any]):
    try:
        req = RequestModel(**request)
        config = get_prompt_config("encoding")
        messages = [
            {"role": "system", "content": config.system_content},
            {"role": "user", "content": req.body}
        ]

        return await openai_client.get_complete_response(
            messages,
            config,
            req.max_tokens,
            req.temperature
        )
    except Exception as e:
        logger.error(f"Error in generate_encoding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-claim-review")
async def generate_claim_review(request: Dict[str, Any]):
    try:
        req = RequestModel(**request)
        config = get_prompt_config("claim_review")
        messages = [{"role": "system", "content": config.system_content}]

        if isinstance(req.body, list):
            for msg in req.body:
                messages.append({"role": "user", "content": msg})
        else:
            messages.append({"role": "user", "content": req.body})

        return await openai_client.get_complete_response(
            messages,
            config,
            req.max_tokens,
            req.temperature
        )
    except Exception as e:
        logger.error(f"Error in generate_claim_review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-criteria-matching-report")
async def generate_criteria_matching_report(request: Dict[str, Any]):
    try:
        req = RequestModel(**request)
        config = get_prompt_config("criteria_matching")
        messages = [{"role": "system", "content": config.system_content}]

        if isinstance(req.body, list):
            for msg in req.body:
                messages.append({"role": "user", "content": msg})
        else:
            messages.append({"role": "user", "content": req.body})

        return await openai_client.get_complete_response(
            messages,
            config,
            req.max_tokens,
            req.temperature
        )
    except Exception as e:
        logger.error(f"Error in generate_criteria_matching_report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-recommendation-and-mapping")
async def generate_recommendation_and_mapping(request: Dict[str, Any]):
    try:
        req = RequestModel(**request)
        config = get_prompt_config("recommendation_mapping")

        # Combine both prompts with the user's input
        combined_content = f"{config.system_content}\n\n{req.body}"

        messages = [
            {"role": "system", "content": combined_content}
        ]

        return await openai_client.get_complete_response(
            messages,
            config,
            req.max_tokens,
            req.temperature
        )
    except Exception as e:
        logger.error(f"Error in generate_recommendation_and_mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-alternative-care-pathway")
async def generate_alternative_care_pathway(request: Dict[str, Any]):
    try:
        req = RequestModel(**request)
        config = get_prompt_config("alternative_care")
        messages = [
            {"role": "system", "content": config.system_content},
            {"role": "user", "content": req.body}
        ]

        return await openai_client.get_complete_response(
            messages,
            config,
            req.max_tokens,
            req.temperature
        )
    except Exception as e:
        logger.error(f"Error in generate_alternative_care_pathway: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-policy-updates")
async def generate_policy_updates(request: Dict[str, Any]):
    try:
        req = RequestModel(**request)
        config = get_prompt_config("policy_updates")
        messages = [
            {"role": "system", "content": config.system_content},
            {"role": "user", "content": req.body}
        ]

        return await openai_client.get_complete_response(
            messages,
            config,
            req.max_tokens,
            req.temperature
        )
    except Exception as e:
        logger.error(f"Error in generate_policy_updates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Session handling endpoints
sessions = {}


@app.post("/api/start-session")
async def start_session(request: Dict[str, Any]):
    try:
        session_id = str(os.urandom(16).hex())
        sessions[session_id] = {
            "context": [
                {
                    "role": "system",
                    "content": "You are an assistant specializing in prior authorization for basys.ai. "
                               "Your responses should be based exclusively on the provided policy encoding and patient data."
                }
            ],
            "last_access": time.time()
        }

        if "body" in request:
            sessions[session_id]["context"].append({
                "role": "user",
                "content": request["body"]
            })

        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Error in start_session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/send-message")
async def send_message(request: Dict[str, Any]):
    try:
        if "session_id" not in request or "message" not in request:
            raise HTTPException(status_code=400, detail="Missing session_id or message")

        session_id = request["session_id"]
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Update session context with new message
        sessions[session_id]["context"].append({
            "role": "user",
            "content": request["message"]
        })

        # Generate response
        messages = sessions[session_id]["context"]
        config = get_prompt_config("encoding")  # Using default config for sessions

        response = await openai_client.get_complete_response(
            messages,
            config,
            4000,  # Default max tokens
            0.7  # Default temperature
        )

        response_content = response if isinstance(response, str) else json.dumps(response)

        # Update session context with assistant's response
        sessions[session_id]["context"].append({
            "role": "assistant",
            "content": response_content
        })

        sessions[session_id]["last_access"] = time.time()

        return {"response": response_content}

    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_client": "initialized",
        "sessions_active": len(sessions)
    }


# Cleanup old sessions periodically
@app.on_event("startup")
@app.on_event("shutdown")
async def cleanup_sessions():
    """Remove sessions that are older than 1 hour."""
    current_time = time.time()
    session_timeout = 3600  # 1 hour in seconds

    for session_id in list(sessions.keys()):
        if current_time - sessions[session_id]["last_access"] > session_timeout:
            del sessions[session_id]