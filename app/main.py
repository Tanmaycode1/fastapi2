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
from fastapi import File, UploadFile, BackgroundTasks
from typing import Dict, Optional
from datetime import datetime
import tempfile
import os
import uuid
import pypdf
from concurrent.futures import ThreadPoolExecutor
import asyncio
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

pdf_tasks: Dict[str, Dict] = {}
executor = ThreadPoolExecutor(max_workers=3)
json_handler = JSONHandler()

class PDFProcessingRequest:
    def __init__(self, 
                 prompt_type: str = "policy_json_conversion",
                 priority: Optional[int] = 3,
                 max_tokens: Optional[int] = 4000,
                 temperature: Optional[float] = 0.7):
        self.prompt_type = prompt_type
        self.priority = priority
        self.max_tokens = max_tokens
        self.temperature = temperature

async def process_pdf_content(
    content: bytes,
    task_id: str,
    prompt_type: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
):
    """Process PDF content asynchronously with JSON handling"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        def extract_text_from_pdf(file_path: str) -> str:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text() + '\n'
                return text

        # Process PDF in thread pool
        loop = asyncio.get_event_loop()
        text_content = await loop.run_in_executor(
            executor, 
            extract_text_from_pdf,
            tmp_path
        )

        # Get prompt config
        config = get_prompt_config(prompt_type)
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": config.system_content},
            {"role": "user", "content": text_content}
        ]

        # Process with OpenAI using chunks
        chunks = []
        partial_content = ""
        retry_count = 0

        while True:
            try:
                # Prepare messages with continuation context
                current_messages = messages.copy()
                if partial_content:
                    current_messages.append({
                        "role": "user",
                        "content": f"Continue from: {partial_content}"
                    })

                # Make API request
                response = await openai_client.get_complete_response(
                    messages=current_messages,
                    config=config,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Process chunk using json handler
                chunk = await json_handler.process_chunk(response)
                if chunk:
                    chunks.append(chunk)

                # Check if response is complete
                if any(marker in str(response) for marker in config.completion_markers):
                    break

                # Get continuation point using chunk handler
                completed, partial_content = json_handler.find_continuation_point(str(response))
                if not partial_content:
                    break

                if len(chunks) >= 5:  # Maximum chunks limit
                    logger.warning("Maximum chunk limit reached")
                    break

            except JSONProcessingError as e:
                logger.warning(f"JSON processing error: {str(e)}")
                retry_count += 1
                if retry_count >= 3:  # Maximum retries
                    if chunks:
                        final_result = await json_handler.merge_chunks(chunks)
                        pdf_tasks[task_id].update({
                            'status': 'completed',
                            'result': final_result,
                            'completion_time': datetime.utcnow().isoformat()
                        })
                    else:
                        pdf_tasks[task_id].update({
                            'status': 'failed',
                            'error': f"Failed to process JSON after {retry_count} retries"
                        })
                    return
                await asyncio.sleep(2 ** (retry_count - 1))
                continue

        # Merge chunks and update task status
        try:
            final_result = await json_handler.merge_chunks(chunks)
            pdf_tasks[task_id].update({
                'status': 'completed',
                'result': final_result,
                'completion_time': datetime.utcnow().isoformat()
            })
        except JSONProcessingError as e:
            if chunks:
                pdf_tasks[task_id].update({
                    'status': 'completed',
                    'result': chunks[-1],
                    'completion_time': datetime.utcnow().isoformat()
                })
            else:
                pdf_tasks[task_id].update({
                    'status': 'failed',
                    'error': f"Failed to merge JSON chunks: {str(e)}"
                })

        # Cleanup temporary file
        os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Error processing PDF {task_id}: {str(e)}")
        pdf_tasks[task_id].update({
            'status': 'failed',
            'error': str(e)
        })

@app.post("/api/process-pdf")
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt_type: str = "policy_json_conversion",
    priority: Optional[int] = 3,
    max_tokens: Optional[int] = 4000,
    temperature: Optional[float] = 0.7
):
    """Process PDF file and return task ID"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        task_id = str(uuid.uuid4())
        content = await file.read()
        
        pdf_tasks[task_id] = {
            'status': 'processing',
            'filename': file.filename,
            'prompt_type': prompt_type,
            'submission_time': datetime.utcnow().isoformat(),
            'priority': priority
        }

        background_tasks.add_task(
            process_pdf_content,
            content,
            task_id,
            prompt_type,
            max_tokens,
            temperature
        )

        return {
            'task_id': task_id,
            'status': 'processing',
            'message': 'PDF processing started'
        }

    except Exception as e:
        logger.error(f"Error initiating PDF processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pdf-status/{task_id}")
async def get_pdf_status(task_id: str):
    """Get PDF processing task status"""
    if task_id not in pdf_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return pdf_tasks[task_id]


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
        "sessions_active": len(sessions),
        "pdf_tasks": {
            "active": len(pdf_tasks),
            "completed": len([t for t in pdf_tasks.values() if t['status'] == 'completed']),
            "failed": len([t for t in pdf_tasks.values() if t['status'] == 'failed'])
        }
    }

# Add cleanup handler
@app.on_event("shutdown")
async def cleanup_pdf_tasks():
    """Cleanup PDF tasks and executor"""
    current_time = datetime.utcnow()
    for task_id in list(pdf_tasks.keys()):
        task = pdf_tasks[task_id]
        if task['status'] in ['completed', 'failed']:
            completion_time = datetime.fromisoformat(
                task.get('completion_time', current_time.isoformat())
            )
            if (current_time - completion_time).total_seconds() > 3600:
                del pdf_tasks[task_id]
    
    executor.shutdown(wait=True)