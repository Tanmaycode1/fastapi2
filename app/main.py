from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging
import os
import time
import json
from pydantic import BaseModel
import tempfile
import uuid
from datetime import datetime, timedelta
import asyncio
from dotenv import load_dotenv
import subprocess
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import pypdf
import shutil
from fastapi import BackgroundTasks, File, UploadFile, Depends
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TEMP_DIR = "/tmp/pdf_processing"
DPI = 300
MAX_THREADS = 4
PDF_QUALITY = 100
SESSION_TIMEOUT = 3600  # 1 hour
MAX_MEMORY_PERCENT = 80  # Maximum memory usage threshold
CLEANUP_INTERVAL = 300  # Cleanup every 5 minutes

# Storage for PDF processing tasks and sessions
pdf_tasks: Dict[str, Dict] = {}
active_sessions: Dict[str, datetime] = {}
session_locks: Dict[str, threading.Lock] = {}

class SessionManager:
    def __init__(self):
        self.session_cleanup_task = None
        self.memory_monitor_task = None
        
    async def start(self):
        """Start background tasks for session and memory management"""
        self.session_cleanup_task = asyncio.create_task(self.cleanup_sessions())
        self.memory_monitor_task = asyncio.create_task(self.monitor_memory())
        
    async def cleanup_sessions(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_sessions = [
                    session_id for session_id, last_access 
                    in active_sessions.items()
                    if (current_time - last_access).total_seconds() > SESSION_TIMEOUT
                ]
                
                for session_id in expired_sessions:
                    await self.cleanup_session(session_id)
                    
                # Force garbage collection after cleanup
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {str(e)}")
                
            await asyncio.sleep(CLEANUP_INTERVAL)
    
    async def monitor_memory(self):
        """Monitor system memory usage and trigger cleanup if needed"""
        while True:
            try:
                memory = psutil.virtual_memory()
                if memory.percent > MAX_MEMORY_PERCENT:
                    logger.warning(f"High memory usage detected: {memory.percent}%")
                    # Force cleanup of oldest sessions until memory usage is acceptable
                    while memory.percent > MAX_MEMORY_PERCENT and active_sessions:
                        oldest_session = min(active_sessions.items(), key=lambda x: x[1])[0]
                        await self.cleanup_session(oldest_session)
                        memory = psutil.virtual_memory()
                        
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error in memory monitoring: {str(e)}")
                
            await asyncio.sleep(60)  # Check every minute
            
    async def cleanup_session(self, session_id: str):
        """Clean up a specific session and its resources"""
        try:
            # Acquire session lock
            lock = session_locks.get(session_id)
            if lock:
                lock.acquire()
                
            # Clean up session files
            session_dir = os.path.join(TEMP_DIR, session_id)
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
                
            # Remove session data
            if session_id in pdf_tasks:
                del pdf_tasks[session_id]
            if session_id in active_sessions:
                del active_sessions[session_id]
            if session_id in session_locks:
                del session_locks[session_id]
                
            logger.info(f"Cleaned up session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {str(e)}")
            
        finally:
            if lock:
                lock.release()

class PDFProcessingRequest(BaseModel):
    prompt_type: str = "policy_json_conversion"
    priority: Optional[int] = 3
    max_tokens: Optional[int] = 4000
    temperature: Optional[float] = 0.7
    ocr_language: str = "eng"

# Initialize session manager
session_manager = SessionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize session management on startup"""
    await session_manager.start()
    os.makedirs(TEMP_DIR, exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Clean up all sessions
    for session_id in list(active_sessions.keys()):
        await session_manager.cleanup_session(session_id)
    
    # Clean up temporary directory
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess image for better OCR accuracy"""
    try:
        # Convert to grayscale
        image = image.convert('L')
        
        # Increase contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Denoise
        from PIL import ImageFilter
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
    finally:
        # Ensure proper cleanup of image objects
        gc.collect()

async def process_pdf_content(
    content: bytes,
    task_id: str,
    session_id: str,
    prompt_type: str,
    ocr_language: str = "eng",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
):
    """Process PDF content with session management"""
    session_dir = os.path.join(TEMP_DIR, session_id)
    temp_path = os.path.join(session_dir, task_id)
    os.makedirs(temp_path, exist_ok=True)
    pdf_path = os.path.join(temp_path, "input.pdf")
    
    try:
        # Update session access time
        active_sessions[session_id] = datetime.utcnow()
        
        with open(pdf_path, 'wb') as f:
            f.write(content)

        text_content = ""
        is_scanned = True
        
        # Try normal PDF extraction first
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += page_text + "\n"
                        is_scanned = False
        except Exception as e:
            logger.warning(f"Standard PDF extraction failed: {str(e)}")

        # Use OCR if needed
        if not text_content.strip() or is_scanned:
            logger.info(f"Using OCR for PDF {task_id}")
            
            images = []
            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=DPI,
                    output_folder=temp_path,
                    fmt="png",
                    thread_count=min(MAX_THREADS, os.cpu_count() or 2)
                )

                text_content = f"Total Pages: {len(images)}\n\n"
                
                for i, image in enumerate(images):
                    try:
                        processed_image = preprocess_image(image)
                        page_text = pytesseract.image_to_string(
                            processed_image,
                            lang=ocr_language,
                            config='--oem 3 --psm 6'
                        )
                        text_content += f"\n{'='*50}\nPage {i + 1}\n{'='*50}\n\n"
                        text_content += page_text.strip() + "\n"
                        
                    except Exception as e:
                        logger.error(f"Error processing page {i + 1}: {str(e)}")
                        text_content += f"[Error processing page {i + 1}]\n"
                    finally:
                        # Clean up page images
                        image.close()
                        del image
                        if 'processed_image' in locals():
                            processed_image.close()
                            del processed_image
                        gc.collect()
                        
            finally:
                # Clean up images list
                del images
                gc.collect()

        pdf_tasks[task_id].update({
            'status': 'completed',
            'result': text_content,
            'completion_time': datetime.utcnow().isoformat(),
            'ocr_used': is_scanned
        })

    except Exception as e:
        logger.error(f"Error processing PDF {task_id}: {str(e)}")
        pdf_tasks[task_id].update({
            'status': 'failed',
            'error': str(e)
        })
    
    finally:
        # Clean up temporary files for this task
        try:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

@app.post("/api/process-pdf")
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: PDFProcessingRequest = Depends()
):
    """Process PDF file with session management"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Create new session and task IDs
        session_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        
        # Initialize session
        active_sessions[session_id] = datetime.utcnow()
        session_locks[session_id] = threading.Lock()
        
        content = await file.read()
        
        pdf_tasks[task_id] = {
            'status': 'processing',
            'filename': file.filename,
            'session_id': session_id,
            'prompt_type': request.prompt_type,
            'submission_time': datetime.utcnow().isoformat(),
            'priority': request.priority,
            'ocr_language': request.ocr_language
        }

        background_tasks.add_task(
            process_pdf_content,
            content,
            task_id,
            session_id,
            request.prompt_type,
            request.ocr_language,
            request.max_tokens,
            request.temperature
        )

        return {
            'task_id': task_id,
            'session_id': session_id,
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
    
    task = pdf_tasks[task_id]
    
    # Update session access time if task belongs to a session
    session_id = task.get('session_id')
    if session_id and session_id in active_sessions:
        active_sessions[session_id] = datetime.utcnow()
    
    if task['status'] == 'completed':
        return {
            'status': 'completed',
            'text_content': task['result'],
            'filename': task.get('filename'),
            'completion_time': task.get('completion_time'),
            'ocr_used': task.get('ocr_used', False)
        }
    elif task['status'] == 'failed':
        return {
            'status': 'failed',
            'error': task.get('error'),
            'filename': task.get('filename')
        }
    else:
        return {
            'status': 'processing',
            'filename': task.get('filename')
        }


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