from openai import AsyncOpenAI, AsyncClient, APIError, RateLimitError
import json
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import logging
import tiktoken
from datetime import datetime
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self):
        self.model = "gpt-4-1106-preview"
        self.client = AsyncOpenAI(api_key=self.get_api_key())
        self.encoding = tiktoken.encoding_for_model(self.model)
        self.max_tokens_per_chunk = 15000

    def get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        return api_key

    def create_system_prompt(self) -> str:
        return """You are a medical text analysis system. Extract and structure information from medical texts into JSON format.

        Extract and organize the following elements if present:
        1. Basic Policy Information:
           - title: Drug/policy name
           - policy_number: If present
           - effective_dates: Both original and current
           - description: Brief overview

        2. Coverage Criteria:
           - indications: List of approved conditions
           - age_requirements: Age restrictions
           - prerequisite_criteria: Required tests or conditions
           - prior_therapy: Required previous treatments
           - contraindications: When not to use
           - renewal_criteria: Requirements for continuation

        3. Specific Requirements:
           - testing_requirements: Like TB tests
           - documentation_needed: Required paperwork
           - quantity_limits: Dosing restrictions
           - specialty_requirements: Prescriber restrictions

        Format all criteria in clear, structured JSON with appropriate nesting and grouping."""

    def create_parsing_prompt(self, text: str) -> str:
        return f"""Parse this medical policy text into structured JSON.
        Extract all relevant criteria, requirements, and guidelines.
        Maintain the original organization but structure it as JSON.

        Text to analyze:
        {text}"""

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using model's tokenizer."""
        return len(self.encoding.encode(text))

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while preserving context."""
        chunks = []
        current_chunk = ""
        current_tokens = 0

        paragraphs = text.split('\n\n')

        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)

            if current_tokens + paragraph_tokens > self.max_tokens_per_chunk:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                current_tokens += paragraph_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=lambda retry_state: logger.info(f"Retrying after error: {retry_state.outcome.exception()}")
    )
    async def process_chunk(self, chunk: str) -> Dict[str, Any]:
        """Process a single chunk of text with retries and error handling."""
        try:
            messages = [
                {"role": "system", "content": self.create_system_prompt()},
                {"role": "user", "content": self.create_parsing_prompt(chunk)}
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )

            result = response.choices[0].message.content

            try:
                return json.loads(result)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {str(e)}\nResponse: {result}")
                raise HTTPException(status_code=500, detail="Failed to parse OpenAI response into JSON")

        except Exception as e:
            logger.error(f"Error in process_chunk: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple chunk results intelligently."""
        if not results:
            return {}

        final_result = results[0].copy()

        for result in results[1:]:
            for key, value in result.items():
                if key not in final_result:
                    final_result[key] = value
                else:
                    if isinstance(value, list) and isinstance(final_result[key], list):
                        final_result[key] = list(dict.fromkeys(final_result[key] + value))
                    elif isinstance(value, dict) and isinstance(final_result[key], dict):
                        final_result[key].update(value)
                    elif isinstance(value, str) and isinstance(final_result[key], str):
                        if value.lower() not in final_result[key].lower():
                            final_result[key] = f"{final_result[key]} {value}".strip()
                    else:
                        if value and not final_result[key]:
                            final_result[key] = value

        return final_result

    async def process_full_text(self, text: str) -> Dict[str, Any]:
        """Process entire text document, handling chunks and merging results."""
        try:
            # Split text into manageable chunks
            chunks = self.split_text(text)

            # Process each chunk
            results = []
            for chunk in chunks:
                chunk_result = await self.process_chunk(chunk)
                results.append(chunk_result)

            # Merge results
            final_result = await self.merge_results(results)

            # Add metadata
            final_result["metadata"] = {
                "processed_at": datetime.now().isoformat(),
                "chunks_processed": len(chunks),
                "model_used": self.model
            }

            return final_result

        except Exception as e:
            logger.error(f"Error in process_full_text: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def estimate_cost(self, text: str) -> float:
        """Estimate processing cost based on token count."""
        token_count = self.count_tokens(text)
        return round((token_count / 1000) * 0.01, 4)  # $0.01 per 1K tokens