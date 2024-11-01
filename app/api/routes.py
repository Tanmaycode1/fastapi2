from openai import AsyncOpenAI, APIError, RateLimitError
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
        api_key = "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"
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        return api_key

    def create_system_prompt(self) -> str:
        return """You are a medical text analysis system. Extract and structure information from medical texts into JSON format.

        Always include these core fields if information is present:
        - title: Main topic, drug name, or policy title
        - description: Overview of the content
        - applicable_drug_list: Relevant medications
        - policy_criteria: Object containing criteria and logic for usage
        - quantity_level_limit: Any quantity restrictions

        Guidelines for JSON structure:
        1. Core fields should be at the top level
        2. Create nested objects for related information
        3. Use arrays for lists and multiple items
        4. Use camelCase for all field names
        5. Include medical sections as separate fields
        6. Preserve medical terminology exactly
        7. Group related information together
        8. Add any relevant fields based on content
        9. Include numerical values where appropriate
        10. Parse criteria into structured format"""

    def create_parsing_prompt(self, text: str, context: Optional[Dict] = None) -> str:
        base_prompt = f"""Parse the following medical text into a structured JSON format.
        Extract all relevant information and organize it according to the guidelines.
        Ensure all numerical values, criteria, and medical terms are preserved accurately.

        Text to analyze:
        {text}"""

        if context:
            base_prompt += f"\n\nPrevious context: {json.dumps(context, indent=2)}"

        return base_prompt

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
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    async def process_chunk(self,
                          chunk: str,
                          previous_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a single chunk of text with retries and error handling."""
        try:
            messages = [
                {"role": "system", "content": self.create_system_prompt()},
                {"role": "user", "content": self.create_parsing_prompt(chunk, previous_context)}
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
                parsed_result = json.loads(result)
                return self.validate_and_clean_json(parsed_result)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {result}")
                raise HTTPException(status_code=500, detail="Failed to parse OpenAI response")

        except Exception as e:
            logger.error(f"Error in process_chunk: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def validate_and_clean_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the JSON structure."""
        cleaned = {}

        # Ensure core fields exist
        core_fields = {
            "title": "",
            "description": "",
            "applicable_drug_list": [],
            "policy_criteria": {
                "criteria": [],
                "logic": ""
            },
            "quantity_level_limit": ""
        }

        # Copy core fields if they exist
        for field, default in core_fields.items():
            if field in data:
                cleaned[field] = data[field]
            else:
                cleaned[field] = default

        # Convert applicable_drug_list to list if it's a string
        if isinstance(cleaned["applicable_drug_list"], str):
            cleaned["applicable_drug_list"] = [x.strip() for x in cleaned["applicable_drug_list"].split(",") if x.strip()]

        # Add any additional fields from the original data
        for key, value in data.items():
            if key not in core_fields:
                cleaned[key] = value

        return cleaned

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
            chunks = self.split_text(text)
            results = []
            previous_context = None

            for chunk in chunks:
                chunk_result = await self.process_chunk(chunk, previous_context)
                results.append(chunk_result)
                previous_context = chunk_result

            final_result = await self.merge_results(results)

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