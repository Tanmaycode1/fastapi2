from typing import Dict, Any, List
import re
from .openai_service import OpenAIService
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self):
        self.openai_service = OpenAIService()
        # Core fields that should always be present
        self.core_fields = {
            "title": "",
            "description": "",
            "applicable_drug_list": "",
            "policy_criteria": {
                "criteria": [],
                "logic": ""
            }
        }

        # Common medical section headers to look for
        self.medical_sections = [
            "Dosage & Administration",
            "Indications & Usage",
            "Dosage Forms & Strengths",
            "Pregnancy & Lactation",
            "Contraindications",
            "Warnings & Precautions",
            "Adverse Reactions",
            "Drug Interactions",
            "Description",
            "Pharmacology",
            "Clinical Studies",
            "How Supplied/Storage & Handling",
            "Instructions for Use",
            "Mechanism of Action"
        ]

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections based on headers in text."""
        sections = {}
        current_section = None
        lines = text.split('\n')

        # Pattern to match section headers (all caps or title case with common medical terms)
        header_pattern = r'^([A-Z][A-Za-z\s&]+:|[A-Z\s&]+$)'

        for line in lines:
            match = re.match(header_pattern, line.strip())
            if match:
                # Clean up header text
                header = match.group(1).replace(':', '').strip()
                if header in self.medical_sections or header.isupper():
                    current_section = header
                    sections[current_section] = ""
                continue

            if current_section and line.strip():
                sections[current_section] = sections.get(current_section, "") + line + "\n"

        return sections

    def create_processing_prompt(self, text: str) -> str:
        """Create a prompt that ensures core fields while allowing flexibility."""
        return f"""Analyze this medical text and structure it as JSON. 

        Required fields to include (if information is found):
        - title: Main topic or drug name
        - description: Brief overview
        - applicable_drug_list: Related medications
        - policy_criteria: Various criteria and logic for usage

        Also include any additional relevant sections you find in the text.
        Create appropriate fields for any medical sections present.
        Group related information together.
        Omit any sections where no relevant information is found.

        Text to analyze:
        {text}
        """

    async def process_text(self, text: str) -> Dict[str, Any]:
        """Process text while maintaining core fields and allowing flexibility."""
        try:
            # Initialize result with core fields
            result = self.core_fields.copy()

            # Extract sections from text
            sections = self.extract_sections(text)

            # Process text with OpenAI
            response = await self.openai_service.process_chunk(
                self.create_processing_prompt(text)
            )

            # Ensure core fields are present
            for key in self.core_fields:
                if key in response:
                    result[key] = response[key]

            # Add any additional sections found
            for key, value in response.items():
                if key not in self.core_fields:
                    result[key] = value

            # Add extracted sections if they're not already included
            for section, content in sections.items():
                section_key = self.normalize_key(section)
                if section_key not in result:
                    result[section_key] = content.strip()

            return result

        except Exception as e:
            logger.error(f"Error in process_text: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def normalize_key(self, key: str) -> str:
        """Convert section headers to valid JSON keys."""
        # Remove special characters and convert to camelCase
        key = re.sub(r'[^\w\s]', '', key)
        words = key.split()
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

    def extract_policy_criteria(self, text: str) -> Dict[str, Any]:
        """Extract policy criteria and logic from text."""
        criteria_pattern = r'(?i)(?:criteria|requirements|guidelines):\s*(.*?)(?=\n\n|\Z)'
        logic_pattern = r'(?i)(?:logic|rules|protocol):\s*(.*?)(?=\n\n|\Z)'

        criteria_match = re.search(criteria_pattern, text, re.DOTALL)
        logic_match = re.search(logic_pattern, text, re.DOTALL)

        criteria = []
        if criteria_match:
            criteria_text = criteria_match.group(1)
            criteria = [c.strip() for c in criteria_text.split('\n') if c.strip()]

        logic = logic_match.group(1).strip() if logic_match else ""

        return {
            "criteria": criteria,
            "logic": logic
        }

    def estimate_cost(self, text: str) -> float:
        """Estimate processing cost based on text length."""
        # Approximate token count (1 token ≈ 4 characters)
        token_count = len(text) / 4
        # GPT-4 costs approximately $0.01 per 1K tokens
        return round((token_count / 1000) * 0.01, 4)