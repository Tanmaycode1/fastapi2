from typing import List, Dict, Any
import re
from ..core.config import get_settings


class ChunkProcessor:
    def __init__(self):
        self.settings = get_settings()

    def split_into_sections(self, text: str) -> Dict[str, str]:
        """Split text into logical sections based on headers."""
        sections = {}
        current_section = "General"
        pattern = r'^([A-Z][A-Za-z\s&]+):'

        for line in text.split('\n'):
            match = re.match(pattern, line)
            if match:
                current_section = match.group(1).strip()
                sections[current_section] = ""
            else:
                sections[current_section] = sections.get(current_section, "") + line + "\n"

        return sections

    def merge_sections(self, sections: List[Dict[str, str]]) -> Dict[str, str]:
        """Merge multiple section dictionaries intelligently."""
        merged = {}
        for section_dict in sections:
            for key, value in section_dict.items():
                if key in merged:
                    merged[key] = merged[key].strip() + "\n" + value.strip()
                else:
                    merged[key] = value.strip()
        return merged