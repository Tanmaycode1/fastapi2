import requests
import json
import re
from typing import Dict, Any

class EncodingVisualizer:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url

    def generate_encoding(self, text: str) -> Dict[str, Any]:
        """Generate encoding from input text"""
        try:
            print("\n1. Generating encoding from text...")
            response = requests.post(
                f"{self.api_base_url}/api/generate-encoding",
                json={"body": text}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Error generating encoding: {str(e)}")

    def visualize_logic_tree(self, logic: str) -> str:
        """Convert logic string to visual tree structure"""
        def parse_logic(logic: str) -> tuple:
            # Clean up the input
            logic = re.sub(r'\s+', ' ', logic.strip())
            
            # Match operator and content
            match = re.match(r'(AND|OR|NOT)\((.*)\)', logic)
            if not match:
                return None, logic
            
            operator, content = match.groups()
            
            # Parse nested content
            depth = 0
            current = ""
            children = []
            
            for char in content:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == ',' and depth == 0:
                    children.append(current.strip())
                    current = ""
                    continue
                current += char
            
            if current:
                children.append(current.strip())
            
            return operator, children

        def build_tree(logic: str, indent: str = "") -> str:
            operator, children = parse_logic(logic)
            
            if not operator:
                # Handle variable definitions with :=
                if ':=' in logic:
                    var, definition = logic.split(':=')
                    result = indent + "├── " + var.strip() + " :=\n"
                    result += build_tree(definition.strip(), indent + "│   ")
                    return result
                return indent + "├── " + logic + "\n"
            
            result = indent + "├── " + operator + "\n"
            for child in children:
                result += build_tree(child, indent + "│   ")
            return result

        return "Logic Tree:\n" + build_tree(logic)

def main():
    visualizer = EncodingVisualizer()
    
    # Example policy text
    policy_text = """
    Medical Policy for Treatment A:
    
    Inclusion Criteria:
    1. Patient must be 18 years or older
    2. Must have diagnosis of condition X
    3. Must meet both:
        a) Failed previous therapy
        b) No contraindications
    
    Exclusion Criteria:
    1. Active infection
    2. Pregnancy
    """
    
    try:
        # Generate encoding
        encoding = visualizer.generate_encoding(policy_text)
        
        # Display the full encoding JSON
        print("\n2. Generated Encoding:")
        print(json.dumps(encoding, indent=2))
        
        # Display the logic tree
        if 'logic' in encoding:
            print("\n3. Logic Tree Visualization:")
            print(visualizer.visualize_logic_tree(encoding['logic']))
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()