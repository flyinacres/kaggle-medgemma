import json5
import re
import sys
import html
from typing import Dict, List, Any, Optional


def extract_json_from_text(raw_text: str) -> Optional[str]:
    """
    Extract JSON from messy LLM output with multiple strategies.
    Returns the last valid JSON object found, or None.
    """
    text = raw_text.strip()

    # Strategy 1: Look for ```json blocks and try the last one
    json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_blocks:
        for block in reversed(json_blocks):
            try:
                json5.loads(block)
                return block
            except ValueError:
                continue
    
    # Strategy 2: Find all complete {...} blocks and try the last valid one
    potential_jsons = []
    depth = 0
    start = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                potential_jsons.append(text[start:i+1])
                start = -1
    
    for candidate in reversed(potential_jsons):
        try:
            json5.loads(candidate)
            return candidate
        except ValueError:
            continue
    
    return None


def safe_get_list(data: Dict, key: str) -> List[str]:
    """Safely extract a list of strings, handling errors and duplicates."""
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    
    seen = set()
    result = []
    for item in value:
        item_str = html.escape(str(item).strip()) if item is not None else ""
        if item_str and item_str not in seen:
            seen.add(item_str)
            result.append(item_str)
    return result


def safe_get_dict_list(data: Dict, key: str) -> List[Dict]:
    """Safely extract list of dictionaries."""
    value = data.get(key)
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def safe_get_string(data: Dict, key: str, default: str = '') -> str:
    """Safely get string value and escape for HTML safety."""
    value = data.get(key)
    if value is None or value == '':
        return default
    return html.escape(str(value).strip())


def format_medical_info(json_data: Dict[str, Any]) -> str:
    """
    Formats the structured medical JSON using semantic HTML tags 
    that Quill's clipboard parser recognizes and maps to its toolbar.
    """
    output_parts = []
    
    # 1. Header & Disclaimer
    output_parts.append("<h2>Medical Summary</h2>")
    output_parts.append("<blockquote><b>⚠️ DISCLAIMER:</b> Not medical advice. Consult a professional.</blockquote>")

    # 2. Key Takeaways (Bullet List)
    takeaways = safe_get_list(json_data, 'key_takeaways')
    if takeaways:
        output_parts.append("<h3>📌 Key Takeaways</h3><ul>")
        for item in takeaways:
            output_parts.append(f"<li>{item}</li>")
        output_parts.append("</ul>")

    # 3. Medications 
    medications = safe_get_dict_list(json_data, 'medications')
    if medications:
        output_parts.append("<h3>💊 Medications</h3>")
        for med in medications:
            name = safe_get_string(med, 'name', 'Unknown')
            dosage = safe_get_string(med, 'dosage')
            admin = safe_get_string(med, 'administration')
            desc = safe_get_string(med, 'description')
            
            # Use single-level paragraphs with indentation or bullets
            output_parts.append(f"<p><b>• {name}</b></p>")
            if dosage: output_parts.append(f"<p style='margin-left: 20px;'>- Dosage: {dosage}</p>")
            if admin:  output_parts.append(f"<p style='margin-left: 20px;'>- How to take: {admin}</p>")
            if desc:   output_parts.append(f"<p style='margin-left: 20px;'><i>{desc}</i></p>")
        output_parts.append("<p><br></p>")

    # 4. Terms (Bold and Inline)
    terms = safe_get_dict_list(json_data, 'medical_terms')
    if terms:
        output_parts.append("<h3>📖 Terms Explained</h3>")
        for term_obj in terms:
            term = safe_get_string(term_obj, 'term', 'Unknown')
            defn = safe_get_string(term_obj, 'definition', 'N/A')
            output_parts.append(f"<p><b>{term}</b>: {defn}</p>")
        output_parts.append("<p><br></p>")

    # 5. Questions (Ordered List)
    questions = safe_get_list(json_data, 'questions_for_provider')
    if questions:
        output_parts.append("<h3>❓ Questions for Provider</h3><ol>")
        for q in questions:
            output_parts.append(f"<li>{q}</li>")
        output_parts.append("</ol>")
        output_parts.append("<p><br></p>")
    
    return "".join(output_parts)


def format_medical_info_from_string(raw_text: str) -> str:
    """Extracts JSON and converts to HTML directly from LLM string."""
    json_str = extract_json_from_text(raw_text)
    if not json_str:
        return "<p>Could not find valid JSON data.</p>"
    
    try:
        json_data = json5.loads(json_str)
        return format_medical_info(json_data)
    except Exception as e:
        return f"<p>Error parsing JSON: {html.escape(str(e))}</p>"