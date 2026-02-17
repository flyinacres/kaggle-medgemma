# core_logic.py

import json5
from services.llm.engine import generate_summary
from core.parse_json import extract_json_from_text, format_medical_info
import re

def strip_xml_tags(text: str) -> str:
    """
    If the text is wrapped in a single XML-like tag, strips the tag.
    Otherwise, returns the original text.
    Handles tags like <answer>...</answer> or <response>...</response>.
    """
    # This pattern looks for <tag>content</tag> and captures the content.
    match = re.search(r"<(?P<tag>\w+)>(?P<content>.*?)</(?P=tag)>", text.strip(), re.DOTALL)
    
    if match:
        # If a match is found, return the captured content, stripped of whitespace.
        return match.group("content").strip()
    
    # If no tags are found, return the original text.
    return text.strip()

def get_llm_summary(text,image_path):
    """
    This function encapsulates the entire long-running task.
    It is executed in the background thread by the animator.
    It must RETURN a final string, not yield.
    """
    # 1. Call the long-running AI function
    summary_result = generate_summary("system_prompt", text, image_path)
    # print("Summary result: ", summary_result)

    # 2. Clean and parse the result
    json_str = extract_json_from_text(summary_result)
    if not json_str:
        # Return an error string for the UI
        # return "Error: The AI engine did not return a valid summary structure. Please try again."
        # Better to return a poorly formatted string, rather than just an error message...
        return summary_result

    json_data = json5.loads(json_str)
    
    # 3. Format the JSON into readable text
    formatted_text = format_medical_info(json_data)
    
    # 4. Return the final, display-ready text
    return formatted_text


def get_follow_up_answer(original_text, summary, history, new_question, image_path):
    """
    (New Function) Generates an answer to a follow-up question.
    
    This function constructs a detailed prompt that gives the LLM all necessary
    context to provide a relevant, grounded answer.
    """
    # Construct a history string from the conversation
    history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in history])

    # Construct the full prompt for the LLM
    chat_text = f"""

<medical_text>
{original_text}
</medical_text>

<summary_of_text>
{summary}
</summary_of_text>

<conversation_history>
{history_str}
</conversation_history>

<user_question>
{new_question}
</user_question>

Provide your answer directly.
"""
    # Make the call to the LLM. This is expected to be a faster, simpler call.
    raw_answer = generate_summary("conversational_prompt", chat_text, image_path) 
    
    # Sometimes the LLM likes to resond with XML...
    clean_answer = strip_xml_tags(raw_answer)

    return clean_answer