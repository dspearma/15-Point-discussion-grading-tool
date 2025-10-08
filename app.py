import streamlit as st
import pandas as pd
import io
import sys
import os
import requests
import csv
import json
import re
import time
import math
from typing import Dict, Any, List, Union
from difflib import SequenceMatcher
import plotly.express as px

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("python-dotenv not installed. API key must be entered manually.")

# ============================================
# ALL FUNCTION DEFINITIONS GO HERE
# ============================================

def validate_api_key(api_key: str) -> bool:
    """Validate the OpenRouter API key format."""
    if not api_key:
        return False
    return api_key.startswith("sk-or-v1-") and len(api_key) > 20

def strip_tags(text: str) -> str:
    """Removes HTML tags from a string."""
    pattern = re.compile(r'<[^>]*>')
    return re.sub(pattern, '', text)

def fix_encoding(text: Union[str, bytes]) -> str:
    """Fix common encoding issues and replace problematic characters."""
    if not isinstance(text, (str, bytes)):
        return str(text)

    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='ignore')

    # Common encoding fixes for smart quotes/dashes/html entities
    replacements = {
        'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '–', 'â€"': '—',
        '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
        '\u2013': '-', '\u2014': '—',
        'â€"': '—',
        '&nbsp;': ' ',
        '&quot;': '"'
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        pass

    return text

def recursively_clean(value: Any) -> Any:
    """Recursively clean all string values in a data structure using fix_encoding."""
    if isinstance(value, str):
        return fix_encoding(value)
    elif isinstance(value, dict):
        return {k: recursively_clean(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [recursively_clean(item) for item in value]
    else:
        return value

def round_nearest_half(value: float) -> float:
    """Round a float to the nearest 0.5 increment."""
    return round(value * 2) / 2

def retry_with_backoff(func, max_retries: int = 3, base_delay: int = 5, max_delay: int = 30):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise e

            delay = min(base_delay * (2 ** attempt), max_delay)
            time.sleep(delay)

def robust_api_call(api_url: str, headers: Dict, payload: Dict, timeout: int = 60, max_retries: int = 3, api_key: str = None) -> str:
    """Make API call with retries and proper error handling."""
    def api_call():
        # Check if API key is provided
        if not api_key:
            raise ValueError("No API key provided")
        
        # Ensure the API key is properly formatted
        if not api_key.startswith("sk-or-v1-"):
            raise ValueError(f"Invalid API key format. Expected format: sk-or-v1-...")
        
        # Create a new headers dictionary to avoid any potential issues
        new_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app-url.com",
            "X-Title": "Discussion Grading Tool"
        }
        
        # Make the API call
        response = requests.post(api_url, headers=new_headers, json=payload, timeout=timeout)
        
        # Handle specific HTTP errors
        if response.status_code == 401:
            raise ValueError("Authentication failed. Please check your API key.")
        elif response.status_code == 429:
            raise ValueError("Rate limit exceeded. Please try again later.")
        elif response.status_code >= 500:
            raise ValueError(f"Server error: {response.status_code}")
        
        response.raise_for_status()
        result = response.json()

        if not result.get("choices"):
            raise ValueError("Invalid API response format - missing choices")

        if result["choices"][0]["message"]["content"] is None:
             raise ValueError("Empty content block received from API.")

        grade_response = result["choices"][0]["message"]["content"]
        if not grade_response.strip():
            raise ValueError("Empty API response")

        return grade_response

    return retry_with_backoff(api_call, max_retries=max_retries)

def robust_json_parsing(response_text: str, max_retries: int = 2) -> Dict:
    """Parse clean JSON, primarily expecting the LLM to follow the format."""
    def parse_json():
        response_text_clean = fix_encoding(response_text.strip())

        # Clean common LLM formatting issues (code fences)
        response_text_clean = re.sub(r'^```json\s*', '', response_text_clean, flags=re.DOTALL)
        response_text_clean = re.sub(r'\s*```$', '', response_text_clean, flags=re.DOTALL)
        response_text_clean = re.sub(r'^```\s*', '', response_text_clean, flags=re.DOTALL)

        # Attempt standard JSON loading
        try:
            return json.loads(response_text_clean)
        except json.JSONDecodeError as e:
            # Use regex to find the likely JSON object
            json_match = re.search(r'\{.*\}', response_text_clean, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                try:
                    return json.loads(extracted_json)
                except json.JSONDecodeError as e2:
                    raise ValueError(f"Could not parse valid JSON even after extraction: {str(e2)}")

            raise ValueError(f"Could not find or parse valid JSON structure: {str(e)}")

    return retry_with_backoff(parse_json, max_retries=max_retries)

def extract_section_robust(text: str, section_name: str, alternative_names: List[str] = None, stop_indicators: List[str] = None) -> str:
    """
    Robust section extraction with better boundary detection.
    Correctly handles headers followed by content on the same line.
    """
    alternative_names = alternative_names or []
    stop_indicators = stop_indicators or ['Discussion', 'Reading', 'Video', 'Key Terms', 'Objective', 'Key Concepts', 'Materials', 'Prompt']

    all_names = [section_name] + alternative_names
    content_lines = []
    in_section = False

    lines = text.split('\n')

    for i, line in enumerate(lines):
        line_clean = line.strip()

        # Check for start of section
        is_header = False

        if not in_section:
            for name in all_names:
                name_escaped = re.escape(name)
                pattern = r'^\s*' + name_escaped + r'[:\s]*(.*)$'
                match_start = re.match(pattern, line_clean, re.IGNORECASE)

                if match_start:
                    in_section = True
                    is_header = True

                    potential_content = match_start.group(1).strip()
                    if potential_content:
                        content_lines.append(potential_content)
                    break

        if is_header:
            continue

        if in_section:
            # Check for stop condition
            is_stop_indicator = False
            for indicator in stop_indicators:
                if re.match(r'^\s*' + re.escape(indicator) + r'[:\s]*', line_clean, re.IGNORECASE) and len(line_clean.split()) < 8:
                    is_stop_indicator = True
                    break

            if is_stop_indicator and line_clean:
                in_section = False
                break

            # Continue collecting content
            if line_clean:
                content_lines.append(line_clean)

    if not content_lines:
        return ""

    content = '\n'.join(content_lines).strip()
    return content

def parse_lesson_plan_comprehensive(docx_content: bytes):
    """Comprehensive lesson plan parser from DOCX content."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx library required. Please install it with: pip install python-docx")

    docx_io = io.BytesIO(docx_content)
    document = Document(docx_io)

    # Extract all text
    full_text = "\n".join([p.text.strip() for p in document.paragraphs if p.text.strip()])
    full_text_clean = fix_encoding(full_text)

    # Define sections and their required alternate names/stop indicators dynamically
    sections_to_extract = {
        "Discussion Prompt": (["Discussion", "Prompt", "Discussion Question", "Question"], ['Reading', 'Video', 'Key Terms', 'Objective', 'Key Concepts', 'Materials']),
        "Reading": (["Assigned Reading", "Required Reading", "READING"], ['Discussion', 'Video', 'Key Terms', 'Objective', 'Key Concepts', 'Materials', 'Prompt']),
        "Video": (["VIDEO", "Assigned Video", "Required Video"], ['Discussion', 'Reading', 'Key Terms', 'Objective', 'Key Concepts', 'Materials', 'Prompt']),
        "Objective": (["Learning Objective", "Goals"], ['Discussion', 'Reading', 'Video', 'Key Terms', 'Key Concepts', 'Materials', 'Prompt']),
        "Key Concepts": (["Concepts", "Main Concepts", "KEY CONCEPTS"], ['Discussion', 'Reading', 'Video', 'Key Terms', 'Objective', 'Materials', 'Prompt']),
        "Key Terms": (["KEY TERMS", "Terms", "Vocabulary"], ['Discussion', 'Reading', 'Video', 'Objective', 'Key Concepts', 'Materials', 'Prompt'])
    }

    parsed_sections = {}
    for name, (alts, stops) in sections_to_extract.items():
        parsed_sections[name] = extract_section_robust(full_text_clean, name, alts, stops)

    # --- Key Terms Specific Parsing ---
    key_terms_str = parsed_sections["Key Terms"]
    key_terms = []

    if key_terms_str:
        key_terms_clean = re.sub(r'[\t\-\*\•]', '\n', key_terms_str)
        terms_list = re.split(r',\s*|\n', key_terms_clean)

        for term in terms_list:
            term = term.strip()
            term = re.sub(r'^[\d\.\s\-\)]+', '', term)
            term = re.sub(r'[\.,;\s\-]+$', '', term)

            if term and len(term) > 2:
                key_terms.append(term)

    unique_key_terms = list(set([term for term in key_terms if len(term) > 2]))
    unique_key_terms.sort()

    # --- Fallback Discussion Prompt ---
    discussion_prompt = parsed_sections["Discussion Prompt"]
    if not discussion_prompt.strip():
        # Fallback 1: Use Objective
        if parsed_sections["Objective"].strip():
            discussion_prompt = parsed_sections["Objective"]
        else:
            # Fallback 2: Look for questions in the main text
            question_match = re.search(r'([A-Za-z\s]+[?])', full_text_clean[-1000:], re.DOTALL)
            if question_match:
                discussion_prompt = question_match.group(1).strip()

    return (
        discussion_prompt,
        parsed_sections["Reading"],
        parsed_sections["Video"],
        unique_key_terms,
        parsed_sections["Objective"],
        parsed_sections["Key Concepts"]
    )

def normalize_text_for_matching(text: str) -> str:
    if not text: return ""
    text = strip_tags(text)
    text = re.sub(r'[*_`#\-\,.:;]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def advanced_term_variations(term: str) -> List[str]:
    term_clean = normalize_text_for_matching(term)
    variations = set([term_clean])

    if not term_clean or len(term_clean) < 3:
        return []

    if term_clean.endswith('s'):
        variations.add(term_clean[:-1])
    elif not term_clean.endswith('s'):
        variations.add(term_clean + 's')

    if '-' in term_clean:
        variations.add(term_clean.replace('-', ' '))

    return list(variations)

def detect_key_terms_presence(submission_text: str, key_terms: List[str]) -> List[str]:
    if not key_terms or not submission_text:
        return []

    submission_norm = normalize_text_for_matching(submission_text)
    detected_terms = []
    detected_base_terms = set()

    for term in key_terms:
        if term in detected_base_terms: continue

        term_variations = advanced_term_variations(term)

        for variation in term_variations:
            try:
                pattern = r'\b' + re.escape(variation) + r'\b'
                if re.search(pattern, submission_norm):
                    detected_terms.append(term)
                    detected_base_terms.add(term)
                    break
                if len(variation.split()) == 1:
                    for submission_word in submission_norm.split():
                        if similarity_ratio(variation, submission_word) >= 0.9:
                            detected_terms.append(term)
                            detected_base_terms.add(term)
                            break
                    if term in detected_base_terms:
                        break
            except re.error:
                continue

    return list(set(detected_terms))

def construct_final_feedback(
    llm_results: Dict,
    local_scores: Dict[str, float],
    local_feedback: Dict[str, str],
    improvement_areas: List[str],
    student_first_name: str,
    grading_scale: str
) -> str:
    prompt_feedback = llm_results.get('prompt_feedback', 'Feedback missing for prompt quality.')
    key_terms_feedback = llm_results.get('key_terms_feedback', local_feedback.get('key_terms_fallback', 'Feedback missing for key terms.'))
    reading_feedback = local_feedback.get('reading_feedback', llm_results.get('reading_feedback', 'Feedback missing for reading reference.'))
    video_feedback = llm_results.get('video_feedback', 'Feedback missing for video reference.')
    general_feedback_llm = llm_results.get('general_feedback', 'Overall submission quality was strong.')

    def transform_to_second_person(text):
        if not text: return ""
        text = re.sub(r'\b(The student|This student|They|Their|He|His|She|Her)\b', lambda m: {'The student': 'You', 'This student': 'You', 'They': 'You', 'Their': 'Your', 'He': 'You', 'His': 'Your', 'She': 'You', 'Her': 'Your'}.get(m.group(1), m.group(1)), text, flags=re.IGNORECASE)
        if text:
            text = text.strip()
            return text[0].upper() + text[1:]
        return text

    prompt_feedback = transform_to_second_person(prompt_feedback)
    key_terms_feedback = transform_to_second_person(key_terms_feedback)
    reading_feedback = transform_to_second_person(reading_feedback)
    video_feedback = transform_to_second_person(video_feedback)
    general_feedback_llm = transform_to_second_person(general_feedback_llm)

    # Format scores based on grading scale
    if grading_scale == "15-point (3 categories)":
        prompt_key_score = local_scores['prompt_key_score']
        reading_score = local_scores['reading_score']
        video_score = local_scores['video_score']
        total_points = 15.0
        
        prompt_key_formatted = f"PROMPT AND KEY TERMS [{prompt_key_score:.1f}/5.0]: {prompt_feedback.strip()} {key_terms_feedback.strip()}."
        video_formatted = f"REFERENCE TO VIDEO [{video_score:.1f}/5.0]: {video_feedback}."
        reading_formatted = f"REFERENCE TO READING [{reading_score:.1f}/5.0]: {reading_feedback}."
    else:  # 16-point (4 categories)
        prompt_score = local_scores['prompt_score']
        key_terms_score = local_scores['key_terms_score']
        reading_score = local_scores['reading_score']
        video_score = local_scores['video_score']
        total_points = 16.0
        
        prompt_formatted = f"PROMPT ADHERENCE [{prompt_score:.1f}/4.0]: {prompt_feedback.strip()}."
        key_terms_formatted = f"KEY TERMS USAGE [{key_terms_score:.1f}/4.0]: {key_terms_feedback.strip()}."
        video_formatted = f"REFERENCE TO VIDEO [{video_score:.1f}/4.0]: {video_feedback}."
        reading_formatted = f"REFERENCE TO READING [{reading_score:.1f}/4.0]: {reading_feedback}."

    if improvement_areas:
        improvement_focus = f"{student_first_name}, while your work demonstrates strong engagement with the content, focus on improving in the area(s) of: {', '.join(improvement_areas)} to maximize your synthesis of the concepts. {general_feedback_llm}"
    else:
        improvement_focus = f"Outstanding work {student_first_name}! You excelled on all sections of this assignment. {general_feedback_llm}"

    general_formatted = f"GENERAL FEEDBACK: {improvement_focus}"

    # Construct final feedback based on grading scale
    if grading_scale == "15-point (3 categories)":
        final_feedback = '\n'.join([
            prompt_key_formatted,
            video_formatted,
            reading_formatted,
            general_formatted
        ])
    else:  # 16-point (4 categories)
        final_feedback = '\n'.join([
            prompt_formatted,
            key_terms_formatted,
            video_formatted,
            reading_formatted,
            general_formatted
        ])

    final_feedback = re.sub(r'\s{2,}', ' ', final_feedback).strip()
    return final_feedback

def similarity_ratio(str1, str2):
    """Default fallback similarity function using SequenceMatcher."""
    if not str1 or not str2:
        return 0
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def grade_submission_with_retries(
    submission_text: str,
    reading_text: str,
    key_terms: List[str],
    discussion_prompt: str,
    student_first_name: str,
    video_text: str,
    replies: List[str],
    api_key: str,
    grading_scale: str
) -> Dict[str, str]:
    """Grade submission with comprehensive local and API scoring."""

    # 1. Local Scoring & Data Preparation
    detected_terms = detect_key_terms_presence(submission_text, key_terms)
    detected_terms_str = ', '.join(detected_terms) if detected_terms else 'none detected'
    reading_info = {}

    # --- Extract REQUIRED READING INFO with MULTIPLE PATTERNS ---
    assigned_author = None

    # First, check if "READING:" is in the text
    if "READING:" in reading_text:
        # Extract the full line that contains "READING:"
        reading_line = ""
        for line in reading_text.split('\n'):
            if "READING:" in line:
                reading_line = line.strip()
                break

        # Extract author (first word after "READING:")
        author_match = re.search(r'READING:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', reading_line)
        if author_match:
            assigned_author = author_match.group(1).strip()
    
    # Fallback if "READING:" is not explicitly in the text or author not found
    if not assigned_author:
        # Try multiple author extraction patterns
        author_patterns = [
            r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "by Author Name"
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',     # "Author Name" at start of line
            r'Reading:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "Reading: Author Name"
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, reading_text, re.IGNORECASE | re.MULTILINE)
            if match:
                assigned_author = match.group(1).strip()
                break
    
    # If still no author, try to extract from the first few words
    if not assigned_author:
        words = reading_text.split()[:5]  # Check first 5 words
        for word in words:
            if re.match(r'^[A-Z][a-z]+$', word) and len(word) > 2:
                assigned_author = word
                break
    
    reading_info = {
        "assigned_author": assigned_author if assigned_author else "Unknown",
        "reading_mentioned": bool(reading_text.strip())
    }

    # --- Extract VIDEO INFO with MULTIPLE PATTERNS ---
    video_mentioned = bool(video_text.strip())
    video_title = None
    
    if video_mentioned:
        # Try to extract video title
        video_patterns = [
            r'VIDEO:\s*(.+)',  # "VIDEO: Title"
            r'Video:\s*(.+)',  # "Video: Title"
            r'Watch:\s*(.+)',  # "Watch: Title"
        ]
        
        for pattern in video_patterns:
            match = re.search(pattern, video_text, re.IGNORECASE)
            if match:
                video_title = match.group(1).strip()
                break
    
    video_info = {
        "video_mentioned": video_mentioned,
        "video_title": video_title if video_title else "Unknown"
    }

    # --- Local Scoring ---
    local_scores = {}
    local_feedback = {}
    improvement_areas = []

    # Key Terms Scoring
    key_terms_ratio = len(detected_terms) / len(key_terms) if key_terms else 0
    key_terms_score = round_nearest_half(min(5.0, key_terms_ratio * 5.0))
    key_terms_feedback = f"You used {len(detected_terms)} of {len(key_terms)} key terms: {detected_terms_str}."
    if key_terms_score < 2.5:
        improvement_areas.append("incorporating key terms")

    # Reading Reference Scoring
    reading_mentioned_local = reading_info["reading_mentioned"]
    author_mentioned = bool(reading_info["assigned_author"] and reading_info["assigned_author"].lower() 
                          in normalize_text_for_matching(submission_text))
    
    reading_score = 0.0
    if reading_mentioned_local and author_mentioned:
        reading_score = 5.0
    elif reading_mentioned_local or author_mentioned:
        reading_score = 2.5
    
    reading_feedback = ""
    if reading_score == 5.0:
        reading_feedback = f"You effectively referenced the assigned reading by {reading_info['assigned_author']}."
    elif reading_score == 2.5:
        if reading_mentioned_local:
            reading_feedback = "You mentioned the reading but could be more specific about the author or concepts."
        else:
            reading_feedback = f"You mentioned {reading_info['assigned_author']} but could more clearly connect to the reading assignment."
    else:
        improvement_areas.append("referencing the assigned reading")
        reading_feedback = "Your submission would benefit from more explicit references to the assigned reading."

    # Video Reference Scoring
    video_mentioned_local = video_info["video_mentioned"]
    video_mentioned_in_submission = any(word.lower() in normalize_text_for_matching(submission_text) 
                                      for word in ["video", "watch", "view", "clip", "recording", "lecture"])
    
    video_score = 0.0
    if video_mentioned_local and video_mentioned_in_submission:
        video_score = 5.0
    elif video_mentioned_local or video_mentioned_in_submission:
        video_score = 2.5
    
    video_feedback = ""
    if video_score == 5.0:
        video_feedback = "You effectively referenced the assigned video content in your response."
    elif video_score == 2.5:
        if video_mentioned_local:
            video_feedback = "You mentioned video content but could strengthen the connection to specific concepts."
        else:
            video_feedback = "Your submission references video content but could be more explicit about the assignment."
    else:
        improvement_areas.append("referencing the assigned video")
        video_feedback = "Your submission would benefit from more explicit references to the assigned video."

    # 2. API Scoring (if available)
    api_results = {}

    if api_key and validate_api_key(api_key):
        try:
            # Prepare the API request
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            
            # Create the grading prompt based on the grading scale
            if grading_scale == "15-point (3 categories)":
                scoring_system = """
                Please score the student's response on a 5-point scale for each category:
                1. Prompt Adherence & Key Terms (5.0 points): Did the student directly address the prompt and use key terms?
                2. Reading Reference (5.0 points): Did the student reference the assigned reading?
                3. Video Reference (5.0 points): Did the student reference the assigned video?
                """
            else:  # 16-point (4 categories)
                scoring_system = """
                Please score the student's response on a 4-point scale for each category:
                1. Prompt Adherence (4.0 points): Did the student directly address the prompt?
                2. Key Terms Usage (4.0 points): Did the student use the key terms appropriately?
                3. Reading Reference (4.0 points): Did the student reference the assigned reading?
                4. Video Reference (4.0 points): Did the student reference the assigned video?
                """

            payload = {
                "model": "anthropic/claude-3-opus",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are an expert instructor grading student discussion posts. {scoring_system} Provide specific, constructive feedback for each category and overall."
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Please grade the following student discussion post:
                        
                        Student Name: {student_first_name}
                        
                        Discussion Prompt: {discussion_prompt}
                        
                        Assigned Reading: {reading_text}
                        
                        Assigned Video: {video_text}
                        
                        Key Terms: {', '.join(key_terms)}
                        
                        Student's Initial Post: {submission_text}
                        
                        Student's Replies: {replies}
                        
                        Please provide your assessment in the following JSON format:
                        {{
                            "prompt_adherence": <score>,
                            "prompt_feedback": "<specific feedback>",
                            "key_terms_score": <score>,
                            "key_terms_feedback": "<specific feedback>",
                            "reading_score": <score>,
                            "reading_feedback": "<specific feedback>",
                            "video_score": <score>,
                            "video_feedback": "<specific feedback>",
                            "general_feedback": "<overall feedback>"
                        }}
                        """
                    }
                ]
            }
            
            # Make the API call with retries
            response_text = robust_api_call(api_url, {}, payload, api_key=api_key)
            api_results = robust_json_parsing(response_text)
            
            # Clean the API results
            api_results = recursively_clean(api_results)
            
        except ValueError as e:
            # Handle specific ValueError exceptions (like authentication errors)
            st.error(f"API Error: {str(e)}")
            api_results = {}
        except Exception as e:
            # Handle other exceptions without exposing the full error message
            st.error("An unexpected error occurred while calling the API. Please try again later.")
            api_results = {}
    else:
        if not api_key:
            st.warning("No API key provided. Using local scoring only.")
        else:
            st.warning("Invalid API key format. Using local scoring only.")

    # 3. Combine Local and API Scores
    if api_results:
        # Use API scores if available
        if grading_scale == "15-point (3 categories)":
            prompt_key_score = (api_results.get("prompt_adherence", 0) + api_results.get("key_terms_score", 0)) / 2
            reading_score = api_results.get("reading_score", 0)
            video_score = api_results.get("video_score", 0)
        else:  # 16-point (4 categories)
            prompt_score = api_results.get("prompt_adherence", 0)
            key_terms_score = api_results.get("key_terms_score", 0)
            reading_score = api_results.get("reading_score", 0)
            video_score = api_results.get("video_score", 0)
    else:
        # Use local scores
        if grading_scale == "15-point (3 categories)":
            prompt_key_score = (5.0 + key_terms_score) / 2  # Assuming perfect prompt adherence if no API
            reading_score = reading_score
            video_score = video_score
        else:  # 16-point (4 categories)
            prompt_score = 4.0  # Assuming perfect prompt adherence if no API
            key_terms_score = key_terms_score
            reading_score = reading_score
            video_score = video_score

    # Calculate total score
    if grading_scale == "15-point (3 categories)":
        total_score = prompt_key_score + reading_score + video_score
        local_scores = {
            "prompt_key_score": prompt_key_score,
            "reading_score": reading_score,
            "video_score": video_score
        }
    else:  # 16-point (4 categories)
        total_score = prompt_score + key_terms_score + reading_score + video_score
        local_scores = {
            "prompt_score": prompt_score,
            "key_terms_score": key_terms_score,
            "reading_score": reading_score,
            "video_score": video_score
        }

    # Prepare feedback
    local_feedback = {
        "reading_feedback": reading_feedback,
        "key_terms_fallback": key_terms_feedback
    }

    # Generate final feedback
    final_feedback = construct_final_feedback(
        api_results,
        local_scores,
        local_feedback,
        improvement_areas,
        student_first_name,
        grading_scale
    )

    # Return the graded submission
    return {
        "total_score": total_score,
        "final_feedback": final_feedback,
        "improvement_areas": improvement_areas,
        "detected_terms": detected_terms
    }

# ============================================
# STREAMLIT APP UI
# ============================================

# Set up the Streamlit page
st.set_page_config(
    page_title="Discussion Grading Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with black text
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .info-box h3 {
        color: black !important;
    }
    .scale-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .step-button {
        color: black !important;
    }
    .download-button {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📚 Automated Discussion Grading Tool</h1>', unsafe_allow_html=True)
st.markdown("Upload your CSV and DOCX files to grade student discussions.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Try to get API key from Streamlit secrets first, then from environment variable, then from user input
api_key = None
api_key_source = "Not set"

# Try Streamlit secrets
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
    api_key_source = "Streamlit secrets"
    # Clean the API key to remove any extra quotes or whitespace
    api_key = api_key.strip().strip('"').strip("'")
    st.sidebar.success("API key loaded from Streamlit secrets")
except (KeyError, FileNotFoundError):
    pass

# Try environment variable if not found in secrets
if not api_key:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        api_key_source = "Environment variable"
        # Clean the API key to remove any extra quotes or whitespace
        api_key = api_key.strip().strip('"').strip("'")
        st.sidebar.success("API key loaded from environment variable")

# Fallback to user input
if not api_key:
    api_key_input = st.sidebar.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key")
    if api_key_input:
        api_key = api_key_input.strip()
        api_key_source = "User input"
else:
    # If we already have an API key, show a masked input
    st.sidebar.text_input("OpenRouter API Key", type="password", value="•••••••••••••••••••••••••••••••••", disabled=True)

# Display API key status
if api_key:
    st.sidebar.info(f"API key source: {api_key_source}")
    
    # Validate the API key format
    if api_key.startswith("sk-or-v1-") and len(api_key) > 20:
        st.sidebar.success("API key format appears valid")
    else:
        st.sidebar.error("Invalid API key format. OpenRouter API keys should start with 'sk-or-v1-' and be at least 20 characters long.")
else:
    st.sidebar.error("No API key provided. Please enter your API key to continue.")

st.sidebar.markdown("Get your API key from [OpenRouter](https://openrouter.ai/)")

# Add grading scale selector
grading_scale = st.sidebar.selectbox(
    "Select Grading Scale",
    ["15-point (3 categories)", "16-point (4 categories)"],
    index=0,
    help="Choose between the 15-point scale (3 categories) or the 16-point scale (4 categories)"
)

# Display information about the selected grading scale
if grading_scale == "15-point (3 categories)":
    st.sidebar.markdown("""
    <div class="scale-info">
    <h4>15-Point Scale (3 Categories)</h4>
    <ul>
        <li>Prompt Adherence & Key Terms (5.0 points)</li>
        <li>Reading Reference (5.0 points)</li>
        <li>Video Reference (5.0 points)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class="scale-info">
    <h4>16-Point Scale (4 Categories)</h4>
    <ul>
        <li>Prompt Adherence (4.0 points)</li>
        <li>Key Terms Usage (4.0 points)</li>
        <li>Reading Reference (4.0 points)</li>
        <li>Video Reference (4.0 points)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="info-box"><h3 class="step-button">Step 1: Upload Files</h3></div>', unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload CSV file with student submissions", type=['csv'], key="csv_uploader")
    st.markdown("The CSV should contain columns for student names, initial posts, and replies.")
    
    # Add debugging information
    if csv_file is not None:
        st.success(f"CSV file uploaded: {csv_file.name}")
    else:
        st.info("Please upload a CSV file to continue.")

with col2:
    st.markdown('<div class="info-box"><h3 class="step-button">Step 2: Upload Lesson Plan</h3></div>', unsafe_allow_html=True)
    docx_file = st.file_uploader("Upload DOCX lesson plan", type=['docx'], key="docx_uploader")
    st.markdown("The lesson plan should contain discussion prompts, reading assignments, and key terms.")
    
    # Add debugging information
    if docx_file is not None:
        st.success(f"DOCX file uploaded: {docx_file.name}")
    else:
        st.info("Please upload a DOCX file to continue.")

# Process files when both are uploaded
if csv_file and docx_file:
    st.markdown("---")
    st.markdown('<div class="info-box"><h3 class="step-button">Step 3: Process Files</h3></div>', unsafe_allow_html=True)
    
    if st.button("🚀 Process Files", type="primary"):
        if not api_key:
            st.error("Please enter your OpenRouter API key in the sidebar.")
        elif not validate_api_key(api_key):
            st.error("Invalid API key format. OpenRouter API keys should start with 'sk-or-v1-' and be at least 20 characters long.")
        else:
            with st.spinner("Processing files... This may take a few minutes."):
                try:
                    # Reset file pointers to the beginning
                    csv_file.seek(0)
                    docx_file.seek(0)
                    
                    # Read files
                    csv_content = csv_file.read()
                    docx_content = docx_file.read()
                    
                    # Add a progress bar
                    progress_bar = st.progress(0)
                    st.text("Parsing lesson plan...")
                    
                    # Call your existing functions
                    discussion_prompt, reading_text, video_text, key_terms, objective, key_concepts = parse_lesson_plan_comprehensive(docx_content)
                    
                    progress_bar.progress(0.25)
                    st.text("Reading CSV file...")
                    
                    # Process CSV
                    try:
                        csv_io = io.StringIO(csv_content.decode('utf-8'))
                        rows = list(csv.DictReader(csv_io))
                    except UnicodeDecodeError:
                        st.error("Error decoding CSV file. Please ensure it's saved in UTF-8 format.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        st.stop()
                    
                    # Check if we got any rows
                    if not rows:
                        st.error("No data found in CSV file. Please check the file format.")
                        st.stop()
                    
                    # Display extracted information
                    progress_bar.progress(0.5)
                    st.text("Analyzing submissions...")
                    
                    # Create a dataframe to store results
                    results = []
                    
                    # Process each student submission
                    total_students = len(rows)
                    for i, row in enumerate(rows):
                        student_name = row.get('Name', row.get('Student Name', row.get('Username', 'Unknown')))
                        initial_post = row.get('Initial Post', row.get('Initial Posts', ''))
                        replies = [row.get(f'Reply {j}', '') for j in range(1, 4) if row.get(f'Reply {j}', '')]
                        
                        # Extract first name for personalization
                        student_first_name = student_name.split()[0] if student_name else "Student"
                        
                        # Grade the submission
                        grade_result = grade_submission_with_retries(
                            initial_post,
                            reading_text,
                            key_terms,
                            discussion_prompt,
                            student_first_name,
                            video_text,
                            replies,
                            api_key,
                            grading_scale
                        )
                        
                        # Add to results
                        results.append({
                            'Name': student_name,
                            'Total Score': grade_result['total_score'],
                            'Feedback': grade_result['final_feedback'],
                            'Improvement Areas': ', '.join(grade_result['improvement_areas']) if grade_result['improvement_areas'] else 'None',
                            'Key Terms Used': ', '.join(grade_result['detected_terms']) if grade_result['detected_terms'] else 'None'
                        })
                        
                        # Update progress
                        progress = (i + 1) / total_students
                        progress_bar.progress(0.5 + progress * 0.5)
                    
                    # Create dataframe from results
                    df = pd.DataFrame(results)
                    
                    # Display results
                    st.success("Processing complete!")
                    st.markdown('<div class="info-box"><h3>Grading Results</h3></div>', unsafe_allow_html=True)
                    
                    # Show statistics
                    st.markdown("### 📊 Grade Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_score = df['Total Score'].mean()
                        st.metric("Average Score", f"{avg_score:.2f}")
                    
                    with col2:
                        median_score = df['Total Score'].median()
                        st.metric("Median Score", f"{median_score:.2f}")
                    
                    with col3:
                        min_score = df['Total Score'].min()
                        max_score = df['Total Score'].max()
                        st.metric("Score Range", f"{min_score:.1f} - {max_score:.1f}")
                    
                    # Show score distribution
                    st.markdown("### 📈 Score Distribution")
                    fig = px.histogram(df, x="Total Score", nbins=20, title="Distribution of Scores")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed results
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(df)
                    
                    # Download button
                    csv_output = df.to_csv(index=False)
                    # Get the original filename and append _GRADED.csv
                    original_filename = csv_file.name
                    base_filename = os.path.splitext(original_filename)[0]  # Remove .csv extension
                    graded_filename = f"{base_filename}_GRADED.csv"

                    st.markdown('<div class="download-button">', unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Graded CSV",
                        data=csv_output,
                        file_name=graded_filename,
                        mime="text/csv"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
