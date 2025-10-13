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
        if not api_key:
            raise ValueError("No API key provided")
        
        if not api_key.startswith("sk-or-v1-"):
            raise ValueError(f"Invalid API key format. Expected format: sk-or-v1-...")
        
        new_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app-url.com",
            "X-Title": "Discussion Grading Tool"
        }
        
        response = requests.post(api_url, headers=new_headers, json=payload, timeout=timeout)
        
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

        response_text_clean = re.sub(r'^```json\s*', '', response_text_clean, flags=re.DOTALL)
        response_text_clean = re.sub(r'\s*```$', '', response_text_clean, flags=re.DOTALL)
        response_text_clean = re.sub(r'^```\s*', '', response_text_clean, flags=re.DOTALL)

        try:
            return json.loads(response_text_clean)
        except json.JSONDecodeError as e:
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
            is_stop_indicator = False
            for indicator in stop_indicators:
                if re.match(r'^\s*' + re.escape(indicator) + r'[:\s]*', line_clean, re.IGNORECASE) and len(line_clean.split()) < 8:
                    is_stop_indicator = True
                    break

            if is_stop_indicator and line_clean:
                in_section = False
                break

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

    full_text = "\n".join([p.text.strip() for p in document.paragraphs if p.text.strip()])
    full_text_clean = fix_encoding(full_text)

    sections_to_extract = {
        "Discussion Prompt": (["Discussion", "Prompt", "Discussion Question", "Question"], ['Reading', 'Video', 'Key Terms', 'Objective', 'Key Concepts', 'Materials']),
        "Reading": (["Assigned Reading", "Required Reading", "READING"], ['Discussion', 'Video', 'Key Terms', 'Objective', 'Key Concepts', 'Materials', 'Prompt']),
        "Video": (["VIDEO", "Assigned Video", "Required Video"], ['Discussion', 'Reading', 'Key Terms', 'Objective', 'Key Concepts', 'Materials', 'Prompt']),
        "Objective": (["Learning Objective", "Goals"], ['Discussion', 'Reading', 'Video', 'Key Terms', 'Objective', 'Materials', 'Prompt']),
        "Key Concepts": (["Concepts", "Main Concepts", "KEY CONCEPTS"], ['Discussion', 'Reading', 'Video', 'Key Terms', 'Objective', 'Materials', 'Prompt']),
        "Key Terms": (["KEY TERMS", "Terms", "Vocabulary"], ['Discussion', 'Reading', 'Video', 'Objective', 'Key Concepts', 'Materials', 'Prompt'])
    }

    parsed_sections = {}
    for name, (alts, stops) in sections_to_extract.items():
        parsed_sections[name] = extract_section_robust(full_text_clean, name, alts, stops)

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

    discussion_prompt = parsed_sections["Discussion Prompt"]
    if not discussion_prompt.strip():
        if parsed_sections["Objective"].strip():
            discussion_prompt = parsed_sections["Objective"]
        else:
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

def analyze_engagement_quality(replies: List[str]) -> Dict[str, Any]:
    replies = [fix_encoding(reply) for reply in replies]
    valid_replies = [reply for reply in replies if reply and len(reply.split()) >= 10]
    num_replies = len(valid_replies)

    if num_replies == 0:
        return {
            'score': 2.0,
            'feedback': 'We encourage you to participate in peer discussion! Substantive replies engage with at least one classmate.',
            'num_replies': 0,
            'highest_quality_score': 2.0
        }

    reply_quality_scores = []

    for reply in valid_replies:
        reply_lower = reply.lower()
        word_count = len(reply.split())
        quality_score = 3.0

        deep_engagement_indicators = [
            'i disagree', 'i believe that', 'i believe it', 'i agree that', 'i understand',
            'building on', 'contrary to', 'critique', 'elaborate', 'perspective', 'i appreciated',
            'can you relate', 'in addition to', 'i think', 'surprised by', 'recommend',
            'interesting point', 'you mentioned', 'your point about', 'i would argue',
            'similar to what you said', 'expanding on', 'different perspective', 'your analysis',
            'i noticed', 'as you pointed out', 'building upon', 'along those lines',
            'your observation', 'another way to think', 'i wonder if', 'what if we consider'
        ]

        supporting_indicators = [
            'because', 'however', 'although', 'critical', 'analysis', 'blueprint',
            'therefore', 'moreover', 'furthermore', 'in contrast', 'similarly',
            'for example', 'for instance', 'this suggests', 'this demonstrates',
            'evidence shows', 'research indicates', 'as shown by', 'considering that'
        ]

        has_deep_engagement = any(term in reply_lower for term in deep_engagement_indicators)
        has_supporting_args = any(term in reply_lower for term in supporting_indicators)
        has_questions = '?' in reply
        has_specific_details = bool(re.search(r'\b\d+\b', reply)) or bool(re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:argued|stated|mentioned|noted|wrote|said)', reply))

        if word_count >= 75 and (has_deep_engagement or has_supporting_args or has_specific_details):
            quality_score = 4.0
        elif word_count >= 60 and (has_deep_engagement or has_supporting_args):
            quality_score = 4.0
        elif word_count >= 50 and has_deep_engagement and has_supporting_args:
            quality_score = 4.0
        elif word_count >= 50 and (has_deep_engagement or has_supporting_args or has_questions):
            quality_score = 3.5
        elif word_count >= 40 and (has_deep_engagement or has_supporting_args):
            quality_score = 3.5
        elif word_count >= 25:
            quality_score = 3.0
        else:
            quality_score = 3.0

        quality_score = min(4.0, quality_score)
        quality_score = round_nearest_half(quality_score)
        reply_quality_scores.append(quality_score)

    highest_quality_score = max(reply_quality_scores)

    recipient_name = "a peer"
    if valid_replies:
        highest_quality_index = reply_quality_scores.index(highest_quality_score)
        highest_quality_reply = valid_replies[highest_quality_index]

        name_match = re.search(r'^(?:Hello|Hi|Dear|Hey|To)\s+([A-Za-z]+)\s*([A-Za-z]*),?', highest_quality_reply.strip(), re.IGNORECASE)

        if name_match:
            recipient_name = name_match.group(1).strip()
        else:
            name_match_only = re.search(r'^([A-Za-z]+),', highest_quality_reply.strip())
            if name_match_only:
                    recipient_name = name_match_only.group(1).strip()

    if highest_quality_score >= 4.0:
        feedback = f"Excellent engagement! Your reply to {recipient_name} demonstrates substantive interaction with their ideas, showing depth of analysis and critical thinking that meets the highest standards for peer discussion."
    elif highest_quality_score >= 3.5:
        feedback = f"Strong engagement. Your meaningful reply to {recipient_name} shows good interaction with their ideas. Your response demonstrates solid understanding and contributes meaningfully to the discussion. To earn full credit (4.0), consider adding more detailed analysis, specific examples, or deeper critical engagement with your peers' arguments."
    elif highest_quality_score >= 3.0:
        feedback = f"Adequate engagement. Your contribution meets the substantive length requirement. To strengthen future replies, consider adding more detailed analysis, specific examples, or deeper critical engagement with your peers' arguments. Points were deducted because your response lacked the depth of analysis expected for higher credit."
    else:
        feedback = 'Your participation meets the minimum requirement, but your replies lack substantive length or meaningful interaction. Focus on directly responding to and debating your peers\' ideas in detail. Points were deducted because your response was too brief or did not engage substantively with your peers\' ideas.'

    return {
        'score': highest_quality_score,
        'feedback': feedback,
        'num_replies': num_replies,
        'highest_quality_score': highest_quality_score
    }

def analyze_video_reference_locally(submission_text: str, video_text: str) -> Dict[str, Any]:
    """Analyzes the submission for references to the video material using local rules."""
    score = 2.0
    feedback = "You successfully integrated concepts from the video, but you did not provide a specific citation as required for higher credit. You must include the author/creator and a timestamp to earn more than the minimum score."

    # Extract creator's name from video_text (assuming it's a prominent capitalized name)
    creator_name = ""
    # Regex to find a capitalized name, likely the creator
    creator_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', video_text)
    if creator_match:
        creator_name = creator_match.group(1).strip()

    # Define regex patterns
    # Pattern for creator and timestamp, e.g., (Darius, 0:12) or Darius (0:12)
    creator_ts_pattern = re.compile(r'\(\s*' + re.escape(creator_name.split()[0]) + r'\s*,\s*\d{1,2}:\d{2}\s*\)|' + re.escape(creator_name) + r'\s*\(\s*\d{1,2}:\d{2}\s*\)', re.IGNORECASE) if creator_name else None
    
    # General timestamp pattern, e.g., (1:02) or at 1:02
    timestamp_pattern = re.compile(r'\b\d{1,2}:\d{2}\b')

    # Creator name pattern
    creator_pattern = re.compile(r'\b' + re.escape(creator_name) + r'\b', re.IGNORECASE) if creator_name else None

    # Generic video keywords
    video_keyword_pattern = re.compile(r'\b(video|youtube|film)\b', re.IGNORECASE)

    # Check for patterns in submission text
    has_creator_and_ts = creator_ts_pattern and creator_ts_pattern.search(submission_text)
    has_timestamp = timestamp_pattern.search(submission_text)
    has_creator = creator_pattern and creator_pattern.search(submission_text)
    has_video_keyword = video_keyword_pattern.search(submission_text)

    # Apply scoring logic
    if has_creator_and_ts:
        score = 4.0
        feedback = f"Excellent! You specifically referenced '{creator_name}' and included timestamps, demonstrating strong engagement with the video content. Full credit awarded."
    elif has_creator and has_timestamp: # Covers cases where they are not in the same parenthesis
        score = 4.0
        feedback = f"Excellent! You specifically referenced '{creator_name}' and included timestamps, demonstrating strong engagement with the video content. Full credit awarded."
    elif has_creator:
        score = 3.5
        feedback = f"Good reference. You mentioned the video's creator, '{creator_name}'. To earn full credit, also include specific timestamps to support your points."
    elif has_timestamp:
        score = 3.0
        feedback = "You included a timestamp, which is great for specificity. To improve, also mention the creator's name to make the citation complete."
    elif has_video_keyword:
        score = 2.5
        feedback = "You mentioned the 'video', but to earn higher credit, you need to provide more specific references, such as the creator's name or timestamps from the video."

    return {'score': score, 'feedback': feedback}


def construct_final_feedback(
    llm_results: Dict,
    local_scores: Dict[str, float],
    local_feedback: Dict[str, str],
    improvement_areas: List[str],
    student_first_name: str,
    grading_scale: str
) -> str:
    combined_prompt_key_score = local_scores['prompt_score'] + local_scores['key_terms_score']
    prompt_feedback = llm_results.get('prompt_feedback', 'Feedback missing for prompt quality.')
    key_terms_feedback = llm_results.get('key_terms_feedback', local_feedback.get('key_terms_fallback', 'Feedback missing for key terms.'))
    reading_feedback = local_feedback.get('reading_feedback', llm_results.get('reading_feedback', 'Feedback missing for reading reference.'))
    video_feedback = local_feedback.get('video_feedback', 'Feedback missing for video reference.') # Prioritize local feedback
    general_feedback_llm = llm_results.get('general_feedback', 'Overall submission quality was strong.')

    engagement_feedback = local_feedback['engagement_feedback']

    def transform_to_second_person(text):
        if not text: return ""
        text = re.sub(r'\byou\'s\b', 'you are', text, flags=re.IGNORECASE)
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

    prompt_key_combined_feedback = f"{prompt_feedback.strip()} {key_terms_feedback.strip()}"
    
    if grading_scale == "15-point (3 categories)":
        scaled_prompt_score = round_nearest_half(local_scores['prompt_score'] * 1.25)
        scaled_key_terms_score = round_nearest_half(local_scores['key_terms_score'] * 1.25)
        scaled_video_score = round_nearest_half(local_scores['video_score'] * 1.25)
        scaled_reading_score = round_nearest_half(local_scores['reading_score'] * 1.25)
        
        prompt_key_formatted = f"PROMPT AND KEY TERMS [{scaled_prompt_score + scaled_key_terms_score:.1f}/5.0]: {prompt_key_combined_feedback}"
        video_formatted = f"REFERENCE TO VIDEO [{scaled_video_score:.1f}/5.0]: {video_feedback}"
        reading_formatted = f"REFERENCE TO READING [{scaled_reading_score:.1f}/5.0]: {reading_feedback}"
    else:  # 16-point (4 categories)
        prompt_key_formatted = f"PROMPT AND KEY TERMS [{combined_prompt_key_score:.1f}/4.0]: {prompt_key_combined_feedback}"
        video_formatted = f"REFERENCE TO VIDEO [{local_scores['video_score']:.1f}/4.0]: {video_feedback}"
        reading_formatted = f"REFERENCE TO READING [{local_scores['reading_score']:.1f}/4.0]: {reading_feedback}"
        engagement_formatted = f"DISCUSSION ENGAGEMENT [{local_scores['engagement_score']:.1f}/4.0]: {engagement_feedback}"

    if improvement_areas:
        if grading_scale == "15-point (3 categories)" and "Discussion Engagement" in improvement_areas:
            improvement_areas = [area for area in improvement_areas if area != "Discussion Engagement"]
            
        if improvement_areas:
            improvement_focus = f"{student_first_name}, while your work demonstrates strong engagement with the content, focus on improving in the area(s) of: {', '.join(improvement_areas)} to maximize your synthesis of the concepts. {general_feedback_llm}"
        else:
            improvement_focus = f"Outstanding work {student_first_name}! You excelled on all sections of this assignment. {general_feedback_llm}"
    else:
        improvement_focus = f"Outstanding work {student_first_name}! You excelled on all sections of this assignment. {general_feedback_llm}"

    general_formatted = f"GENERAL FEEDBACK: {improvement_focus}"

    if grading_scale == "15-point (3 categories)":
        final_feedback = '\n'.join([
            prompt_key_formatted,
            video_formatted,
            reading_formatted,
            general_formatted
        ])
    else:  # 16-point (4 categories)
        final_feedback = '\n'.join([
            prompt_key_formatted,
            video_formatted,
            reading_formatted,
            engagement_formatted,
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
    engagement_analysis = analyze_engagement_quality(replies)
    video_analysis = analyze_video_reference_locally(submission_text, video_text) # NEW: Local video check
    engagement_score = engagement_analysis['score']
    detected_terms = detect_key_terms_presence(submission_text, key_terms)
    detected_terms_str = ', '.join(detected_terms) if detected_terms else 'none detected'
    reading_info = {}

    if "READING:" in reading_text:
        reading_line = ""
        for line in reading_text.split('\n'):
            if "READING:" in line:
                reading_line = line.strip()
                break
        author_match = re.search(r'READING:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', reading_line)
        assigned_author = author_match.group(1).strip() if author_match else ""
    else:
        author_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', reading_text.strip())
        assigned_author = author_match.group(1).strip() if author_match else ""

    page_numbers = []
    
    patterns = [
        r'pages?\s+([\d.,\s-]+)',
        r'p\.?\s*([\d.,\s-]+)',
        r'([\d.,\s-]+)'
    ]

    for pattern in patterns:
        pages_match = re.search(pattern, reading_text, re.IGNORECASE)
        if pages_match:
            page_str = pages_match.group(1)
            page_numbers = re.findall(r'[\d.]+', page_str)
            page_numbers = [float(p) for p in page_numbers]
            break

    if assigned_author:
        reading_info['author_last_name'] = assigned_author
    else:
        reading_info['author_last_name'] = ""

    if page_numbers:
        page_numbers.sort()
        if len(page_numbers) == 1:
            page_num = page_numbers[0]
            if page_num.is_integer():
                reading_info['page_range_expected'] = f"page {int(page_num)}"
            else:
                reading_info['page_range_expected'] = f"page {page_num}"
        elif len(page_numbers) == 2:
            if page_numbers[0].is_integer() and page_numbers[1].is_integer():
                reading_info['page_range_expected'] = f"pages {int(page_numbers[0])}-{int(page_numbers[1])}"
            else:
                reading_info['page_range_expected'] = f"pages {page_numbers[0]}-{page_numbers[1]}"
        else:
            formatted_pages = []
            for p in page_numbers:
                if p.is_integer():
                    formatted_pages.append(str(int(p)))
                else:
                    formatted_pages.append(str(p))
            reading_info['page_range_expected'] = f"pages {', '.join(formatted_pages)}"
    else:
        reading_info['page_range_expected'] = "unspecified pages"

    highest_max_reading_score = 2.0
    best_citation_status_msg = f"NO CLEAR REFERENCE TO THE ASSIGNED READING WAS DETECTED. The minimum score of **2.0** applies."
    detected_author = ""

    if reading_info['author_last_name']:
        assigned_author_lower = reading_info['author_last_name'].lower()
        author_present = re.search(r'\b' + re.escape(assigned_author_lower) + r'\b', submission_text.lower())
        page_present = False
        detected_pages = []
        if page_numbers:
            if len(page_numbers) == 2:
                start_page = int(min(page_numbers))
                end_page = int(max(page_numbers))
                cited_pages = re.findall(r'(?:p|pg|page)s?\.?\s*(\d+)', submission_text, re.IGNORECASE)
                for page_str in cited_pages:
                    cited_num = int(page_str)
                    if start_page <= cited_num <= end_page:
                        page_present = True
                        detected_pages.append(page_str)
            else:
                for page in page_numbers:
                    page_str = str(int(page)) if isinstance(page, float) and page.is_integer() else str(page)
                    patterns = [
                        r'\b' + re.escape(page_str) + r'\b',
                        r'\bp\.?\s*' + re.escape(page_str) + r'\b',
                        r'\bpage\s*' + re.escape(page_str) + r'\b',
                        r'\bpages?\s*' + re.escape(page_str) + r'\b'
                    ]
                    for pattern in patterns:
                        if re.search(pattern, submission_text, re.IGNORECASE):
                            page_present = True
                            detected_pages.append(page_str)
                            break
                    if page_present and len(page_numbers) > 1:
                        break

        if author_present and page_present:
            highest_max_reading_score = 4.0
            best_citation_status_msg = f"Both the author ('{assigned_author}') and a relevant page number from the assigned reading were detected. Full credit awarded."
        elif author_present:
            highest_max_reading_score = 3.0
            best_citation_status_msg = f"The author ('{assigned_author}') was mentioned, but no specific page number from the assigned reading was detected. Partial credit awarded."
        elif page_present:
            highest_max_reading_score = 3.5
            best_citation_status_msg = f"A page number from the assigned reading was detected, but the author ('{assigned_author}') was not mentioned. Partial credit awarded."

        if not author_present:
            potential_authors = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', submission_text)
            for potential_author in potential_authors:
                if potential_author.lower() in ['the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by']:
                    continue
                if len(potential_author) > 3:
                    detected_author = potential_author
                    if page_present:
                        highest_max_reading_score = 3.5
                        best_citation_status_msg = f"A reference to '{potential_author}' with page number {', '.join(detected_pages)} was detected, but this does not match the assigned author ('{assigned_author}'). Partial credit awarded for the correct page reference."
                    else:
                        highest_max_reading_score = 2.5
                        best_citation_status_msg = f"A reference to '{potential_author}' was detected, but this does not match the assigned author ('{assigned_author}'). Partial credit awarded."
                    break
    else:
        highest_max_reading_score = 2.0
        best_citation_status_msg = f"The assigned reading information did not specify an author. The minimum score of **2.0** applies."

    max_reading_score = highest_max_reading_score
    citation_status_msg = best_citation_status_msg

    if max_reading_score == 4.0:
        reading_feedback_local = f"You successfully integrated concepts from the reading and provided a specific citation with a page number from {assigned_author}'s text ({reading_info['page_range_expected']}), earning full credit for this section"
    elif max_reading_score == 3.5:
        if detected_author:
            reading_feedback_local = f"You referenced page number(s) from the assigned reading, but cited the wrong author. Be sure to include the correct author to earn full credit"
        else:
            reading_feedback_local = f"A page number from the assigned reading was detected, but the author was not mentioned. Include both the author and page number for full credit"
    elif max_reading_score == 3.0:
        reading_feedback_local = f"You mentioned the author ({assigned_author}), demonstrating engagement with the reading. However, you did not provide a specific page number from the assigned reading ({reading_info['page_range_expected']}) as required for higher credit. Include specific page references to earn full credit"
    elif max_reading_score == 2.5:
        reading_feedback_local = f"You referenced the wrong author in your submission. Be sure to include the correct author and include a page number to earn more credit"
    else:
        if assigned_author:
            reading_feedback_local = f"You successfully integrated concepts from the reading, but you did not provide a specific citation with a page number from {assigned_author}'s text ({reading_info['page_range_expected']}) as required for higher credit. You must include the author and a page number from the assigned reading to earn more than the minimum score"
        else:
            reading_feedback_local = f"You successfully integrated concepts from the reading, but you did not provide a specific citation with a page number as required for higher credit. You must include the author and a page number from the assigned reading to earn more than the minimum score"

    local_scores = {
        'engagement_score': engagement_score,
        'prompt_score': 0.0,
        'reading_score': max_reading_score,
        'key_terms_score': 0.0,
        'video_score': video_analysis['score'] # SET FROM PYTHON DETECTION
    }
    local_feedback = {
        'engagement_feedback': engagement_analysis['feedback'],
        'reading_feedback': reading_feedback_local,
        'video_feedback': video_analysis['feedback'], # SET FROM PYTHON DETECTION
        'key_terms_fallback': f"LLM failed to provide key terms feedback. Detected terms: {detected_terms_str}"
    }

    llm_scoring_criteria = f"""
SCORING Guidelines for LLM (4 points total - Reading & Video are scored separately):
1. PROMPT ADHERENCE (Minimum 1.0 - 2.0): How well does the student address the entire prompt? (2.0 Maximum)
2. READING REFERENCE: **This section is scored separately by the system as {max_reading_score:.1f}. Do not provide a reading_score.**
   - Citation Status (for context): {citation_status_msg}
3. VIDEO REFERENCE: **This section is scored separately by the system as {video_analysis['score']:.1f}. Do not provide a video_score.**
4. KEY TERMS USAGE (Minimum 1.0 - 2.0): Did the student use at least one key term (from the detected list) in a way that demonstrates contextual understanding? (2.0 Maximum)
   - FULL CREDIT (2.0) MUST BE AWARDED if ONE or more terms are used meaningfully.

Detected Key Terms to review for usage: "{detected_terms_str}"
"""

    prompt_for_llm = f"""Grade this student discussion submission based ONLY on the following criteria. Reading Reference ({max_reading_score:.1f}), Video Reference ({video_analysis['score']:.1f}), and Engagement ({engagement_score}) are scored separately by the system.

STUDENT: {student_first_name}

ASSIGNMENT CONTEXT:
Prompt: {discussion_prompt[:300]}...
Reading: {reading_text[:200]}...
Video: {video_text[:200]}...

{llm_scoring_criteria}

IMPORTANT: Provide SPECIFIC and ENCOURAGING feedback in the second person ("You", "Your").
**DO NOT include "reading_score" or "video_score" in your JSON response - they are handled separately.**

Respond with ONLY valid JSON. Omit any markdown fences (```json). Use floating point numbers rounded to the nearest 0.5.

{{
  "prompt_score": "2.0",
  "key_terms_score": "2.0",
  "prompt_feedback": "You successfully articulated how involuntary servitude was preserved and connected this theme to present-day issues.",
  "key_terms_feedback": "Your contextual usage of key terms earns full credit, demonstrating clear understanding of the material.",
  "general_feedback": "Your arguments were well-structured and demonstrated impressive critical thinking."
}}

SUBMISSION TEXT:
{submission_text[:1500]}
"""

    api_results = {}

    if api_key and validate_api_key(api_key):
        try:
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            
            payload = {
                "model": "google/gemini-2.5-flash-preview-09-2025",
                "messages": [{"role": "user", "content": prompt_for_llm}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "max_tokens_for_reasoning": 512
            }
            
            response_text = robust_api_call(api_url, {}, payload, api_key=api_key)
            api_results = robust_json_parsing(response_text)
            
            api_results = recursively_clean(api_results)
            
        except ValueError as e:
            st.error(f"API Error: {str(e)}")
            api_results = {}
        except Exception as e:
            st.error("An unexpected error occurred while calling the API. Please try again later.")
            api_results = {}
    else:
        if not api_key:
            st.warning("No API key provided. Using local scoring only.")
        else:
            st.warning("Invalid API key format. Using local scoring only.")

    try:
        local_scores['prompt_score'] = round_nearest_half(max(1.0, min(2.0, float(api_results.get("prompt_score", 1.0)))))
        local_scores['key_terms_score'] = round_nearest_half(max(1.0, min(2.0, float(api_results.get("key_terms_score", 1.0)))))
    except (ValueError, TypeError):
        local_scores['prompt_score'] = 1.0
        local_scores['key_terms_score'] = 1.0

    if grading_scale == "15-point (3 categories)":
        weighted_scores = {
            "Prompt Adherence": local_scores['prompt_score'] / 2.0,
            "Key Terms Usage": local_scores['key_terms_score'] / 2.0,
            "Reading Reference": local_scores['reading_score'] / 4.0,
            "Video Reference": local_scores['video_score'] / 4.0
        }
    else:
        weighted_scores = {
            "Prompt Adherence and Key Terms": (local_scores['prompt_score'] + local_scores['key_terms_score']) / 4.0,
            "Reading Reference": local_scores['reading_score'] / 4.0,
            "Video Reference": local_scores['video_score'] / 4.0,
            "Discussion Engagement": local_scores['engagement_score'] / 4.0
        }
    
    sorted_improvement = sorted(weighted_scores.items(), key=lambda item: item[1])
    improvement_areas = [name for name, score in sorted_improvement if score < 1.0 and score > 0.0]

    total = sum(local_scores.values())
    total_score = round_nearest_half(total)

    if grading_scale == "15-point (3 categories)":
        scaled_prompt_score = round_nearest_half(local_scores['prompt_score'] * 1.25)
        scaled_key_terms_score = round_nearest_half(local_scores['key_terms_score'] * 1.25)
        scaled_video_score = round_nearest_half(local_scores['video_score'] * 1.25)
        scaled_reading_score = round_nearest_half(local_scores['reading_score'] * 1.25)
        scaled_total = round_nearest_half(scaled_prompt_score + scaled_key_terms_score + scaled_video_score + scaled_reading_score)
        
        final_grades = {
            "prompt_score": scaled_prompt_score,
            "key_terms_score": scaled_key_terms_score,
            "video_score": scaled_video_score,
            "reading_score": scaled_reading_score,
            "engagement_score": 0,
            "total_score": scaled_total,
        }
    else:
        final_grades = {
            "prompt_score": local_scores['prompt_score'],
            "key_terms_score": local_scores['key_terms_score'],
            "video_score": local_scores['video_score'],
            "reading_score": local_scores['reading_score'],
            "engagement_score": local_scores['engagement_score'],
            "total_score": total_score,
        }

    final_grades["feedback"] = construct_final_feedback(api_results, local_scores, local_feedback, improvement_areas, student_first_name, grading_scale)

    return final_grades

# ============================================
# STREAMLIT APP UI
# ============================================

st.set_page_config(
    page_title="Discussion Grading Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        color: black !important;
    }
    .scale-info h4 {
        color: black !important;
    }
    .scale-info ul {
        color: black !important;
    }
    .step-button {
        color: black !important;
    }
    .download-button {
        color: black !important;
    }
    /* Fix for the dropdown menu text color - ALL WHITE */
    .stSelectbox > div > div > div {
        color: white !important;
    }
    .stSelectbox > div > div > div > div {
        color: white !important;
    }
    .stSelectbox label {
        color: white !important;
    }
    .stSelectbox > label > div {
        color: white !important;
    }
    .stSelectbox > label > div > div {
        color: white !important;
    }
    .stSelectbox > label > div > div > div {
        color: white !important;
    }
    .stSelectbox > label > div > div > div > div {
        color: white !important;
    }
    .stSidebar .stSelectbox {
        color: white !important;
    }
    .stSidebar .stSelectbox > div > div > div {
        color: white !important;
    }
    .stSidebar .stSelectbox > div > div > div > div {
        color: white !important;
    }
    .stSidebar .stSelectbox label {
        color: white !important;
    }
    .stSidebar .stSelectbox > label > div {
        color: white !important;
    }
    .stSidebar .stSelectbox > label > div > div {
        color: white !important;
    }
    .stSidebar .stSelectbox > label > div > div > div {
        color: white !important;
    }
    .stSidebar .stSelectbox > label > div > div > div > div {
        color: white !important;
    }
    /* Additional CSS to ensure dropdown options are visible */
    div[data-baseweb="select"] {
        color: white !important;
    }
    div[data-baseweb="select"] > div {
        color: white !important;
    }
    div[data-baseweb="select"] > div > div {
        color: white !important;
    }
    div[data-baseweb="select"] > div > div > div {
        color: white !important;
    }
    div[data-baseweb="select"] > div > div > div > div {
        color: white !important;
    }
    div[data-baseweb="select"] div[role="listbox"] {
        color: white !important;
    }
    div[data-baseweb="select"] div[role="listbox"] > div {
        color: white !important;
    }
    div[data-baseweb="select"] div[role="listbox"] > div > div {
        color: white !important;
    }
    div[data-baseweb="select"] div[role="listbox"] > div > div > div {
        color: white !important;
    }
    /* Fix for selected text in dropdown - WHITE */
    div[data-baseweb="select"] div[role="listbox"] > div[data-selected="true"] {
        color: white !important;
    }
    /* Fix for hover state in dropdown - WHITE */
    div[data-baseweb="select"] div[role="listbox"] > div:hover {
        color: white !important;
    }
    /* Fix for dropdown arrow */
    svg[data-testid="stSelectboxDropdownIcon"] {
        fill: white !important;
    }
    /* Fix for the selected value display - WHITE */
    div[data-baseweb="select"] > div > div > div > div > div {
        color: white !important;
    }
    /* Additional fix for the selected value - WHITE */
    div[data-baseweb="select"] > div > div > div > div > div > div {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📚 Automated Discussion Grading Tool</h1>', unsafe_allow_html=True)
st.markdown("Upload your CSV and DOCX files to grade student discussions.")

st.sidebar.header("Configuration")

api_key = None
api_key_source = "Not set"

try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
    api_key_source = "Streamlit secrets"
    api_key = api_key.strip().strip('"').strip("'")
    st.sidebar.success("API key loaded from Streamlit secrets")
except (KeyError, FileNotFoundError):
    pass

if not api_key:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        api_key_source = "Environment variable"
        api_key = api_key.strip().strip('"').strip("'")
        st.sidebar.success("API key loaded from environment variable")

if not api_key:
    api_key_input = st.sidebar.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key")
    if api_key_input:
        api_key = api_key_input.strip()
        api_key_source = "User input"
else:
    st.sidebar.text_input("OpenRouter API Key", type="password", value="•••••••••••••••••••••••••••••••••", disabled=True)

if api_key:
    st.sidebar.info(f"API key source: {api_key_source}")
    
    if api_key.startswith("sk-or-v1-") and len(api_key) > 20:
        st.sidebar.success("API key format appears valid")
    else:
        st.sidebar.error("Invalid API key format. OpenRouter API keys should start with 'sk-or-v1-' and be at least 20 characters long.")
else:
    st.sidebar.error("No API key provided. Please enter your API key to continue.")

st.sidebar.markdown("Get your API key from [OpenRouter](https://openrouter.ai/)")

grading_scale = st.sidebar.selectbox(
    "Select Grading Scale",
    ["15-point (3 categories)", "16-point (4 categories)"],
    index=0,
    help="Choose between the 15-point scale (3 categories) or the 16-point scale (4 categories)"
)

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
        <li>Prompt Adherence and Key Terms (4.0 points)</li>
        <li>Reading Reference (4.0 points)</li>
        <li>Video Reference (4.0 points)</li>
        <li>Discussion Engagement (4.0 points)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="info-box"><h3 class="step-button">Step 1: Upload Files</h3></div>', unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload CSV file with student submissions", type=['csv'], key="csv_uploader")
    st.markdown("The CSV should contain columns for student names, initial posts, and replies.")
    
    if csv_file is not None:
        st.success(f"CSV file uploaded: {csv_file.name}")
    else:
        st.info("Please upload a CSV file to continue.")

with col2:
    st.markdown('<div class="info-box"><h3 class="step-button">Step 2: Upload Lesson Plan</h3></div>', unsafe_allow_html=True)
    docx_file = st.file_uploader("Upload DOCX lesson plan", type=['docx'], key="docx_uploader")
    st.markdown("The lesson plan should contain discussion prompts, reading assignments, and key terms.")
    
    if docx_file is not None:
        st.success(f"DOCX file uploaded: {docx_file.name}")
    else:
        st.info("Please upload a DOCX file to continue.")

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
                    csv_file.seek(0)
                    docx_file.seek(0)
                    
                    csv_content = csv_file.read()
                    docx_content = docx_file.read()
                    
                    progress_bar = st.progress(0)
                    st.text("Parsing lesson plan...")
                    
                    discussion_prompt, reading_text, video_text, key_terms, objective, key_concepts = parse_lesson_plan_comprehensive(docx_content)
                    
                    progress_bar.progress(0.25)
                    st.text("Reading CSV file...")
                    
                    try:
                        csv_io = io.StringIO(csv_content.decode('utf-8'))
                        rows = list(csv.DictReader(csv_io))
                    except UnicodeDecodeError:
                        st.error("Error decoding CSV file. Please ensure it's saved in UTF-8 format.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        st.stop()
                    
                    if not rows:
                        st.error("No data found in CSV file. Please check the file format.")
                        st.stop()
                    
                    original_columns = list(rows[0].keys()) if rows else []
                    
                    progress_bar.progress(0.5)
                    st.text("Analyzing submissions...")
                    
                    results = []
                    
                    total_students = len(rows)
                    for i, row in enumerate(rows):
                        student_name = row.get('Name', row.get('Student Name', row.get('Username', 'Unknown')))
                        initial_post = row.get('Initial Post', row.get('Initial Posts', ''))
                        replies = [row.get(f'Reply {j}', '') for j in range(1, 4) if row.get(f'Reply {j}', '')]
                        
                        student_first_name = student_name.split()[0] if student_name else "Student"
                        
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
                        
                        result_row = row.copy()
                        result_row.update(grade_result)
                        results.append(result_row)
                        
                        progress = (i + 1) / total_students
                        progress_bar.progress(0.5 + progress * 0.5)
                    
                    df = pd.DataFrame(results)
                    
                    st.success("Processing complete!")
                    st.markdown('<div class="info-box"><h3>Grading Results</h3></div>', unsafe_allow_html=True)
                    
                    st.markdown("### 📊 Grade Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_score = df['total_score'].mean()
                        st.metric("Average Score", f"{avg_score:.2f}")
                    
                    with col2:
                        median_score = df['total_score'].median()
                        st.metric("Median Score", f"{median_score:.2f}")
                    
                    with col3:
                        min_score = df['total_score'].min()
                        max_score = df['total_score'].max()
                        st.metric("Score Range", f"{min_score:.1f} - {max_score:.1f}")
                    
                    st.markdown("### 📈 Score Distribution")
                    fig = px.histogram(df, x="total_score", nbins=20, title="Distribution of Scores")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(df)
                    
                    csv_output = df.to_csv(index=False)
                    original_filename = csv_file.name
                    base_filename = os.path.splitext(original_filename)[0]
                    graded_filename = f"GRADED_{base_filename}.csv"

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
