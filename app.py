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

from typing import Dict, Any, List, Set, Union

from difflib import SequenceMatcher

import plotly.express as px


# ============================================
# CONSTANTS
# ============================================

BRONZE_COLOR = "#937329"

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
    pattern = re.compile(r"<[^>]*>")
    return re.sub(pattern, "", text)


def fix_encoding(text: Union[str, bytes]) -> str:
    """Fix common encoding issues and replace problematic characters."""
    if not isinstance(text, (str, bytes)):
        return str(text)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    # Common encoding fixes for smart quotes/dashes/html entities
    replacements = [
        ("\u00e2\u0080\u0099", "'"),
        ("\u00e2\u0080\u009c", '"'),
        ("\u00e2\u0080\u009d", '"'),
        ("\u00e2\u0080\u0098", "'"),
        ("\u00e2\u0080\u0093", "–"),
        ("\u00e2\u0080\u0094", "—"),
        ("â€", '"'),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u2013", "-"),
        ("\u2014", "—"),
        ("&nbsp;", " "),
        ("&quot;", '"'),
    ]
    for original, replacement in replacements:
        text = text.replace(original, replacement)
    try:
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
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


def retry_with_backoff(
    func, max_retries: int = 3, base_delay: int = 5, max_delay: int = 30
):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            delay = min(base_delay * (2**attempt), max_delay)
            time.sleep(delay)


def robust_api_call(
    api_url: str,
    headers: Dict,
    payload: Dict,
    timeout: int = 60,
    max_retries: int = 3,
    api_key: str = None,
) -> str:
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
            "X-Title": "Discussion Grading Tool",
        }
        # Make the API call
        response = requests.post(
            api_url, headers=new_headers, json=payload, timeout=timeout
        )
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
        response_text_clean = re.sub(
            r"^```json\s*", "", response_text_clean, flags=re.DOTALL
        )
        response_text_clean = re.sub(
            r"\s*```$", "", response_text_clean, flags=re.DOTALL
        )
        response_text_clean = re.sub(
            r"^```\s*", "", response_text_clean, flags=re.DOTALL
        )
        # Attempt standard JSON loading
        try:
            return json.loads(response_text_clean)
        except json.JSONDecodeError as e:
            # Use regex to find the likely JSON object
            json_match = re.search(r"\{.*\}", response_text_clean, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                try:
                    return json.loads(extracted_json)
                except json.JSONDecodeError as e2:
                    raise ValueError(
                        f"Could not parse valid JSON even after extraction: {str(e2)}"
                    )
            raise ValueError(f"Could not find or parse valid JSON structure: {str(e)}")

    return retry_with_backoff(parse_json, max_retries=max_retries)


def extract_section_robust(
    text: str,
    section_name: str,
    alternative_names: List[str] = None,
    stop_indicators: List[str] = None,
) -> str:
    """
    Robust section extraction with better boundary detection.
    Correctly handles headers followed by content on the same line.
    """
    alternative_names = alternative_names or []
    stop_indicators = stop_indicators or [
        "Discussion",
        "Reading",
        "Video",
        "Key Terms",
        "Objective",
        "Key Concepts",
        "Materials",
        "Prompt",
    ]
    all_names = [section_name] + alternative_names
    content_lines = []
    in_section = False
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line_clean = line.strip()
        # Check for start of section
        is_header = False
        if not in_section:
            for name in all_names:
                name_escaped = re.escape(name)
                pattern = r"^\s*" + name_escaped + r"[:\s]*(.*)$"
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
                if (
                    re.match(
                        r"^\s*" + re.escape(indicator) + r"[:\s]*",
                        line_clean,
                        re.IGNORECASE,
                    )
                    and len(line_clean.split()) < 8
                ):
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
    content = "\n".join(content_lines).strip()
    return content


def parse_lesson_plan_comprehensive(docx_content: bytes):
    """Comprehensive lesson plan parser from DOCX content."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx library required. Please install it with: pip install python-docx"
        )
    docx_io = io.BytesIO(docx_content)
    document = Document(docx_io)
    # Extract all text
    full_text = "\n".join(
        [p.text.strip() for p in document.paragraphs if p.text.strip()]
    )
    full_text_clean = fix_encoding(full_text)
    # Define sections and their required alternate names/stop indicators dynamically
    sections_to_extract = {
        "Discussion Prompt": (
            ["Discussion", "Prompt", "Discussion Question", "Question"],
            ["Reading", "Video", "Key Terms", "Objective", "Key Concepts", "Materials"],
        ),
        "Reading": (
            ["Assigned Reading", "Required Reading", "READING"],
            [
                "Discussion",
                "Video",
                "Key Terms",
                "Objective",
                "Key Concepts",
                "Materials",
                "Prompt",
            ],
        ),
        "Video": (
            ["VIDEO", "Assigned Video", "Required Video"],
            [
                "Discussion",
                "Reading",
                "Key Terms",
                "Objective",
                "Key Concepts",
                "Materials",
                "Prompt",
            ],
        ),
        "Objective": (
            ["Learning Objective", "Goals"],
            [
                "Discussion",
                "Reading",
                "Video",
                "Key Terms",
                "Objective",
                "Materials",
                "Prompt",
            ],
        ),
        "Key Concepts": (
            ["Concepts", "Main Concepts", "KEY CONCEPTS"],
            [
                "Discussion",
                "Reading",
                "Video",
                "Key Terms",
                "Objective",
                "Materials",
                "Prompt",
            ],
        ),
        "Key Terms": (
            ["KEY TERMS", "Terms", "Vocabulary"],
            [
                "Discussion",
                "Reading",
                "Video",
                "Objective",
                "Key Concepts",
                "Materials",
                "Prompt",
            ],
        ),
    }
    parsed_sections = {}
    for name, (alts, stops) in sections_to_extract.items():
        parsed_sections[name] = extract_section_robust(
            full_text_clean, name, alts, stops
        )
    # --- Key Terms Specific Parsing ---
    key_terms_str = parsed_sections["Key Terms"]
    key_terms = []
    if key_terms_str:
        key_terms_clean = re.sub(r"[\t\-\*\•]", "\n", key_terms_str)
        terms_list = re.split(r",\s*|\n", key_terms_clean)
        for term in terms_list:
            term = term.strip()
            term = re.sub(r"^[\d\.\s\-\)]+", "", term)
            term = re.sub(r"[\.,;\s\-]+$", "", term)
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
            question_match = re.search(
                r"([A-Za-z\s]+[?])", full_text_clean[-1000:], re.DOTALL
            )
            if question_match:
                discussion_prompt = question_match.group(1).strip()
    return (
        discussion_prompt,
        parsed_sections["Reading"],
        parsed_sections["Video"],
        unique_key_terms,
        parsed_sections["Objective"],
        parsed_sections["Key Concepts"],
    )


def normalize_text_for_matching(text: str) -> str:
    if not text:
        return ""
    text = strip_tags(text)
    text = re.sub(r"[*_`#\-,.:;]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def advanced_term_variations(term: str) -> List[str]:
    term_clean = normalize_text_for_matching(term)
    variations = set([term_clean])
    if not term_clean or len(term_clean) < 3:
        return []
    if term_clean.endswith("s"):
        variations.add(term_clean[:-1])
    elif not term_clean.endswith("s"):
        variations.add(term_clean + "s")
    if "-" in term_clean:
        variations.add(term_clean.replace("-", " "))
    return list(variations)


def detect_key_terms_presence(submission_text: str, key_terms: List[str]) -> List[str]:
    if not key_terms or not submission_text:
        return []
    submission_norm = normalize_text_for_matching(submission_text)
    detected_terms = []
    detected_base_terms = set()
    for term in key_terms:
        if term in detected_base_terms:
            continue
        term_variations = advanced_term_variations(term)
        for variation in term_variations:
            try:
                pattern = r"\b" + re.escape(variation) + r"\b"
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


# EXACT Engagement Quality Analysis from your provided code


def _unique_preserve_order(values: List[str]) -> List[str]:
    """Return unique values preserving their first-seen order (case-insensitive)."""
    seen = set()
    ordered = []
    for value in values:
        value_normalized = value.lower()
        if value_normalized not in seen:
            seen.add(value_normalized)
            ordered.append(value.strip())
    return ordered


def generate_video_aliases(video_text: str) -> Set[str]:
    """Generate a set of aliases that could reference the assigned video."""
    base_aliases: Set[str] = {
        "video",
        "youtube",
        "clip",
        "documentary",
        "film",
        "episode",
        "part",
        "lecture",
        "lesson",
    }
    if not video_text:
        return base_aliases
    video_clean = fix_encoding(video_text)
    # Look for quoted titles
    quoted_titles = re.findall(r"[\"'\u201c\u201d]([^\"'\u201c\u201d]{3,120})[\"'\u201c\u201d]", video_clean)
    for title in quoted_titles:
        cleaned_title = title.strip()
        if cleaned_title:
            base_aliases.add(cleaned_title.lower())
            for word in re.findall(r"[A-Za-z][A-Za-z'-]+", cleaned_title):
                if len(word) >= 4:
                    base_aliases.add(word.lower())
    # Capture prominent proper nouns (speaker names, key phrases)
    proper_nouns = re.findall(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", video_clean
    )
    skip_words = {"The", "And", "With", "From", "Into", "Part"}
    for noun in proper_nouns:
        if noun in skip_words:
            continue
        cleaned_noun = noun.strip()
        if len(cleaned_noun) >= 4:
            base_aliases.add(cleaned_noun.lower())
            for word in cleaned_noun.split():
                if len(word) >= 4:
                    base_aliases.add(word.lower())
    return {alias for alias in base_aliases if alias}


def detect_video_references(submission_text: str, video_text: str) -> Dict[str, Any]:
    """Detect explicit references to the assigned video within the submission."""

    detection = {
        "has_reference": False,
        "aliases_found": [],
        "timestamps": [],
        "context_snippets": [],
    }
    if not submission_text:
        return detection
    submission_clean = fix_encoding(submission_text)
    aliases = generate_video_aliases(video_text)
    phrase_aliases = {alias for alias in aliases if " " in alias}
    single_aliases = {alias for alias in aliases if " " not in alias}
    alias_matches = []
    context_matches = []
    # Match phrase aliases directly
    for alias in phrase_aliases:
        try:
            pattern = re.compile(re.escape(alias), re.IGNORECASE)
        except re.error:
            continue
        for match in pattern.finditer(submission_clean):
            alias_matches.append(match.group(0))
            start, end = match.start(), match.end()
            snippet = submission_clean[
                max(0, start - 80) : min(len(submission_clean), end + 80)
            ]
            context_matches.append(snippet.strip())
    # Match single aliases with word boundaries
    for alias in single_aliases:
        if len(alias) < 4 and alias not in {"film", "part"}:
            continue
        pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
        for match in pattern.finditer(submission_clean):
            alias_matches.append(match.group(0))
            start, end = match.start(), match.end()
            snippet = submission_clean[
                max(0, start - 80) : min(len(submission_clean), end + 80)
            ]
            context_matches.append(snippet.strip())
    # Detect timestamps such as 0:12, 1:02:30, or 3 minutes 15 seconds
    timestamp_patterns = [
        re.compile(r"\b(?:[0-9]{1,2}:){1,2}[0-9]{2}\b"),
        re.compile(
            r"\b[0-9]+\s*(?:minutes|min|minute)\s*(?:and\s*)?(?:[0-9]+\s*(?:seconds|secs|sec|s))?\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b[0-9]+\s*(?:second|sec|secs)\b", re.IGNORECASE),
    ]
    timestamp_matches = []
    for pattern in timestamp_patterns:
        for match in pattern.finditer(submission_clean):
            timestamp_matches.append(match.group(0))
            start, end = match.start(), match.end()
            snippet = submission_clean[
                max(0, start - 80) : min(len(submission_clean), end + 80)
            ]
            context_matches.append(snippet.strip())
    alias_matches = _unique_preserve_order(alias_matches)
    timestamp_matches = _unique_preserve_order(timestamp_matches)
    context_matches = _unique_preserve_order(context_matches)[:5]
    detection["aliases_found"] = alias_matches
    detection["timestamps"] = timestamp_matches
    detection["context_snippets"] = context_matches
    detection["has_reference"] = bool(alias_matches or timestamp_matches)
    return detection


def format_video_detection_summary(video_detection: Dict[str, Any]) -> str:
    """Create a concise summary of detected video references for prompts."""

    if not video_detection.get("has_reference"):
        return "No explicit video references detected in the student's submission."
    summary_parts = []
    aliases = video_detection.get("aliases_found", [])
    timestamps = video_detection.get("timestamps", [])
    if aliases:
        alias_display = ", ".join(aliases[:5])
        summary_parts.append(f"Mentions detected: {alias_display}")
    if timestamps:
        timestamp_display = ", ".join(timestamps[:5])
        summary_parts.append(f"Timestamp references: {timestamp_display}")
    return "; ".join(summary_parts) if summary_parts else "Video references detected."


def build_video_feedback_from_detection(video_detection: Dict[str, Any]) -> str:
    """Create human-readable video feedback from detection heuristics."""

    if not video_detection or not video_detection.get("has_reference"):
        return (
            "I didn't see you connect your post to the assigned video. Highlight "
            "specific ideas or examples from the video to show how it informs your "
            "response."
        )
    aliases = video_detection.get("aliases_found", [])
    timestamps = video_detection.get("timestamps", [])
    snippets = video_detection.get("context_snippets", [])
    parts = []
    if aliases:
        alias_display = ", ".join(aliases[:3])
        parts.append(f"highlighting {alias_display}")
    if timestamps:
        timestamp_display = ", ".join(timestamps[:3])
        parts.append(f"pointing to moments such as {timestamp_display}")
    if not parts:
        parts.append("drawing clear connections to the assigned video")
    feedback = (
        "You connected your post to the assigned video by " + " and ".join(parts) + "."
    )
    if snippets:
        example = snippets[0].replace("\n", " ").strip()
        if len(example) > 220:
            example = example[:217].rstrip() + "..."
        feedback += f' For example, you wrote "{example}".'
    return feedback


def analyze_engagement_quality(replies: List[str]) -> Dict[str, Any]:
    replies = [fix_encoding(reply) for reply in replies]
    valid_replies = [reply for reply in replies if reply and len(reply.split()) >= 10]
    num_replies = len(valid_replies)
    if num_replies == 0:
        return {
            "score": 2.0,
            "feedback": "We encourage you to participate in peer discussion! Substantive replies engage with at least one classmate.",
            "num_replies": 0,
            "highest_quality_score": 2.0,
        }
    reply_quality_scores = []
    for reply in valid_replies:
        reply_lower = reply.lower()
        word_count = len(reply.split())
        quality_score = 3.0
        # Indicators of deep engagement (expanded and more flexible)
        deep_engagement_indicators = [
            "i disagree",
            "i believe that",
            "i believe it",
            "i agree that",
            "i understand",
            "building on",
            "contrary to",
            "critique",
            "elaborate",
            "perspective",
            "i appreciated",
            "can you relate",
            "in addition to",
            "i think",
            "surprised by",
            "recommend",
            "interesting point",
            "you mentioned",
            "your point about",
            "i would argue",
            "similar to what you said",
            "expanding on",
            "different perspective",
            "your analysis",
            "i noticed",
            "as you pointed out",
            "building upon",
            "along those lines",
            "your observation",
            "another way to think",
            "i wonder if",
            "what if we consider",
        ]
        # Indicators of supporting arguments (expanded)
        supporting_indicators = [
            "because",
            "however",
            "although",
            "critical",
            "analysis",
            "blueprint",
            "therefore",
            "moreover",
            "furthermore",
            "in contrast",
            "similarly",
            "for example",
            "for instance",
            "this suggests",
            "this demonstrates",
            "evidence shows",
            "research indicates",
            "as shown by",
            "considering that",
        ]
        # Check for engagement indicators
        has_deep_engagement = any(
            term in reply_lower for term in deep_engagement_indicators
        )
        has_supporting_args = any(term in reply_lower for term in supporting_indicators)
        # Check for question marks (asking thoughtful questions)
        has_questions = "?" in reply
        # Check for specific examples or concrete details (numbers, names, specific events)
        has_specific_details = bool(re.search(r"\b\d+\b", reply)) or bool(
            re.search(
                r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:argued|stated|mentioned|noted|wrote|said)",
                reply,
            )
        )
        # Scoring logic - more generous and quality-focused
        if word_count >= 75 and (
            has_deep_engagement or has_supporting_args or has_specific_details
        ):
            quality_score = 4.0
        elif word_count >= 60 and (has_deep_engagement or has_supporting_args):
            quality_score = 4.0
        elif word_count >= 50 and has_deep_engagement and has_supporting_args:
            quality_score = 4.0
        elif word_count >= 50 and (
            has_deep_engagement or has_supporting_args or has_questions
        ):
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
        name_match = re.search(
            r"^(?:Hello|Hi|Dear|Hey|To)\s+([A-Za-z]+)\s*([A-Za-z]*),?",
            highest_quality_reply.strip(),
            re.IGNORECASE,
        )
        if name_match:
            recipient_name = name_match.group(1).strip()
        else:
            name_match_only = re.search(r"^([A-Za-z]+),", highest_quality_reply.strip())
            if name_match_only:
                recipient_name = name_match_only.group(1).strip()
    # Generate specific feedback for point deductions
    if highest_quality_score >= 4.0:
        feedback = f"Excellent engagement! Your reply to {recipient_name} demonstrates substantive interaction with their ideas, showing depth of analysis and critical thinking that meets the highest standards for peer discussion."
    elif highest_quality_score >= 3.5:
        feedback = f"Strong engagement. Your meaningful reply to {recipient_name} shows good interaction with their ideas. Your response demonstrates solid understanding and contributes meaningfully to the discussion. To earn full credit (4.0), consider adding more detailed analysis, specific examples, or deeper critical engagement with your peers' arguments."
    elif highest_quality_score >= 3.0:
        feedback = f"Adequate engagement. Your contribution meets the substantive length requirement. To strengthen future replies, consider adding more detailed analysis, specific examples, or deeper critical engagement with your peers' arguments. Points were deducted because your response lacked the depth of analysis expected for higher credit."
    else:
        feedback = "Your participation meets the minimum requirement, but your replies lack substantive length or meaningful interaction. Focus on directly responding to and debating your peers' ideas in detail. Points were deducted because your response was too brief or did not engage substantively with your peers' ideas."
    return {
        "score": highest_quality_score,
        "feedback": feedback,
        "num_replies": num_replies,
        "highest_quality_score": highest_quality_score,
    }


def construct_final_feedback(
    llm_results: Dict,
    local_scores: Dict[str, float],
    local_feedback: Dict[str, str],
    improvement_areas: List[str],
    student_first_name: str,
    grading_scale: str,
) -> str:
    combined_prompt_key_score = (
        local_scores["prompt_score"] + local_scores["key_terms_score"]
    )
    prompt_feedback = llm_results.get(
        "prompt_feedback", "Feedback missing for prompt quality."
    )
    key_terms_feedback = llm_results.get(
        "key_terms_feedback",
        local_feedback.get("key_terms_fallback", "Feedback missing for key terms."),
    )
    reading_feedback = local_feedback.get(
        "reading_feedback",
        llm_results.get("reading_feedback", "Feedback missing for reading reference."),
    )
    video_feedback = llm_results.get(
        "video_feedback", "Feedback missing for video reference."
    )
    video_detection = local_feedback.get("video_detection", {})
    if (
        not video_feedback
        or "Feedback missing" in video_feedback
        or not video_feedback.strip()
    ):
        video_feedback = build_video_feedback_from_detection(video_detection)
    general_feedback_llm = llm_results.get(
        "general_feedback", "Overall submission quality was strong."
    )
    engagement_feedback = local_feedback["engagement_feedback"]

    def transform_to_second_person(text):
        if not text:
            return ""
        # Fix the "you's" issue by replacing it with "you are"
        text = re.sub(r"\byou\'s\b", "you are", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\b(The student|This student|They|Their|He|His|She|Her)\b",
            lambda m: {
                "The student": "You",
                "This student": "You",
                "They": "You",
                "Their": "Your",
                "He": "You",
                "His": "Your",
                "She": "You",
                "Her": "Your",
            }.get(m.group(1), m.group(1)),
            text,
            flags=re.IGNORECASE,
        )
        if text:
            text = text.strip()
            return text[0].upper() + text[1:]
        return text

    prompt_feedback = transform_to_second_person(prompt_feedback)
    key_terms_feedback = transform_to_second_person(key_terms_feedback)
    reading_feedback = transform_to_second_person(reading_feedback)
    video_feedback = transform_to_second_person(video_feedback)
    general_feedback_llm = transform_to_second_person(general_feedback_llm)
    prompt_key_combined_feedback = (
        f"{prompt_feedback.strip()} {key_terms_feedback.strip()}"
    )
    # Format scores based on grading scale
    if grading_scale == "15-point (3 categories)":
        # Scale up the scores for 15-point scale (5 points per category)
        # For 15-point scale, prompt and key terms are 2.5 points each
        scaled_prompt_score = round_nearest_half(
            local_scores["prompt_score"] * 1.25
        )  # Scale from 2.0 to 2.5
        scaled_key_terms_score = round_nearest_half(
            local_scores["key_terms_score"] * 1.25
        )  # Scale from 2.0 to 2.5
        scaled_video_score = round_nearest_half(
            local_scores["video_score"] * 1.25
        )  # Scale from 4.0 to 5.0
        scaled_reading_score = round_nearest_half(
            local_scores["reading_score"] * 1.25
        )  # Scale from 4.0 to 5.0
        prompt_key_formatted = f"PROMPT AND KEY TERMS [{scaled_prompt_score + scaled_key_terms_score:.1f}/5.0]: {prompt_key_combined_feedback}"
        video_formatted = (
            f"REFERENCE TO VIDEO [{scaled_video_score:.1f}]: {video_feedback}"
        )
        reading_formatted = (
            f"REFERENCE TO READING [{scaled_reading_score:.1f}]: {reading_feedback}"
        )
    else:  # 16-point (4 categories)
        prompt_key_formatted = f"PROMPT AND KEY TERMS [{combined_prompt_key_score:.1f}]: {prompt_key_combined_feedback}"
        video_formatted = (
            f"REFERENCE TO VIDEO [{local_scores['video_score']:.1f}]: {video_feedback}"
        )
        reading_formatted = f"REFERENCE TO READING [{local_scores['reading_score']:.1f}]: {reading_feedback}"
        engagement_formatted = f"DISCUSSION ENGAGEMENT [{local_scores['engagement_score']:.1f}]: {engagement_feedback}"
    if improvement_areas:
        # Remove "Discussion Engagement" from improvement areas for 15-point scale
        if (
            grading_scale == "15-point (3 categories)"
            and "Discussion Engagement" in improvement_areas
        ):
            improvement_areas = [
                area for area in improvement_areas if area != "Discussion Engagement"
            ]
        if improvement_areas:
            improvement_focus = f"{student_first_name}, while your work demonstrates strong engagement with the content, focus on improving in the area(s) of: {', '.join(improvement_areas)} to maximize your synthesis of the concepts. {general_feedback_llm}"
        else:
            improvement_focus = f"Outstanding work {student_first_name}! You excelled on all sections of this assignment. {general_feedback_llm}"
    else:
        improvement_focus = f"Outstanding work {student_first_name}! You excelled on all sections of this assignment. {general_feedback_llm}"
    general_formatted = f"GENERAL FEEDBACK: {improvement_focus}"
    # Construct final feedback based on grading scale
    if grading_scale == "15-point (3 categories)":
        final_feedback = "\n".join(
            [
                prompt_key_formatted,
                video_formatted,
                reading_formatted,
                general_formatted,
            ]
        )
    else:  # 16-point (4 categories)
        final_feedback = "\n".join(
            [
                prompt_key_formatted,
                video_formatted,
                reading_formatted,
                engagement_formatted,
                general_formatted,
            ]
        )
    final_feedback = re.sub(r"\s{2,}", " ", final_feedback).strip()
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
    grading_scale: str,
) -> Dict[str, str]:
    """Grade submission with comprehensive local and API scoring."""
    # 1. Local Scoring & Data Preparation
    engagement_analysis = analyze_engagement_quality(replies)
    engagement_score = engagement_analysis["score"]
    detected_terms = detect_key_terms_presence(submission_text, key_terms)
    detected_terms_str = (
        ", ".join(detected_terms) if detected_terms else "none detected"
    )
    video_detection = detect_video_references(submission_text, video_text)
    video_detection_summary = format_video_detection_summary(video_detection)
    reading_info = {}
    # --- Extract REQUIRED READING INFO ---
    # The lesson plan will always say "READING: [reading details]"
    # Extract the author and page numbers from the reading text
    # First, check if "READING:" is in the text
    if "READING:" in reading_text:
        # Extract the full line that contains "READING:"
        reading_line = ""
        for line in reading_text.split("\n"):
            if "READING:" in line:
                reading_line = line.strip()
                break
        # Extract author (first word after "READING:")
        author_match = re.search(
            r"READING:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", reading_line
        )
        assigned_author = author_match.group(1).strip() if author_match else ""
    else:
        # Fallback if "READING:" is not explicitly in the text
        # Extract author (first word before the first comma)
        author_match = re.search(
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", reading_text.strip()
        )
        assigned_author = author_match.group(1).strip() if author_match else ""
    # Look for page numbers in various formats
    page_numbers = []
    # Try multiple patterns to extract page numbers
    patterns = [
        r"pages?\s+([\d.,\s-]+)",  # pages 81-83 or page 81
        r"p\.?\s*([\d.,\s-]+)",  # p.81 or p 81
        r"([\d.,\s-]+)",  # just numbers (fallback)
    ]
    for pattern in patterns:
        pages_match = re.search(pattern, reading_text, re.IGNORECASE)
        if pages_match:
            page_str = pages_match.group(1)
            # Handle various page formats like "3.1, 3.2, 3.3, and 3.4" or "10-15" or "10, 12, 14"
            page_numbers = re.findall(r"[\d.]+", page_str)
            # Convert to float for comparison
            page_numbers = [float(p) for p in page_numbers]
            break
    # Set reading info based on extracted data
    if assigned_author:
        reading_info["author_last_name"] = assigned_author
    else:
        reading_info["author_last_name"] = ""
    if page_numbers:
        page_numbers.sort()
        # Format page range for feedback - fix for issue 1
        if len(page_numbers) == 1:
            page_num = page_numbers[0]
            if page_num.is_integer():
                reading_info["page_range_expected"] = f"page {int(page_num)}"
            else:
                reading_info["page_range_expected"] = f"page {page_num}"
        elif len(page_numbers) == 2:
            # Always use a hyphen for two pages, which are now sorted
            if page_numbers[0].is_integer() and page_numbers[1].is_integer():
                reading_info["page_range_expected"] = (
                    f"pages {int(page_numbers[0])}-{int(page_numbers[1])}"
                )
            else:
                reading_info["page_range_expected"] = (
                    f"pages {page_numbers[0]}-{page_numbers[1]}"
                )
        else:
            # Use a comma for lists of three or more pages
            formatted_pages = []
            for p in page_numbers:
                if p.is_integer():
                    formatted_pages.append(str(int(p)))
                else:
                    formatted_pages.append(str(p))
            reading_info["page_range_expected"] = f"pages {', '.join(formatted_pages)}"
    else:
        reading_info["page_range_expected"] = "unspecified pages"
    # ----------------------------------------------------
    # D. Analyze citation presence based on updated rules
    highest_max_reading_score = 2.0  # Default Minimum Score
    best_citation_status_msg = (
        "I didn't see a clear reference to the assigned reading. The minimum score of "
        "**2.0** applies."
    )
    detected_author = ""
    # Only proceed with citation checking if we have an author
    if reading_info["author_last_name"]:
        assigned_author_lower = reading_info["author_last_name"].lower()
        # Check if the author name is present in the submission
        author_present = re.search(
            r"\b" + re.escape(assigned_author_lower) + r"\b", submission_text.lower()
        )
        # Check if any of the page numbers are present in the submission
        page_present = False
        detected_pages = []
        if page_numbers:
            # If the assigned reading is a range, check for numbers within it
            if len(page_numbers) == 2:
                start_page = int(min(page_numbers))
                end_page = int(max(page_numbers))
                # Find all numbers in the submission that look like page citations
                cited_pages = re.findall(
                    r"(?:p|pg|page)s?\.?\s*(\d+)", submission_text, re.IGNORECASE
                )
                for page_str in cited_pages:
                    cited_num = int(page_str)
                    if start_page <= cited_num <= end_page:
                        page_present = True
                        detected_pages.append(page_str)
            # Otherwise, check for specific pages listed
            else:
                for page in page_numbers:
                    page_str = (
                        str(int(page))
                        if isinstance(page, float) and page.is_integer()
                        else str(page)
                    )
                    patterns = [
                        r"\b" + re.escape(page_str) + r"\b",
                        r"\bp\.?\s*" + re.escape(page_str) + r"\b",
                        r"\bpage\s*" + re.escape(page_str) + r"\b",
                        r"\bpages?\s*" + re.escape(page_str) + r"\b",
                    ]
                    for pattern in patterns:
                        if re.search(pattern, submission_text, re.IGNORECASE):
                            page_present = True
                            detected_pages.append(page_str)
                            break  # Found this page, move to the next assigned page
                    if page_present and len(page_numbers) > 1:
                        # For multiple specific pages, finding one is enough.
                        break
        # Determine score based on author and page presence
        if author_present and page_present:
            highest_max_reading_score = 4.0
            best_citation_status_msg = (
                f"You included both the author ('{assigned_author}') and a relevant page "
                "number from the assigned reading. Full credit awarded."
            )
        elif author_present:
            highest_max_reading_score = 3.0
            best_citation_status_msg = (
                f"You mentioned the author ('{assigned_author}'), but you didn't include a "
                "specific page number from the assigned reading. Partial credit awarded."
            )
        elif page_present:
            highest_max_reading_score = 3.5
            best_citation_status_msg = (
                f"You cited a page number from the assigned reading, but you didn't "
                f"mention the author ('{assigned_author}'). Partial credit awarded."
            )
        # Check for incorrect author if the correct one wasn't found
        if not author_present:
            # Look for any capitalized name that might be an incorrect author
            potential_authors = re.findall(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", submission_text
            )
            for potential_author in potential_authors:
                # Skip common words that might be capitalized
                if potential_author.lower() in [
                    "the",
                    "and",
                    "but",
                    "or",
                    "for",
                    "nor",
                    "on",
                    "at",
                    "to",
                    "from",
                    "by",
                ]:
                    continue
                # Check if this could be an author citation
                if (
                    len(potential_author) > 3
                ):  # Only consider names longer than 3 characters
                    detected_author = potential_author
                    if page_present:
                        highest_max_reading_score = 3.5
                        best_citation_status_msg = (
                            f"You cited page {', '.join(detected_pages)} but attributed it to "
                            f"'{potential_author}', which doesn't match the assigned author "
                            f"('{assigned_author}'). Partial credit awarded for the page "
                            "reference."
                        )
                    else:
                        highest_max_reading_score = 2.5
                        best_citation_status_msg = (
                            f"You mentioned '{potential_author}', but that doesn't match the "
                            f"assigned author ('{assigned_author}'). Partial credit awarded."
                        )
                    break
    else:
        # No author found in reading text, so we can't check citations
        highest_max_reading_score = 2.0
        best_citation_status_msg = f"The assigned reading information did not specify an author. The minimum score of **2.0** applies."
    max_reading_score = highest_max_reading_score
    citation_status_msg = best_citation_status_msg
    # --- GENERATE READING FEEDBACK BASED ON SCORE ---
    if max_reading_score == 4.0:
        reading_feedback_local = (
            f"You grounded your response in {assigned_author}'s text by naming the author "
            f"and citing {reading_info['page_range_expected']}. That thorough connection "
            "earns full credit for this section."
        )
    elif max_reading_score == 3.5:
        if detected_author:
            # Fix for issue 2 - don't mention the specific wrong author
            reading_feedback_local = (
                "You referenced page number(s) from the assigned reading, but the author "
                "you mentioned doesn't match the assigned text. Double-check the author "
                "next time to earn full credit."
            )
        else:
            reading_feedback_local = (
                "You cited a page number from the assigned reading, but you didn't "
                "mention the author. Include both elements next time to secure full "
                "credit."
            )
    elif max_reading_score == 3.0:
        reading_feedback_local = (
            f"You mentioned {assigned_author}, which shows you're drawing from the text. "
            f"To reach full credit, add a specific page reference from "
            f"{reading_info['page_range_expected']}."
        )
    elif max_reading_score == 2.5:
        # Fix for issue 2 - don't mention the specific wrong author
        reading_feedback_local = (
            "You cited a different author than the one assigned. Bring in the correct "
            "author and add a page reference to earn more credit."
        )
    else:  # 2.0
        if assigned_author:
            reading_feedback_local = (
                f"Your response nods to ideas from {assigned_author}, but it doesn't "
                f"include the required combination of the author's name and a page "
                f"reference from {reading_info['page_range_expected']}. Add both pieces "
                "to move beyond the minimum score."
            )
        else:
            reading_feedback_local = (
                "Your response engages with the reading, but it doesn't include the "
                "required author and page reference. Add both to move beyond the "
                "minimum score."
            )
    # ----------------------------------------------------
    # Local scores - reading score is SET here, not by LLM
    local_scores = {
        "engagement_score": engagement_score,
        "prompt_score": 0.0,
        "reading_score": max_reading_score,  # DIRECTLY SET FROM PYTHON DETECTION
        "key_terms_score": 0.0,
        "video_score": 0.0,
    }
    local_feedback = {
        "engagement_feedback": engagement_analysis["feedback"],
        "reading_feedback": reading_feedback_local,  # PYTHON-GENERATED FEEDBACK
        "key_terms_fallback": f"LLM failed to provide key terms feedback. Detected terms: {detected_terms_str}",
        "video_detection": video_detection,
        "video_detection_summary": video_detection_summary,
    }
    # LLM scoring criteria - reading section is informational only
    llm_scoring_criteria = f"""

SCORING Guidelines for LLM (10 points total - Reading is scored separately):

1. PROMPT ADHERENCE (Minimum 1.0 - 2.0): How well does the student address the entire prompt? (2.0 Maximum)

2. READING REFERENCE: **This section is scored separately by the system as {max_reading_score:.1f}. Do not provide a reading_score in your response.**

    - Citation Status (for context): {citation_status_msg}

3. VIDEO REFERENCE (Minimum 2.0 - 4.0): How specific and relevant is the use of the assigned video material?

    - Full credit (4.0) requires clear use of concepts demonstrated by specific examples or accurate summaries.
    - **A specific timestamp is NOT required for a 4.0 score.**

4. KEY TERMS USAGE (Minimum 1.0 - 2.0): Did the student use at least one key term (from the detected list) in a way that demonstrates contextual understanding? (2.0 Maximum)

    - FULL CREDIT (2.0) MUST BE AWARDED if ONE or more terms are used meaningfully.

Detected Key Terms to review for usage: "{detected_terms_str}"

Detected Video Evidence identified locally: "{video_detection_summary}"

"""
    prompt_for_llm = f"""Grade this student discussion submission based ONLY on the following criteria. Reading Reference ({max_reading_score:.1f}) and Engagement ({engagement_score}) are scored separately by the system.

STUDENT: {student_first_name}

ASSIGNMENT CONTEXT:

Prompt: {discussion_prompt[:300]}...

Reading: {reading_text[:200]}...

Video: {video_text[:200]}...

{llm_scoring_criteria}

IMPORTANT: Provide SPECIFIC and ENCOURAGING feedback in the second person ("You", "Your").

**DO NOT include "reading_score" in your JSON response - it is handled separately.**

Respond with ONLY valid JSON. Omit any markdown fences (```json). Use floating point numbers rounded to the nearest 0.5.

{{

    "prompt_score": "2.0",
    "video_score": "4.0",
    "key_terms_score": "2.0",
    "prompt_feedback": "You successfully articulated how involuntary servitude was preserved and connected this theme to present-day issues.",
    "video_feedback": "You clearly referenced the video context regarding convict leasing and the continuation of forced labor, demonstrating a strong grasp of the material.",
    "key_terms_feedback": "Your contextual usage of key terms earns full credit, demonstrating clear understanding of the material.",
    "general_feedback": "Your arguments were well-structured and demonstrated impressive critical thinking."

}}

SUBMISSION TEXT:

{submission_text[:1500]}

"""
    # 2. API Scoring (if available)
    api_results = {}
    if api_key and validate_api_key(api_key):
        try:
            # Prepare the API request
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            payload = {
                "model": "google/gemini-2.5-flash-preview-09-2025",
                "messages": [{"role": "user", "content": prompt_for_llm}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "max_tokens_for_reasoning": 512,
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
            st.error(
                "An unexpected error occurred while calling the API. Please try again later."
            )
            api_results = {}
    else:
        if not api_key:
            st.warning("No API key provided. Using local scoring only.")
        else:
            st.warning("Invalid API key format. Using local scoring only.")
    # 3. Final Score Compilation
    try:
        local_scores["prompt_score"] = round_nearest_half(
            max(1.0, min(2.0, float(api_results.get("prompt_score", 1.0))))
        )
        if "video_score" in api_results:
            local_scores["video_score"] = round_nearest_half(
                max(2.0, min(4.0, float(api_results.get("video_score", 2.0))))
            )
        else:
            fallback_video_score = 3.0 if video_detection.get("has_reference") else 2.0
            local_scores["video_score"] = round_nearest_half(fallback_video_score)
        local_scores["key_terms_score"] = round_nearest_half(
            max(1.0, min(2.0, float(api_results.get("key_terms_score", 1.0))))
        )
        # reading_score already set from Python detection
    except (ValueError, TypeError):
        local_scores["prompt_score"] = 1.0
        local_scores["video_score"] = round_nearest_half(
            3.0 if video_detection.get("has_reference") else 2.0
        )
        local_scores["key_terms_score"] = 1.0
        # reading_score remains as set from Python detection
    # Identify lowest scoring component for General Feedback
    # For 15-point scale, we only consider 3 categories (excluding engagement)
    if grading_scale == "15-point (3 categories)":
        weighted_scores = {
            "Prompt Adherence": local_scores["prompt_score"] / 2.0,
            "Key Terms Usage": local_scores["key_terms_score"] / 2.0,
            "Reading Reference": local_scores["reading_score"] / 4.0,
            "Video Reference": local_scores["video_score"] / 4.0,
        }
    else:  # 16-point (4 categories)
        weighted_scores = {
            "Prompt Adherence and Key Terms": (
                local_scores["prompt_score"] + local_scores["key_terms_score"]
            )
            / 4.0,
            "Reading Reference": local_scores["reading_score"] / 4.0,
            "Video Reference": local_scores["video_score"] / 4.0,
            "Discussion Engagement": local_scores["engagement_score"] / 4.0,
        }
    sorted_improvement = sorted(weighted_scores.items(), key=lambda item: item[1])
    improvement_areas = [
        name for name, score in sorted_improvement if score < 1.0 and score > 0.0
    ]
    total = sum(local_scores.values())
    total_score = round_nearest_half(total)
    # Scale scores for 15-point system
    if grading_scale == "15-point (3 categories)":
        # Scale up the scores for 15-point scale (5 points per category)
        # For 15-point scale, prompt and key terms are 2.5 points each
        scaled_prompt_score = round_nearest_half(
            local_scores["prompt_score"] * 1.25
        )  # Scale from 2.0 to 2.5
        scaled_key_terms_score = round_nearest_half(
            local_scores["key_terms_score"] * 1.25
        )  # Scale from 2.0 to 2.5
        scaled_video_score = round_nearest_half(
            local_scores["video_score"] * 1.25
        )  # Scale from 4.0 to 5.0
        scaled_reading_score = round_nearest_half(
            local_scores["reading_score"] * 1.25
        )  # Scale from 4.0 to 5.0
        scaled_total = round_nearest_half(
            scaled_prompt_score
            + scaled_key_terms_score
            + scaled_video_score
            + scaled_reading_score
        )
        # FIX: Keep scores as numeric values, not strings
        final_grades = {
            "prompt_score": scaled_prompt_score,  # Use scaled value
            "key_terms_score": scaled_key_terms_score,  # Use scaled value
            "video_score": scaled_video_score,  # Use scaled value
            "reading_score": scaled_reading_score,  # Use scaled value
            "engagement_score": 0,  # Not used in 15-point scale
            "total_score": scaled_total,  # Use scaled total
        }
    else:
        # FIX: Keep scores as numeric values, not strings
        final_grades = {
            "prompt_score": local_scores["prompt_score"],
            "key_terms_score": local_scores["key_terms_score"],
            "video_score": local_scores["video_score"],
            "reading_score": local_scores["reading_score"],
            "engagement_score": local_scores["engagement_score"],
            "total_score": total_score,
        }
    final_grades["feedback"] = construct_final_feedback(
        api_results,
        local_scores,
        local_feedback,
        improvement_areas,
        student_first_name,
        grading_scale,
    )
    return final_grades


# ============================================

# STREAMLIT APP UI

# ============================================

# Set up the Streamlit page

st.set_page_config(
    page_title="Discussion Grading Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling with white text in dropdown

st.markdown(
    f"""

<style>

    .main-header {{
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
    }}
    .info-box {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }}
    .info-box h3 {{
        color: black !important;
    }}
    .scale-info {{
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
        color: black !important;
    }}
    .scale-info h4 {{
        color: black !important;
    }}
    .scale-info ul {{
        color: black !important;
    }}
    .step-button {{
        color: black !important;
    }}
    .download-button {{
        color: black !important;
    }}
    /* Bronze text styling for dropdown UI */
    .stSelectbox > div > div > div,
    .stSelectbox > div > div > div > div,
    .stSelectbox label,
    .stSelectbox > label > div,
    .stSelectbox > label > div > div,
    .stSelectbox > label > div > div > div,
    .stSelectbox > label > div > div > div > div,
    .stSidebar .stSelectbox,
    .stSidebar .stSelectbox > div > div > div,
    .stSidebar .stSelectbox > div > div > div > div,
    .stSidebar .stSelectbox label,
    .stSidebar .stSelectbox > label > div,
    .stSidebar .stSelectbox > label > div > div,
    .stSidebar .stSelectbox > label > div > div > div,
    .stSidebar .stSelectbox > label > div > div > div > div,
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div,
    div[data-baseweb="select"] > div > div > div,
    div[data-baseweb="select"] > div > div > div > div,
    div[data-baseweb="select"] > div > div > div > div > div,
    div[data-baseweb="select"] > div > div > div > div > div > div,
    div[data-baseweb="select"] div[role="listbox"],
    div[data-baseweb="select"] div[role="listbox"] > div,
    div[data-baseweb="select"] div[role="listbox"] > div > div,
    div[data-baseweb="select"] div[role="listbox"] > div > div > div {{
        color: {BRONZE_COLOR} !important;
    }}
    /* Selected and hover states */
    div[data-baseweb="select"] div[role="listbox"] > div[data-selected="true"],
    div[data-baseweb="select"] div[role="listbox"] > div:hover {{
        color: {BRONZE_COLOR} !important;
    }}
    /* Dropdown arrow */
    svg[data-testid="stSelectboxDropdownIcon"] {{
        fill: {BRONZE_COLOR} !important;
    }}

</style>

""",
    unsafe_allow_html=True,
)

# Header

st.markdown(
    '<h1 class="main-header">📚 Automated Discussion Grading Tool</h1>',
    unsafe_allow_html=True,
)

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
    api_key_input = st.sidebar.text_input(
        "OpenRouter API Key", type="password", help="Enter your OpenRouter API key"
    )
    if api_key_input:
        api_key = api_key_input.strip()
        api_key_source = "User input"

else:
    # If we already have an API key, show a masked input
    st.sidebar.text_input(
        "OpenRouter API Key",
        type="password",
        value="•••••••••••••••••••••••••••••••••",
        disabled=True,
    )

# Display API key status

    if api_key.startswith("sk-or-v1-") and len(api_key) > 20:
        st.sidebar.success("API key format appears valid")
    else:
        st.sidebar.error(
            "Invalid API key format. OpenRouter API keys should start with 'sk-or-v1-' and be at least 20 characters long."
        )

else:
    st.sidebar.error("No API key provided. Please enter your API key to continue.")

st.sidebar.markdown("Get your API key from [OpenRouter](https://openrouter.ai/)")

# Add grading scale selector

grading_scale = st.sidebar.selectbox(
    "Select Grading Scale",
    ["15-point (3 categories)", "16-point (4 categories)"],
    index=0,
    help="Choose between the 15-point scale (3 categories) or the 16-point scale (4 categories)",
)

# Display information about the selected grading scale

if grading_scale == "15-point (3 categories)":
    st.sidebar.markdown(
        """
    <div class="scale-info">
    <h4>15-Point Scale (3 Categories)</h4>
    <ul>
        <li>Prompt Adherence (2.5 points)</li>
        <li>Key Terms (2.5 points)</li>
        <li>Reading Reference (5.0 points)</li>
        <li>Video Reference (5.0 points)</li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

else:
    st.sidebar.markdown(
        """
    <div class="scale-info">
    <h4>16-Point Scale (4 Categories)</h4>
    <ul>
        <li>Prompt Adherence and Key Terms (4.0 points)</li>
        <li>Reading Reference (4.0 points)</li>
        <li>Video Reference (4.0 points)</li>
        <li>Discussion Engagement (4.0 points)</li>
    </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")

# Main content area

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="info-box"><h3 class="step-button">Step 1: Upload Files</h3></div>',
        unsafe_allow_html=True,
    )
    csv_file = st.file_uploader(
        "Upload CSV file with student submissions", type=["csv"], key="csv_uploader"
    )
    st.markdown(
        "The CSV should contain columns for student names, initial posts, and replies."
    )
    # Add debugging information
    if csv_file is not None:
        st.success(f"CSV file uploaded: {csv_file.name}")
    else:
        st.info("Please upload a CSV file to continue.")

with col2:
    st.markdown(
        '<div class="info-box"><h3 class="step-button">Step 2: Upload Lesson Plan</h3></div>',
        unsafe_allow_html=True,
    )
    docx_file = st.file_uploader(
        "Upload DOCX lesson plan", type=["docx"], key="docx_uploader"
    )
    st.markdown(
        "The lesson plan should contain discussion prompts, reading assignments, and key terms."
    )
    # Add debugging information
    if docx_file is not None:
        st.success(f"DOCX file uploaded: {docx_file.name}")
    else:
        st.info("Please upload a DOCX file to continue.")

# Process files when both are uploaded

if csv_file and docx_file:
    st.markdown("---")
    st.markdown(
        '<div class="info-box"><h3 class="step-button">Step 3: Process Files</h3></div>',
        unsafe_allow_html=True,
    )
    if st.button("🚀 Process Files", type="primary"):
        if not api_key:
            st.error("Please enter your OpenRouter API key in the sidebar.")
        elif not validate_api_key(api_key):
            st.error(
                "Invalid API key format. OpenRouter API keys should start with 'sk-or-v1-' and be at least 20 characters long."
            )
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
                    (
                        discussion_prompt,
                        reading_text,
                        video_text,
                        key_terms,
                        objective,
                        key_concepts,
                    ) = parse_lesson_plan_comprehensive(docx_content)
                    progress_bar.progress(0.25)
                    st.text("Reading CSV file...")
                    # Process CSV
                    try:
                        csv_io = io.StringIO(csv_content.decode("utf-8"))
                        rows = list(csv.DictReader(csv_io))
                    except UnicodeDecodeError:
                        st.error(
                            "Error decoding CSV file. Please ensure it's saved in UTF-8 format."
                        )
                        st.stop()
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        st.stop()
                    # Check if we got any rows
                    if not rows:
                        st.error(
                            "No data found in CSV file. Please check the file format."
                        )
                        st.stop()
                    # Store original columns
                    original_columns = list(rows[0].keys()) if rows else []
                    # Display extracted information
                    progress_bar.progress(0.5)
                    st.text("Analyzing submissions...")
                    # Create a dataframe to store results
                    results = []
                    # Process each student submission
                    total_students = len(rows)
                    for i, row in enumerate(rows):
                        student_name = row.get(
                            "Name",
                            row.get("Student Name", row.get("Username", "Unknown")),
                        )
                        initial_post = row.get(
                            "Initial Post", row.get("Initial Posts", "")
                        )
                        replies = [
                            row.get(f"Reply {j}", "")
                            for j in range(1, 4)
                            if row.get(f"Reply {j}", "")
                        ]
                        # Extract first name for personalization
                        student_first_name = (
                            student_name.split()[0] if student_name else "Student"
                        )
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
                            grading_scale,
                        )
                        # Add to results - preserving all original columns
                        result_row = row.copy()
                        result_row.update(grade_result)
                        results.append(result_row)
                        # Update progress
                        progress = (i + 1) / total_students
                        progress_bar.progress(0.5 + progress * 0.5)
                    # Create dataframe from results
                    df = pd.DataFrame(results)
                    # Display results
                    st.success("Processing complete!")
                    st.markdown(
                        '<div class="info-box"><h3>Grading Results</h3></div>',
                        unsafe_allow_html=True,
                    )
                    # Show statistics
                    st.markdown("### 📊 Grade Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_score = df["total_score"].mean()
                        st.metric("Average Score", f"{avg_score:.2f}")
                    with col2:
                        median_score = df["total_score"].median()
                        st.metric("Median Score", f"{median_score:.2f}")
                    with col3:
                        min_score = df["total_score"].min()
                        max_score = df["total_score"].max()
                        st.metric("Score Range", f"{min_score:.1f} - {max_score:.1f}")
                    # Show score distribution
                    st.markdown("### 📈 Score Distribution")
                    fig = px.histogram(
                        df, x="total_score", nbins=20, title="Distribution of Scores"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    # Show detailed results
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(df)
                    # Download button
                    csv_output = df.to_csv(index=False)
                    # Get the original filename and add GRADED_ prefix
                    original_filename = csv_file.name
                    base_filename = os.path.splitext(original_filename)[
                        0
                    ]  # Remove .csv extension
                    graded_filename = f"GRADED_{base_filename}.csv"
                    st.markdown('<div class="download-button">', unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Graded CSV",
                        data=csv_output,
                        file_name=graded_filename,
                        mime="text/csv",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    import traceback

                    st.error(traceback.format_exc())
