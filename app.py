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

# Set up the Streamlit page
st.set_page_config(
    page_title="Discussion Grading Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📚 Automated Discussion Grading Tool</h1>', unsafe_allow_html=True)
st.markdown("Upload your CSV and DOCX files to grade student discussions.")

# Sidebar for configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
st.sidebar.markdown("Get your API key from [OpenRouter](https://openrouter.ai/)")
st.sidebar.markdown("---")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="info-box"><h3>Step 1: Upload Files</h3></div>', unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload CSV file with student submissions", type=['csv'])
    st.markdown("The CSV should contain columns for student names, initial posts, and replies.")

with col2:
    st.markdown('<div class="info-box"><h3>Step 2: Upload Lesson Plan</h3></div>', unsafe_allow_html=True)
    docx_file = st.file_uploader("Upload DOCX lesson plan", type=['docx'])
    st.markdown("The lesson plan should contain discussion prompts, reading assignments, and key terms.")

# Process files when both are uploaded
if csv_file and docx_file:
    if st.button("🚀 Process Files", type="primary"):
        if not api_key:
            st.error("Please enter your OpenRouter API key in the sidebar.")
        else:
            with st.spinner("Processing files... This may take a few minutes."):
                try:
                    # Read files
                    csv_content = csv_file.read()
                    docx_content = docx_file.read()
                    
                    # Add a progress bar
                    progress_bar = st.progress(0)
                    st.text("Parsing lesson plan...")
                    
                    # Call your existing functions
                    discussion_prompt, reading_text, video_text, key_terms, objective, key_concepts = parse_lesson_plan_comprehensive(docx_content)
                    
                    progress_bar.progress(25)
                    st.text("Reading CSV file...")
                    
                    # Process CSV
                    csv_io = io.StringIO(csv_content.decode('utf-8'))
                    rows = list(csv.DictReader(csv_io))
                    
                    progress_bar.progress(50)
                    st.text(f"Processing {len(rows)} submissions...")
                    
                    # Initialize results list
                    results = []
                    failed_submissions = []
                    
                    # Process each submission
                    for idx, row in enumerate(rows):
                        submission_text = row.get("Initial Posts", "").strip()
                        if not submission_text:
                            continue
                        
                        username = row.get("Username", "")
                        student_first_name = username.split()[0] if username else f"Student {idx+1}"
                        
                        reply_columns = [col for col in row.keys() if col.startswith('Reply')]
                        replies = [row[col].strip() for col in reply_columns if row.get(col, '').strip()]
                        
                        try:
                            # Update progress
                            progress = 50 + (idx / len(rows)) * 40
                            progress_bar.progress(progress)
                            
                            # Call your grading function
                            grades = grade_submission_with_retries(
                                strip_tags(submission_text), 
                                reading_text, 
                                key_terms, 
                                discussion_prompt, 
                                student_first_name, 
                                video_text, 
                                replies,
                                api_key  # Pass the API key
                            )
                            
                            # Create result dictionary
                            graded_result = row.copy()
                            graded_result.update(grades)
                            graded_result['num_replies_analyzed'] = str(len(replies))
                            
                            results.append(graded_result)
                            
                        except Exception as e:
                            failed_submissions.append((username, str(e)))
                    
                    progress_bar.progress(90)
                    st.text("Finalizing results...")
                    
                    # Create DataFrame for display
                    if results:
                        df = pd.DataFrame(results)
                        
                        # Show success message
                        st.success(f"Grading complete! Successfully graded {len(results)} submissions.")
                        
                        if failed_submissions:
                            st.warning(f"Failed to grade {len(failed_submissions)} submissions.")
                            with st.expander("View Failed Submissions"):
                                for username, error in failed_submissions:
                                    st.write(f"**{username}**: {error}")
                        
                        # Display results
                        st.subheader("Grading Results")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_score = df['total_score'].astype(float).mean()
                            st.metric("Average Score", f"{avg_score:.2f}/15.0")
                        
                        with col2:
                            high_scorers = len(df[df['total_score'].astype(float) >= 12.0])
                            st.metric("High Scorers (≥12)", high_scorers)
                        
                        with col3:
                            low_scorers = len(df[df['total_score'].astype(float) < 8.0])
                            st.metric("Low Scorers (<8)", low_scorers)
                        
                        # Tabs for different views
                        tab1, tab2, tab3 = st.tabs(["📊 Data View", "📈 Score Distribution", "📝 Sample Feedback"])
                        
                        with tab1:
                            # Display the data
                            st.dataframe(df)
                        
                        with tab2:
                            # Score distribution chart
                            import plotly.express as px
                            fig = px.histogram(
                                df, 
                                x="total_score", 
                                nbins=10,
                                title="Score Distribution",
                                labels={"total_score": "Total Score", "count": "Number of Students"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab3:
                            # Show a few examples of feedback
                            st.subheader("Sample Feedback")
                            sample_size = min(3, len(df))
                            for i in range(sample_size):
                                with st.expander(f"Feedback for {df.iloc[i]['Username']}"):
                                    st.write(df.iloc[i]['feedback'])
                        
                        # Download button
                        csv_output = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Graded CSV",
                            data=csv_output,
                            file_name="graded_discussions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("No submissions were successfully graded.")
                    
                    progress_bar.progress(100)
                    
                except Exception as e:
                    st.error(f"Error processing files: {e}")
                    st.exception(e)
else:
    st.info("Please upload both files to begin grading.")

# Include all your existing functions below
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
            print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)

def robust_api_call(api_url: str, headers: Dict, payload: Dict, timeout: int = 60, max_retries: int = 3, api_key: str = None) -> str:
    """Make API call with retries and proper error handling."""
    def api_call():
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
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
            print(f"Initial JSON parse failed: {e}. Attempting extraction.")

            # Use regex to find the likely JSON object
            json_match = re.search(r'\{.*\}', response_text_clean, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                try:
                    return json.loads(extracted_json)
                except json.JSONDecodeError as e2:
                    raise ValueError(f"Could not parse valid JSON even after extraction: {e2}")

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
    student_first_name: str
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

    prompt_key_combined_feedback = f"{prompt_feedback.strip()} {key_terms_feedback.strip()}"
    prompt_key_formatted = f"PROMPT AND KEY TERMS [{local_scores['prompt_key_score']:.1f}/5.0]: {prompt_key_combined_feedback}."
    video_formatted = f"REFERENCE TO VIDEO [{local_scores['video_score']:.1f}/5.0]: {video_feedback}."
    reading_formatted = f"REFERENCE TO READING [{local_scores['reading_score']:.1f}/5.0]: {reading_feedback}."

    if improvement_areas:
        improvement_focus = f"{student_first_name}, while your work demonstrates strong engagement with the content, focus on improving in the area(s) of: {', '.join(improvement_areas)} to maximize your synthesis of the concepts. {general_feedback_llm}"
    else:
        improvement_focus = f"Outstanding work {student_first_name}! You excelled on all sections of this assignment. {general_feedback_llm}"

    general_formatted = f"GENERAL FEEDBACK: {improvement_focus}"

    final_feedback = '\n'.join([
        prompt_key_formatted,
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
    api_key: str
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
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),',     # "Author, ..." at start of line
            r'([A-Z][a-z]+),\s*Chapter',                # "Author, Chapter"
            r'([A-Z][a-z]+),\s*pp?\.',                  # "Author, p." or "Author, pp."
            r'([A-Z][a-z]+)\s+\((?:19|20)\d{2}\)',     # "Author (Year)"
        ]

        for pattern in author_patterns:
            author_match = re.search(pattern, reading_text, re.MULTILINE)
            if author_match:
                assigned_author = author_match.group(1).split()[-1]  # Get last name
                break

    # Final fallback: look for any capitalized word before a comma in first 100 chars
    if not assigned_author:
        fallback_match = re.search(r'\b([A-Z][a-z]+)\b', reading_text[:100])
        if fallback_match:
            assigned_author = fallback_match.group(1)

    # Final fallback
    if not assigned_author:
        assigned_author = "Unknown Author"
        print(f"⚠️ WARNING: Could not extract author from reading text. Using fallback: '{assigned_author}'")

    reading_info['author_last_name'] = assigned_author
    print(f"📖 DEBUG: Extracted author from lesson plan: '{assigned_author}'")

    # Look for "pages" followed by page numbers
    pages_match = re.search(r'pages\s+([\d.,\s]+)', reading_text, re.IGNORECASE)

    # Extract page numbers and convert to a list
    page_numbers = []
    if pages_match:
        page_str = pages_match.group(1)
        # Handle various page formats like "3.1, 3.2, 3.3, and 3.4" or "10-15" or "10, 12, 14"
        page_numbers = re.findall(r'[\d.]+', page_str)
        # Convert to float for comparison
        page_numbers = [float(p) for p in page_numbers]

    # Set reading info based on extracted data
    if page_numbers:
        # Format page range for feedback
        if len(page_numbers) == 1:
            reading_info['page_range_expected'] = f"page {page_numbers[0]}"
        else:
            reading_info['page_range_expected'] = f"pages {', '.join(map(str, page_numbers))}"
        print(f"DEBUG: Found page numbers: {page_numbers}")
    else:
        reading_info['page_range_expected'] = "unspecified pages"
        print("DEBUG: No page numbers found in reading text.")

    # Debug output for the entire reading text
    print(f"DEBUG: Full reading text: {reading_text}")

    # ----------------------------------------------------
    # D. Analyze citation presence based on updated rules

    highest_max_reading_score = 2.5 # Default Minimum Score
    best_citation_status_msg = f"NO CLEAR REFERENCE TO THE ASSIGNED READING WAS DETECTED. The minimum score of **2.5** applies."
    detected_author = ""

    # Only proceed with citation checking if we have an author
    if reading_info['author_last_name']:
        assigned_author_lower = reading_info['author_last_name'].lower()

        # Check if the author name is present in the submission
        author_present = re.search(r'\b' + re.escape(assigned_author_lower) + r'\b', submission_text.lower())

        # Check if any of the page numbers are present in the submission
        page_present = False
        detected_pages = []
        if page_numbers:
            for page in page_numbers:
                # Convert to string for searching
                page_str = str(page)
                # Look for the page number in the submission
                if re.search(r'\b' + re.escape(page_str) + r'\b', submission_text):
                    page_present = True
                    detected_pages.append(page_str)

        # Debug output for citation detection
        print(f"DEBUG: Citation detection - Author present: {bool(author_present)}, Page present: {page_present}")
        print(f"DEBUG: Detected pages: {detected_pages}")

        # Determine score based on author and page presence
        if author_present and page_present:
            highest_max_reading_score = 5.0
            best_citation_status_msg = f"Both the author ('{assigned_author}') and a relevant page number from the assigned reading were detected. Full credit awarded."
        elif author_present:
            highest_max_reading_score = 4.0
            best_citation_status_msg = f"The author ('{assigned_author}') was mentioned, but no specific page number from the assigned reading was detected. Partial credit awarded."
        elif page_present:
            highest_max_reading_score = 4.5
            best_citation_status_msg = f"A page number from the assigned reading was detected, but the author ('{assigned_author}') was not mentioned. Partial credit awarded."

        # Check for incorrect author if the correct one wasn't found
        if not author_present:
            # Look for any capitalized name that might be an incorrect author
            potential_authors = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', submission_text)
            for potential_author in potential_authors:
                # Skip common words that might be capitalized
                if potential_author.lower() in ['the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by']:
                    continue

                # Check if this could be an author citation
                if len(potential_author) > 3:  # Only consider names longer than 3 characters
                    detected_author = potential_author
                    if page_present:
                        highest_max_reading_score = 4.5
                        best_citation_status_msg = f"A reference to '{potential_author}' with page number {', '.join(detected_pages)} was detected, but this does not match the assigned author ('{assigned_author}'). Partial credit awarded for the correct page reference."
                    else:
                        highest_max_reading_score = 3.0
                        best_citation_status_msg = f"A reference to '{potential_author}' was detected, but this does not match the assigned author ('{assigned_author}'). Partial credit awarded."
                    break
    else:
        # No author found in reading text, so we can't check citations
        highest_max_reading_score = 2.5
        best_citation_status_msg = f"The assigned reading information did not specify an author. The minimum score of **2.5** applies."

    max_reading_score = highest_max_reading_score
    citation_status_msg = best_citation_status_msg

    # --- GENERATE READING FEEDBACK BASED ON SCORE ---
    if max_reading_score == 5.0:
        reading_feedback_local = f"You successfully integrated concepts from the reading and provided a specific citation with a page number from {assigned_author}'s text ({reading_info['page_range_expected']}), earning full credit for this section."
    elif max_reading_score == 4.5:
        if detected_author:
            reading_feedback_local = f"You referenced page number(s) {', '.join(detected_pages)} from the assigned reading, but cited '{detected_author}' instead of the correct author '{assigned_author}'. Partial credit awarded for the correct page reference. Ensure you reference the correct author to earn full credit."
        else:
            reading_feedback_local = f"A page number from the assigned reading was detected, but the author was not mentioned. Include both the author and page number for full credit."
    elif max_reading_score == 4.0:
        reading_feedback_local = f"You mentioned the author ({assigned_author}), demonstrating engagement with the reading. However, you did not provide a specific page number from the assigned reading ({reading_info['page_range_expected']}) as required for higher credit. Include specific page references to earn full credit."
    elif max_reading_score == 3.0:
        reading_feedback_local = f"You referenced '{detected_author}' in your submission, but this does not match the assigned author '{assigned_author}'. Ensure you reference the correct author and include a page number to earn more credit."
    else:  # 2.5
        if assigned_author:
            reading_feedback_local = f"You successfully integrated concepts from the reading, but you did not provide a specific citation with a page number from {assigned_author}'s text ({reading_info['page_range_expected']}) as required for higher credit. You must include the author and a page number from the assigned reading to earn more than the minimum score."
        else:
            reading_feedback_local = f"You successfully integrated concepts from the reading, but you did not provide a specific citation with a page number as required for higher credit. You must include the author and a page number from the assigned reading to earn more than the minimum score."

    # ----------------------------------------------------

    # Local scores - reading score is SET here, not by LLM
    local_scores = {
        'prompt_key_score': 0.0,
        'reading_score': max_reading_score,  # DIRECTLY SET FROM PYTHON DETECTION (5-point scale)
        'video_score': 0.0
    }
    local_feedback = {
        'reading_feedback': reading_feedback_local,  # PYTHON-GENERATED FEEDBACK
        'key_terms_fallback': f"LLM failed to provide key terms feedback. Detected terms: {detected_terms_str}"
    }

    # LLM scoring criteria - reading section is informational only
    llm_scoring_criteria = f"""
SCORING Guidelines for LLM (10 points total - Reading is scored separately):
1. PROMPT ADHERENCE AND KEY TERMS (Minimum 1.5 - 5.0): How well does the student address the prompt AND use key terms? (5.0 Maximum)
    - Prompt adherence: Does the submission fully answer all parts of the discussion question?
    - Key terms usage: Did the student use at least one key term meaningfully?
    - FULL CREDIT (5.0) requires BOTH excellent prompt adherence AND meaningful key term usage.
    - Detected Key Terms to review: "{detected_terms_str}"

2. READING REFERENCE: **This section is scored separately by the system as {max_reading_score:.1f}. Do not provide a reading_score in your response.**
    - Citation Status (for context): {citation_status_msg}

3. VIDEO REFERENCE (Minimum 2.5 - 5.0): How specific and relevant is the use of the assigned video material?
    - Full credit (5.0) requires clear use of concepts demonstrated by specific examples or accurate summaries.
    - **A specific timestamp is NOT required for a 5.0 score.**
"""

    prompt_for_llm = f"""Grade this student discussion submission based ONLY on the following criteria. Reading Reference ({max_reading_score:.1f}) is scored separately by the system.

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
    "prompt_key_score": "5.0",
    "video_score": "5.0",
    "prompt_feedback": "You successfully articulated how involuntary servitude was preserved and connected this theme to present-day issues.",
    "key_terms_feedback": "Your contextual usage of key terms earns full credit, demonstrating clear understanding of the material.",
    "video_feedback": "You clearly referenced the video context regarding convict leasing and the continuation of forced labor, demonstrating a strong grasp of the material.",
    "general_feedback": "Your arguments were well-structured and demonstrated impressive critical thinking."
}}

SUBMISSION TEXT:
{submission_text[:1500]}
"""

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemini-2.5-flash-lite-preview-09-2025",
        "messages": [{"role": "user", "content": prompt_for_llm}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "max_tokens_for_reasoning": 512
    }

    print("📞 Calling API for Prompt/Key Terms and Video grading...")
    grade_response = robust_api_call(api_url, headers, payload, timeout=60, max_retries=3, api_key=api_key)

    llm_results = robust_json_parsing(grade_response, max_retries=2)
    llm_results = recursively_clean(llm_results)

    # 3. Final Score Compilation
    try:
        local_scores['prompt_key_score'] = round_nearest_half(max(1.5, min(5.0, float(llm_results.get("prompt_key_score", 1.5)))))
        local_scores['video_score'] = round_nearest_half(max(2.5, min(5.0, float(llm_results.get("video_score", 2.5)))))
        # reading_score already set from Python detection

    except (ValueError, TypeError):
        print("⚠️ LLM returned invalid score data. Assigning minimums.")
        local_scores['prompt_key_score'] = 1.5
        local_scores['video_score'] = 2.5
        # reading_score remains as set from Python detection

    # Identify lowest scoring component for General Feedback
    weighted_scores = {
        "Prompt Adherence and Key Terms": local_scores['prompt_key_score'] / 5.0,
        "Reading Reference": local_scores['reading_score'] / 5.0,
        "Video Reference": local_scores['video_score'] / 5.0
    }
    sorted_improvement = sorted(weighted_scores.items(), key=lambda item: item[1])
    improvement_areas = [name for name, score in sorted_improvement if score < 1.0 and score > 0.0]

    total = sum(local_scores.values())
    total_score = round_nearest_half(total)

    final_grades = {
        "prompt_key_score": str(local_scores['prompt_key_score']),
        "video_score": str(local_scores['video_score']),
        "reading_score": str(local_scores['reading_score']),
        "total_score": str(total_score),
    }

    final_grades["feedback"] = construct_final_feedback(llm_results, local_scores, local_feedback, improvement_areas, student_first_name)

    return final_grades
