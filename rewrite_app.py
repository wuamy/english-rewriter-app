import traceback
import streamlit as st
import os
import json
import time
import requests
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App Configuration ---
st.set_page_config(
    page_title="English Rewriter App (Gemini)",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Gemini API Key Setup ---
st.sidebar.title("Gemini API Key")
st.sidebar.markdown(
    """
    To use this app, you need a Gemini API key.
    1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
    2. Create a new API key.
    3. Save it in a `.env` file in your project directory
       as `GOOGLE_API_KEY=YOUR_API_KEY_HERE`.
    """
)

gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not gemini_api_key:
    st.warning(
        "Gemini API Key not found. "
        "Please create a `.env` file in your project directory "
        "with `GOOGLE_API_KEY=YOUR_API_KEY_HERE`."
    )
    st.stop()

# --- Gemini API Constants ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
MODEL_NAME = "gemini-1.5-flash-latest"

# --- Utility for Gemini API Call with Exponential Backoff ---
def call_gemini_api(prompt_text, api_key, max_retries=5, initial_delay=1):
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 256,
        },
    }
    params = {'key': api_key}

    for i in range(max_retries):
        try:
            response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} for Gemini API call: {e.response.text}")
            if e.response.status_code == 429 and i < max_retries - 1:
                delay = initial_delay * (2 ** i)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during Gemini API call: {e}")
            if i < max_retries - 1:
                delay = initial_delay * (2 ** i)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error during Gemini API call: {e}")
            if i < max_retries - 1:
                delay = initial_delay * (2 ** i)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred during Gemini API call: {e}")
            raise e
    return None

# --- Streamlit App UI ---
st.title("✍️ TextTidy")
st.markdown("Enter a sentence or paragraph below, and I'll rewrite it into proper English.")

# Tone selection
selected_tone = st.selectbox(
    "Select desired tone:",
    ("Standard", "Professional", "Casual", "Formal", "Friendly", "Concise"),
    index=0
)

# Initialize session state for output and clear flag if they don't exist
if 'rewritten_output' not in st.session_state:
    st.session_state['rewritten_output'] = ""
if 'clear_input_flag' not in st.session_state:
    st.session_state['clear_input_flag'] = False
if 'input_text_value' not in st.session_state: # To manage the actual input text value
    st.session_state['input_text_value'] = ""

# Input text area (cleared by flag on next rerun)
# The value is explicitly set to empty if clear_input_flag is True, then the flag is reset.
input_text_display_value = ""
if st.session_state['clear_input_flag']:
    st.session_state['input_text_value'] = "" # Also clear the stored value
    st.session_state['clear_input_flag'] = False # Reset flag immediately
else:
    input_text_display_value = st.session_state['input_text_value'] # Use stored value

input_text = st.text_area(
    "Enter your text here:",
    value=input_text_display_value, # Pass the dynamically set value
    height=150,
    placeholder="e.g., My friend, he no good at english, so I help him write this.",
    key="input_text_area",
    on_change=lambda: st.session_state.update(input_text_value=st.session_state.input_text_area) # Update session state on change
)

# --- Buttons for Rewrite and Clear (Moved below the text area) ---
col1_buttons, col2_placeholder = st.columns([2, 1]) # Adjusted column ratio for buttons

with col1_buttons:
    rewrite_button_col, clear_button_col = st.columns([1, 1]) # Nested columns for close buttons

    with rewrite_button_col:
        if st.button("Rewrite Text"):
            if not gemini_api_key:
                st.warning("Gemini API Key not found. Please set it in your `.env` file.")
            elif st.session_state['input_text_value']: # Use the session state value for input
                with st.spinner("Rewriting your text..."):
                    try:
                        tone_instruction = ""
                        if selected_tone == "Professional":
                            tone_instruction = "Ensure the tone is professional, sophisticated, and appropriate for business or academic contexts."
                        elif selected_tone == "Casual":
                            tone_instruction = "Make the tone casual, informal, and conversational, suitable for friends or relaxed settings."
                        elif selected_tone == "Formal":
                            tone_instruction = "Use a highly formal tone, suitable for official documents or serious academic writing."
                        elif selected_tone == "Friendly":
                            tone_instruction = "Adopt a warm, approachable, and encouraging tone."
                        elif selected_tone == "Concise":
                            tone_instruction = "Rewrite the text to be as brief and to-the-point as possible, without losing essential meaning."

                        full_prompt = f"""
                        You are an expert English language editor. Your task is to rewrite the following text into grammatically correct, natural, and proper English, while preserving its original meaning. {tone_instruction}

                        Text to rewrite:
                        "{st.session_state['input_text_value']}" # Use session state for prompt

                        Rewritten text:
                        """

                        response_json = call_gemini_api(full_prompt, gemini_api_key)

                        if response_json and response_json.get("candidates"):
                            rewritten_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                            st.session_state['rewritten_output'] = rewritten_text.strip()
                        else:
                            st.error("No valid response received from Gemini API. Please check your API key and input.")
                            logger.error(f"Full Gemini API response: {response_json}")
                            st.session_state['rewritten_output'] = ""

                    except Exception as e:
                        st.error(f"An error occurred during rewriting: {e}")
                        st.error("Here's the detailed error traceback:")
                        st.code(traceback.format_exc())
                        st.info("Please check your Gemini API key's validity, internet connection, and the input text length.")
                        st.session_state['rewritten_output'] = ""
            else:
                st.warning("Please enter some text to rewrite.")

    with clear_button_col:
        if st.button("Clear Input", key="clear_button"):
            st.session_state['clear_input_flag'] = True
            st.session_state['rewritten_output'] = ""


# --- Output Display (now hidden until output is available) ---
if st.session_state['rewritten_output']: # Only display if there's output
    st.subheader(f"Rewritten Text ({selected_tone} Tone):")
    st.text_area(
        "Rewritten Output:",
        value=st.session_state['rewritten_output'],
        height=150,
        key="output_text_area"
    )
    st.info("To copy, simply highlight the text above and press Ctrl+C (Windows/Linux) or Cmd+C (macOS).")
else:
    st.write("Your rewritten text will appear here after you click 'Rewrite Text'.")

