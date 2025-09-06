import traceback
import streamlit as st
import os
import time
import requests
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App Config ---
st.set_page_config(
    page_title="English Rewriter App",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Model Selection ---
model_choice = st.selectbox(
    "Select LLM Provider:",
    ("Gemini", "OpenRouter", "Groq"),
    index=2
)

# --- Dynamically pick API key based on model ---
if model_choice == "Gemini":
    api_key = os.getenv("GOOGLE_API_KEY")
elif model_choice == "OpenRouter":
    api_key = os.getenv("OPENROUTER_API_KEY")
elif model_choice == "Groq":
    api_key = os.getenv("GROQ_API_KEY")
else:
    api_key = None

if not api_key:
    st.warning(f"{model_choice} API Key not found. Please add it in your .env file.")
    st.stop()

# --- API URLs and Models ---
API_URLS = {
    "Gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent",
    "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
    "Groq": "https://api.groq.com/openai/v1/chat/completions"
}

MODEL_IDS = {
    "Gemini": "gemini-1.5-flash-latest",
    "OpenRouter": "openai/gpt-4o-mini",
    "Groq": "llama-3.1-8b-instant"  # Working model
}

# --- API Call Utility ---
def call_api(prompt_text, model_name, api_key, model_type="Gemini", max_retries=5):
    for i in range(max_retries):
        try:
            if model_type == "Gemini":
                # Gemini uses key in URL params, not Authorization header
                payload = {
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generationConfig": {"temperature": 0.5, "maxOutputTokens": 256}
                }
                params = {"key": api_key}
                response = requests.post(API_URLS["Gemini"], headers={"Content-Type": "application/json"},
                                         params=params, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()

            else:  # OpenRouter or Groq
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": 0.5,
                    "max_tokens": 256
                }
                response = requests.post(API_URLS[model_type], headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"{model_type} API error: {e}")
            if i < max_retries - 1:
                delay = 2 ** i
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e

# --- Tone Selection ---
selected_tone = st.selectbox(
    "Select desired tone:",
    ("Standard", "Professional", "Casual", "Formal", "Friendly", "Concise"),
    index=0
)

# --- Session State ---
if 'rewritten_output' not in st.session_state: st.session_state['rewritten_output'] = ""
if 'clear_input_flag' not in st.session_state: st.session_state['clear_input_flag'] = False
if 'input_text_value' not in st.session_state: st.session_state['input_text_value'] = ""

# --- Input Text Area ---
input_text_display_value = "" if st.session_state['clear_input_flag'] else st.session_state['input_text_value']
input_text = st.text_area(
    "Enter your text here:",
    value=input_text_display_value,
    height=150,
    placeholder="e.g., My friend, he no good at english, so I help him write this.",
    key="input_text_area",
    on_change=lambda: st.session_state.update(input_text_value=st.session_state.input_text_area)
)

# --- Buttons ---
col1_buttons, col2_placeholder = st.columns([2,1])
with col1_buttons:
    rewrite_col, clear_col = st.columns([1,1])

    with rewrite_col:
        if st.button("Rewrite Text"):
            if not st.session_state['input_text_value']:
                st.warning("Please enter text to rewrite.")
            else:
                with st.spinner("Rewriting your text..."):
                    try:
                        tone_instruction = ""
                        if selected_tone == "Professional":
                            tone_instruction = "Ensure the tone is professional and sophisticated."
                        elif selected_tone == "Casual":
                            tone_instruction = "Make the tone casual and conversational."
                        elif selected_tone == "Formal":
                            tone_instruction = "Use a highly formal tone."
                        elif selected_tone == "Friendly":
                            tone_instruction = "Adopt a warm, approachable tone."
                        elif selected_tone == "Concise":
                            tone_instruction = "Rewrite to be concise and to-the-point."

                        full_prompt = f"""
Rewrite the following text into grammatically correct, natural, proper English, preserving meaning. {tone_instruction}

Text to rewrite:
"{st.session_state['input_text_value']}"

Rewritten text:
"""

                        response_json = call_api(full_prompt, MODEL_IDS[model_choice], api_key, model_type=model_choice)

                        rewritten_text = ""
                        if model_choice == "Gemini":
                            if response_json and response_json.get("candidates"):
                                rewritten_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                        else:  # OpenRouter or Groq
                            if response_json and "choices" in response_json:
                                rewritten_text = response_json["choices"][0]["message"]["content"]

                        if rewritten_text:
                            st.session_state['rewritten_output'] = rewritten_text.strip()
                        else:
                            st.error(f"No valid response from {model_choice} API.")
                            logger.error(f"Full {model_choice} API response: {response_json}")
                            st.session_state['rewritten_output'] = ""

                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.code(traceback.format_exc())
                        st.session_state['rewritten_output'] = ""

    with clear_col:
        if st.button("Clear Input"):
            st.session_state['clear_input_flag'] = True
            st.session_state['rewritten_output'] = ""

# --- Output Display ---
if st.session_state['rewritten_output']:
    st.subheader(f"Rewritten Text ({selected_tone} Tone, {model_choice})")
    st.text_area(
        "Rewritten Output:",
        value=st.session_state['rewritten_output'],
        height=150,
        key="output_text_area"
    )
    st.info("Highlight and copy the text above to use it.")
else:
    st.write("Your rewritten text will appear here after you click 'Rewrite Text'.")
