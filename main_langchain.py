import os
import time
import logging
import traceback
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import ClassVar, Optional, List
from pydantic import PrivateAttr
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM

# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit page config ---
st.set_page_config(
    page_title="English Rewriter App",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Model selection ---
model_choice = st.selectbox(
    "Select LLM Provider:",
    ("Gemini", "OpenRouter", "Groq"),
    index=2
)

# --- Get API key dynamically ---
api_key = {
    "Gemini": os.getenv("GOOGLE_API_KEY"),
    "OpenRouter": os.getenv("OPENROUTER_API_KEY"),
    "Groq": os.getenv("GROQ_API_KEY")
}.get(model_choice)

if not api_key:
    st.warning(f"{model_choice} API Key not found. Please add it in your .env file.")
    st.stop()

# --- Tone selection ---
selected_tone = st.selectbox(
    "Select desired tone:",
    ("Standard", "Professional", "Casual", "Formal", "Friendly", "Concise"),
    index=0
)

# --- Session state defaults ---
if 'rewritten_output' not in st.session_state:
    st.session_state['rewritten_output'] = ""
if 'clear_input_flag' not in st.session_state:
    st.session_state['clear_input_flag'] = False
if 'input_text_value' not in st.session_state:
    st.session_state['input_text_value'] = ""
if 'last_raw_response' not in st.session_state:
    st.session_state['last_raw_response'] = "No response yet."

# --- Input text area ---
input_text_display_value = "" if st.session_state['clear_input_flag'] else st.session_state['input_text_value']
input_text = st.text_area(
    "Enter your text here:",
    value=input_text_display_value,
    height=150,
    placeholder="e.g., My friend, he no good at english, so I help him write this.",
    key="input_text_area",
    on_change=lambda: st.session_state.update(input_text_value=st.session_state.input_text_area)
)

# --- Tone instructions ---
TONE_INSTRUCTIONS = {
    "Standard": "",
    "Professional": "Ensure the tone is professional and sophisticated.",
    "Casual": "Make the tone casual and conversational.",
    "Formal": "Use a highly formal tone.",
    "Friendly": "Adopt a warm, approachable tone.",
    "Concise": "Rewrite to be concise and to-the-point."
}

# --- Prompt template ---
prompt_template = PromptTemplate(
    input_variables=["text", "tone_instruction"],
    template="""
Rewrite the following text into grammatically correct, natural, proper English, preserving meaning.
{tone_instruction}

Text to rewrite:
"{text}"

Rewritten text:
"""
)

# --- Gemini LLM ---
class GeminiLLM(LLM):
    api_url: ClassVar[str] = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

    _api_key: str = PrivateAttr()
    _temperature: float = PrivateAttr()
    _max_output_tokens: int = PrivateAttr()

    def __init__(self, api_key: str, temperature: float = 0.5, max_output_tokens: int = 256):
        super().__init__()
        self._api_key = api_key
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_output_tokens
            }
        }
        params = {"key": self._api_key}
        response = requests.post(self.api_url, json=payload, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("candidates"):
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return ""

    def _agenerate(self, prompts, stop=None):
        raise NotImplementedError("Async not implemented for GeminiLLM")

# --- OpenRouter / Groq LLM wrapper ---
class CustomHTTPLLM(LLM):
    _api_key: str = PrivateAttr()
    _api_url: str = PrivateAttr()
    _model_name: str = PrivateAttr()
    _temperature: float = PrivateAttr()
    _max_tokens: int = PrivateAttr()

    def __init__(self, api_key: str, api_url: str, model_name: str, temperature: float = 0.5, max_tokens: int = 256):
        super().__init__()
        self._api_key = api_key
        self._api_url = api_url
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        return "http_llm"

    def _call(self, prompt: str, stop=None) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens
        }
        response = requests.post(self._api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        return ""

    def _agenerate(self, prompts, stop=None):
        raise NotImplementedError("Async not implemented")

# --- LLM factory ---
def get_llm(model_choice: str, api_key: str):
    if model_choice == "Gemini":
        return GeminiLLM(api_key)
    elif model_choice == "OpenRouter":
        return CustomHTTPLLM(
            api_key=api_key,
            api_url="https://openrouter.ai/api/v1/chat/completions",
            model_name="openai/gpt-4o-mini"
        )
    else:  # Groq
        return CustomHTTPLLM(
            api_key=api_key,
            api_url="https://api.groq.com/openai/v1/chat/completions",
            model_name="llama-3.1-8b-instant"
        )

# --- Rewrite function with retry ---
def rewrite_text_with_retry(text: str, tone_instruction: str, retries: int = 5) -> str:
    llm = get_llm(model_choice, api_key)
    chain = LLMChain(prompt=prompt_template, llm=llm)
    for attempt in range(retries):
        try:
            result = chain.run(text=text, tone_instruction=tone_instruction)
            st.session_state['last_raw_response'] = result
            return result.strip()
        except Exception as e:
            logger.error(f"{model_choice} LLM error: {e}")
            st.session_state['last_raw_response'] = f"Error: {e}\n{traceback.format_exc()}"
            if attempt < retries - 1:
                delay = 2 ** attempt
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error(f"Failed after {retries} attempts: {e}")
                return ""

# --- Buttons & actions ---
col1_buttons, col2_placeholder = st.columns([2,1])
with col1_buttons:
    rewrite_col, clear_col = st.columns([1,1])

    with rewrite_col:
        if st.button("Rewrite Text"):
            if not st.session_state['input_text_value']:
                st.warning("Please enter text to rewrite.")
            else:
                with st.spinner("Rewriting your text..."):
                    tone_instruction = TONE_INSTRUCTIONS.get(selected_tone, "")
                    rewritten_text = rewrite_text_with_retry(
                        st.session_state['input_text_value'],
                        tone_instruction
                    )
                    st.session_state['rewritten_output'] = rewritten_text

    with clear_col:
        if st.button("Clear Input"):
            st.session_state['clear_input_flag'] = True
            st.session_state['rewritten_output'] = ""
            st.session_state['last_raw_response'] = "No response yet."

# --- Output display ---
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

# --- Debug Sidebar ---
with st.sidebar:
    st.subheader("Debug / Raw API Response")
    raw_resp = st.session_state.get('last_raw_response', "No response yet.")
    st.text_area(
        "Last LLM Response:",
        value=raw_resp,
        height=300
    )
