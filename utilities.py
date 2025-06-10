import os
import json
import datetime
import re

CHAT_DIR = "chat_sessions"
LOG_DIR = "chat_logs"
TEMP_DIR = "temp"

os.makedirs(CHAT_DIR, exist_ok=True)

def load_sessions():
    """List all saved session IDs (filenames without .json)"""
    return [f[:-5] for f in os.listdir(CHAT_DIR) if f.endswith(".json")]

def save_session(session_id, messages):
    """Save messages to a .json file under session_id"""
    with open(os.path.join(CHAT_DIR, f"{session_id}.json"), "w") as f:
        json.dump(messages, f, indent=2)

def load_session(session_id):
    """Load a chat session by session_id"""
    with open(os.path.join(CHAT_DIR, f"{session_id}.json"), "r") as f:
        return json.load(f)

def clean_llm_output(text):
    """Strip <think>...</think> blocks from LLM output"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def log_chat(user_input: str, full_response: str, log_dir: str = LOG_DIR):
    """Log full chat message to timestamped .txt file"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(log_dir, f"chat_{timestamp}.txt")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"User: {user_input}\n")
        f.write(f"Assistant (full): {full_response}\n")
        f.write("-" * 40 + "\n")

def clear_temp_files():
    """Delete all files in the temp/ directory"""
    if os.path.exists(TEMP_DIR):
        for f in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
