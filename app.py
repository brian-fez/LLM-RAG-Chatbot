import streamlit as st
import datetime
import os
import uuid
from vectors import EmbeddingsManager
from chatbot import ChatbotManager
from utilities import (
    load_sessions,
    save_session,
    load_session,
    clean_llm_output,
    log_chat,
    clear_temp_files
)

# Initialize session state
if "chatbot_manager" not in st.session_state:
    st.session_state.chatbot_manager = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Page config
st.set_page_config(page_title="Document RAG Chat", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("### 💬 Chat Sessions")

    # Start new session
    new_name = st.text_input("Name your new session", placeholder="e.g. invoice-2025 or chat with law.pdf")
    if st.button("➕ Start New Chat"):
        session_id = new_name if new_name else datetime.datetime.now().strftime("session-%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
        st.session_state.session_id = session_id
        st.session_state.messages = []

        clear_temp_files()
        st.session_state.pop("uploaded_file", None)

        st.rerun()

    # Load previous sessions
    st.markdown("#### 📂 Your Chats")
    sessions = load_sessions()

    if sessions:
        for session in sessions:
            if st.button(session):
                st.session_state.session_id = session
                st.session_state.messages = load_session(session)
                st.rerun()
    else:
        st.markdown("*No saved sessions yet.*")
    st.markdown("---")
    st.markdown("© 2025 Document RAG. No rights reserved. 🛡️")

# Main UI
st.title("🤖 Document RAG Chat")
st.markdown("Upload a PDF to chat with it using retrieval-augmented generation (RAG).")

uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_pdf_path = os.path.join("temp", f"{uuid.uuid4()}.pdf")
    os.makedirs("temp", exist_ok=True)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.spinner("🔄 Creating Embeddings and Initializing Chat..."):
            embed = EmbeddingsManager()
            embed.create_embeddings(temp_pdf_path)
            chatbot = ChatbotManager()
            chatbot.set_vector_store(embed.vector_store)
            st.session_state.chatbot_manager = chatbot
        st.success("✅ Embeddings created and chatbot ready!")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Chat UI
if st.session_state.chatbot_manager:
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        if st.session_state.session_id:
            save_session(st.session_state.session_id, [])
        st.rerun()

    for msg in st.session_state.messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            st.chat_message(msg["role"]).markdown(clean_llm_output(msg["content"]))

    if user_input := st.chat_input("Type your message here..."):
        user_msg = {"role": "user", "content": user_input}
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append(user_msg)

        with st.spinner("🤖 Responding..."):
            full_response = st.session_state.chatbot_manager.get_response(user_input)
            visible_response = clean_llm_output(full_response)
            st.chat_message("assistant").markdown(visible_response)
            log_chat(user_input, full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            if st.session_state.session_id:
                save_session(st.session_state.session_id, st.session_state.messages)
