import streamlit as st
import os
from chatbot_utilities import Head_Agent

st.set_page_config(page_title="Multi-Agent RAGbot: ML Knowledge Assistant")

st.title("Multi-Agent RAGbot: ML Knowledge Assistant")

# Initialize the Head_Agent
def get_secret_or_env(key):
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return os.getenv(key)

OPENAI_API_KEY = get_secret_or_env("OPENAI_API_KEY")
PINECONE_API_KEY = get_secret_or_env("PINECONE_API_KEY")
PINECONE_INDEX = get_secret_or_env("PINECONE_INDEX_NAME")

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX:
    st.error("Please set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX_NAME in your environment.")
    st.stop()

# Initialize the Head_Agent in session state to persist memory
if "bot" not in st.session_state:
    st.session_state.bot = Head_Agent(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX)
bot = st.session_state.bot

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for storing multiple conversation histories
if "history" not in st.session_state:
    st.session_state.history = []

# Add a sidebar for controls
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Conversation History"):
        # Archive the current conversation before clearing
        if st.session_state.messages:
            st.session_state.history.append(st.session_state.messages)
        st.session_state.messages = []
        # Reset the bot to clear its memory/state
        if "bot" in st.session_state:
            del st.session_state.bot
        st.rerun()

    # Display archived conversations
    if st.session_state.history:
        st.header("Archived Conversations")
        for i, chat in enumerate(st.session_state.history):
            # Create a title for the expander from the first user message
            first_user_message = next((msg['content'] for msg in chat if msg['role'] == 'user'), f"Conversation {i + 1}")
            summary = (first_user_message[:40] + '...') if len(first_user_message) > 40 else first_user_message
            with st.expander(summary):
                for message in chat:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message.get("agent_label"):
                            st.caption(f"Agent Path: {message['agent_label']}")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("agent_label"):
            st.caption(f"Agent Path: {message['agent_label']}")

# UI Mappings for Agent Paths
path_labels = {
    "RETRIEVAL": "📚 RAG (Retrieval)",
    "LLM_ONLY": "🤖 LLM Knowledge",
    "LLM_ONLY_FALLBACK": "🤖 LLM Fallback",
    "REFUSAL_IRRELEVANT": "🚫 Refusal (Not ML Related)",
    "REFUSAL_OBNOXIOUS": "🚫 Refusal (Obnoxious Query)"
}

# Use chat_input to get user input
if user_input := st.chat_input("Type your message here:"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate bot response using handle_one_turn
    with st.chat_message("assistant"):
        result = bot.handle_one_turn(user_input)
        response = result.get("response", "")
        agent_path = result.get("agent_path")

        st.markdown(response)
        label = None
        if agent_path:
            label = path_labels.get(agent_path, agent_path)
            st.caption(f"Source: {label}")

    # Append bot response
    st.session_state.messages.append({"role": "assistant", "content": response, "agent_label": label})
