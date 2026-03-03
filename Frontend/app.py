import streamlit as st
import requests
import uuid

st.set_page_config(
    page_title="Visa Assistant",
    page_icon="✈️",
    layout="wide"
)

# API endpoints
CHAT_API_URL = "http://localhost:8002/chat"
HEALTH_URL = "http://localhost:8002/health"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    # Generate a unique session ID for this chat session
    st.session_state.session_id = str(uuid.uuid4())

def send_message_to_api(message):
    """Send message to backend API using the new progressive context collection endpoint"""
    
    payload = {
        "question": message,
        "session_id": st.session_state.session_id
    }
    
    try:
        response = requests.post(CHAT_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("answer", "Sorry, I couldn't get a response.")
        else:
            return f"⚠️ Backend returned error {response.status_code}. Please check if the server is running properly."
    except requests.exceptions.ConnectionError:
        return "❌ **Cannot connect to backend server.** Please ensure the backend is running on http://localhost:8002"
    except requests.exceptions.Timeout:
        return "⏱️ **Request timed out.** The backend might be processing a complex query. Please try again."
    except Exception as e:
        return f"❌ **Unexpected error:** {str(e)}"

def clear_chat():
    """Clear the current chat and start fresh"""
    st.session_state.messages = []
    # Generate a new session ID for a fresh start
    st.session_state.session_id = str(uuid.uuid4())

# Main UI
st.title("💬 Visa Assistant")

# Check backend health (non-blocking)
try:
    health_resp = requests.get(HEALTH_URL, timeout=2)
    backend_ready = health_resp.status_code == 200 and health_resp.json().get("ready", False)
except Exception:
    backend_ready = False

# Show warning but don't stop the app
if not backend_ready:
    st.error("🔌 Backend connection issue. Please ensure the backend server is running on http://localhost:8002")
    st.info("💡 **To start the backend**: Open terminal → Navigate to Backend folder → Run `python main.py`")
    # Don't stop the app, just show the interface with warnings

# Sidebar with controls
with st.sidebar:
    st.header("🔧 Chat Controls")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_chat()
        st.success("✅ Chat cleared! Start a new conversation below.")
    
    st.divider()
    
    # Connection status
    if backend_ready:
        st.success("🟢 Backend Connected")
    else:
        st.error("🔴 Backend Disconnected")
    
    st.divider()
    
    # Session info
    st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
    
    st.divider()
    
    # Instructions
    with st.expander("ℹ️ How it works", expanded=True):
        st.markdown("""
        **Simple Chat Interface:**
        - Just start chatting! No need to pre-select travel details
        - Ask visa questions naturally: *"Do I need a visa from India to USA?"*
        - The assistant will ask for missing information as needed
        - Context is maintained throughout the conversation
        - Ask general questions too - the bot remembers your travel context
        """)

# Initial welcome message if no conversation started
if not st.session_state.messages:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi there! 👋 I'm your visa assistant. I can help you with visa requirements for travel between different countries. Just ask me anything - like 'Do I need a visa from India to USA?' or 'What documents do I need for tourism to Singapore?' I'll collect the necessary travel details as we chat."
    }]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask me about visa requirements, travel documents, or anything else...")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input
    })
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get AI response with proper error handling
    with st.chat_message("assistant"):
        if backend_ready:
            with st.spinner("🤔 Thinking..."):
                ai_response = send_message_to_api(user_input)
        else:
            ai_response = "❌ **Backend not available.** Please start the backend server first."
        
        st.write(ai_response)
    
    # Add AI response to messages
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })
