# Visa Assistant - Quick Start Guide

## Prerequisites
- Python 3.8+
- Git

## Installation

1. **Clone/Download the project**
   ```bash
   cd "/path/to/Visa Assistant"
   ```

2. **Set up your API key**

   Copy the example env file and add your key:
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and set:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   > **Never commit `.env` to version control.** It is already listed in `.gitignore`.

3. **Install Python dependencies**
   ```bash
   pip install fastapi uvicorn streamlit
   pip install langchain-google-genai
   pip install faiss-cpu sentence-transformers
   pip install requests pydantic python-dotenv
   ```

## Running the System

### Step 1: Start the Backend Server
```bash
cd Backend
python main.py
```
- Server starts on: `http://localhost:8002`
- Wait for "Agent ready – accepting requests" message

### Step 2: Start the Frontend (New Terminal)
```bash
cd Frontend
streamlit run app.py --server.port 8501
```
- Frontend opens at: `http://localhost:8501`
- Browser should open automatically

## Testing

### Test Backend API
```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "I want to travel to USA", "session_id": "test"}'
```

### Test Health Check
```bash
curl http://localhost:8002/health
```

## Usage

1. Open browser to `http://localhost:8501`
2. Type your visa question (e.g., "I want to travel to USA from India")
3. Follow the conversation flow:
   - Destination country → Source country → Purpose → Duration → Budget
4. Get detailed visa information and requirements

## Supported Countries
- USA
- Singapore  
- Qatar

## Stopping the System
- Backend: Press `Ctrl+C`
- Frontend: Press `Ctrl+C`

## Troubleshooting

**Backend won't start:**
- Check if port 8002 is free: `lsof -ti:8002`
- Kill existing process: `lsof -ti:8002 | xargs kill -9`

**Frontend won't start:**
- Check if port 8501 is free: `lsof -ti:8501`
- Kill existing process: `lsof -ti:8501 | xargs kill -9`

**No response from backend:**
- Verify backend is running: `curl http://localhost:8002/health`
- Check terminal logs for errors