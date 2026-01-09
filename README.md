# Voice AI Agent

Real-time voice AI agent with ≤2s end-to-end latency using LiveKit, Soniox STT, Groq LLM, and ElevenLabs TTS.

**Includes self-hosted KITT frontend (LiveKit Agents Playground).**

## Quick Start

### Prerequisites
- Docker and Docker Compose
- API keys for: **LiveKit Cloud**, Soniox, Groq, ElevenLabs
- Node.js 20+ and pnpm (for local frontend development)

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (especially LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
```

### 2. Run Full Stack with Docker

```bash
docker-compose up --build
```

This starts:
- **API** (port 8000): Health check and metrics
- **Agent**: LiveKit voice agent worker
- **Frontend** (port 3000): KITT interface

### 3. Access the Application

- **KITT Frontend**: http://localhost:3000
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

---

## Architecture

```
┌─────────────────┐      LiveKit WebRTC      ┌─────────────────────────────┐
│  KITT Frontend  │◄──────────────────────►  │     LiveKit Cloud/Server    │
│   (port 3000)   │                          │                             │
└─────────────────┘                          └──────────────┬──────────────┘
                                                            │
                                             ┌──────────────▼──────────────┐
                                             │      LiveKit Agent          │
                                             │  ┌─────────────────────────┐│
                                             │  │ Soniox → Groq → 11Labs  ││
                                             │  │  STT      LLM     TTS   ││
                                             │  └─────────────────────────┘│
                                             └─────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Latency & cost metrics |
| `/metrics/{session_id}` | GET | Session-specific metrics |

## Local Development

### Backend
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run API server
python -m uvicorn app.main:app --reload

# Run LiveKit agent (separate terminal)
python worker.py start
```

### Frontend
```bash
cd frontend
pnpm install
cp .env.example .env.local
# Edit .env.local with your LiveKit credentials
pnpm run dev
```

Then open http://localhost:3000

## Latency Optimization (≤2s Target)

| Technique | Impact |
|-----------|--------|
| Streaming STT | First words in ~300ms |
| Groq inference | ~200ms to first token |
| Streaming TTS | First audio in ~200ms |
| WebRTC (LiveKit) | Sub-100ms audio transport |

**Typical end-to-end: 700-1500ms**

## Provider Selection & Cost

| Component | Provider | Cost |
|-----------|----------|------|
| **STT** | Soniox | ~$0.12/hr |
| **LLM** | Groq llama-3.3-70b | $0.59/1M input tokens |
| **TTS** | ElevenLabs | ~$0.24/1K chars |

**Per-turn estimate: ~$0.015**

## Environment Variables

```bash
# LiveKit (REQUIRED)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret

# Soniox STT
SONIOX_API_KEY=your_soniox_key

# Groq LLM  
GROQ_API_KEY=your_groq_key

# ElevenLabs TTS
ELEVEN_API_KEY=your_elevenlabs_key
```

## Project Structure

```
attiva/
├── app/                     # Python backend
│   ├── main.py              # FastAPI application
│   ├── agent.py             # LiveKit voice agent
│   ├── tools.py             # LLM tool definitions
│   ├── cost_tracker.py      # Cost estimation
│   ├── latency_tracker.py   # Latency tracking
│   └── session_manager.py   # Session management
├── frontend/                # KITT frontend (Next.js)
│   ├── src/                 # React components
│   ├── Dockerfile           # Frontend container
│   └── .env.example         # Frontend config
├── worker.py                # LiveKit agent worker
├── Dockerfile               # Backend container
├── docker-compose.yml       # Full stack orchestration
├── test_client.html         # Simple WebSocket test client
└── README.md
```

## Failure Scenarios

### STT Service Unavailable
1. Connection timeout after 5s
2. Error logged with structured logging
3. Client receives error message
4. Automatic reconnection with exponential backoff

## Demo Video

[Link to demo video will be added here]

## License

MIT
