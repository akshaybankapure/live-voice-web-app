"""
FastAPI Main Application for Voice AI Agent

Provides:
- WebSocket endpoint /ws/talk for bidirectional audio streaming
- GET /health for health checks
- GET /metrics for latency and cost metrics
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from .session_manager import session_manager, Session
from .cost_tracker import CostTracker
from .latency_tracker import LatencyTracker
from .tools import AudioPlaybackTool, SAMPLE_AUDIO_URLS

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("application_startup", 
                host=os.getenv("HOST", "0.0.0.0"),
                port=os.getenv("PORT", "8000"))
    yield
    # Cleanup on shutdown
    logger.info("application_shutdown",
                active_sessions=session_manager.active_session_count)


# Create FastAPI app
app = FastAPI(
    title="Voice AI Agent",
    description="Real-time voice AI agent with LiveKit, Soniox STT, Groq LLM, and ElevenLabs TTS",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get metrics including:
    - Active sessions count
    - Average end-to-end latency
    - Cost per turn and per conversation breakdown
    """
    metrics = session_manager.get_aggregate_metrics()
    return metrics


@app.get("/metrics/{session_id}")
async def get_session_metrics(session_id: str) -> Dict[str, Any]:
    """Get detailed metrics for a specific session."""
    details = session_manager.get_session_details(session_id)
    if not details:
        raise HTTPException(status_code=404, detail="Session not found")
    return details


@app.websocket("/ws/talk")
async def websocket_talk(websocket: WebSocket):
    """
    Bidirectional WebSocket for voice conversation.
    
    Protocol:
    - Client sends: Binary audio frames (PCM 16-bit, 16kHz, mono)
    - Client sends: JSON control messages {"type": "start"|"stop"|"cancel"}
    - Server sends: JSON transcript updates {"type": "transcript", "text": "...", "is_final": bool}
    - Server sends: Binary audio frames (TTS output)
    - Server sends: JSON metadata {"type": "metadata", "latency": {...}, "cost": {...}}
    """
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    session = await session_manager.create_session(session_id)
    
    logger.info("websocket_connected", session_id=session_id)
    
    # Send session info to client
    await websocket.send_json({
        "type": "session_start",
        "session_id": session_id,
    })
    
    try:
        await handle_voice_session(websocket, session)
    except WebSocketDisconnect:
        logger.info("websocket_disconnected", session_id=session_id)
    except Exception as e:
        logger.exception("websocket_error", session_id=session_id, error=str(e))
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except:
            pass
    finally:
        # Clean up session
        await session_manager.remove_session(session_id)
        logger.info("session_cleanup_complete", session_id=session_id)


async def handle_voice_session(websocket: WebSocket, session: Session):
    """
    Main voice session handler.
    
    This simulates the voice pipeline without requiring actual LiveKit connection.
    For production, this would integrate with the LiveKit agent worker.
    """
    cost_tracker = session.cost_tracker
    latency_tracker = session.latency_tracker
    
    # Audio playback callback
    async def play_audio_callback(audio_url: str) -> bool:
        """Send audio playback command to client."""
        try:
            await websocket.send_json({
                "type": "play_audio",
                "url": audio_url,
            })
            return True
        except Exception as e:
            logger.error("play_audio_send_failed", error=str(e))
            return False
    
    audio_tool = AudioPlaybackTool(play_audio_callback)
    
    # Simulated conversation state
    conversation_history = []
    
    while True:
        try:
            # Receive message from client
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
            
            # Handle binary audio data
            if "bytes" in message:
                audio_data = message["bytes"]
                await process_audio_frame(
                    websocket, 
                    session, 
                    audio_data,
                    audio_tool,
                    conversation_history
                )
            
            # Handle JSON control messages
            elif "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")
                
                if msg_type == "text_input":
                    # Handle text input for testing (simulates STT output)
                    text = data.get("text", "")
                    await process_text_input(
                        websocket,
                        session,
                        text,
                        audio_tool,
                        conversation_history
                    )
                
                elif msg_type == "stop":
                    logger.info("client_requested_stop", session_id=session.session_id)
                    break
                
                elif msg_type == "cancel":
                    logger.info("client_cancelled_turn", session_id=session.session_id)
                    # Cancel current processing
                
        except json.JSONDecodeError:
            logger.warning("invalid_json_message", session_id=session.session_id)
        except Exception as e:
            logger.exception("message_processing_error", 
                           session_id=session.session_id, 
                           error=str(e))
            raise


async def process_audio_frame(
    websocket: WebSocket,
    session: Session,
    audio_data: bytes,
    audio_tool: AudioPlaybackTool,
    conversation_history: list,
):
    """
    Process incoming audio frame through the voice pipeline.
    
    In a real implementation, this would:
    1. Buffer audio frames
    2. Send to Soniox STT for transcription
    3. Process transcript through Groq LLM
    4. Generate response via ElevenLabs TTS
    5. Stream audio back to client
    
    For this demo, we simulate the pipeline with mock responses.
    """
    # For demo: accumulate audio and simulate transcription
    # Real implementation would stream to Soniox
    pass


async def process_text_input(
    websocket: WebSocket,
    session: Session,
    text: str,
    audio_tool: AudioPlaybackTool,
    conversation_history: list,
):
    """
    Process text input through LLM and TTS.
    
    This allows testing the pipeline without actual audio input.
    """
    if not text.strip():
        return
    
    cost_tracker = session.cost_tracker
    latency_tracker = session.latency_tracker
    
    # Start tracking this turn
    latency_tracker.start_stt()
    
    # Simulate STT completion (instant for text input)
    latency_tracker.stt_first_result()
    latency_tracker.end_stt()
    
    # Estimate STT cost (based on word count)
    word_count = len(text.split())
    audio_duration = (word_count / 150) * 60  # ~150 words/min
    cost_tracker.add_stt_cost(audio_duration)
    
    # Send transcript to client
    await websocket.send_json({
        "type": "transcript",
        "text": text,
        "is_final": True,
    })
    
    # Start LLM processing
    latency_tracker.start_llm()
    
    # Build conversation context
    conversation_history.append({"role": "user", "content": text})
    
    # Simulate LLM response (in production, call Groq API)
    response = await simulate_llm_response(text, conversation_history)
    
    latency_tracker.llm_first_token()
    
    # Check for tool calls
    if "play_audio" in text.lower() or "play sound" in text.lower():
        latency_tracker.start_tool()
        await audio_tool.play_audio(
            SAMPLE_AUDIO_URLS["notification"],
            "notification sound"
        )
        latency_tracker.end_tool()
        response = "I've played a notification sound for you!"
    
    latency_tracker.end_llm()
    
    # Estimate LLM cost
    input_tokens = len(text.split()) * 2  # Rough estimate
    output_tokens = len(response.split()) * 2
    cost_tracker.add_llm_cost(input_tokens, output_tokens)
    
    conversation_history.append({"role": "assistant", "content": response})
    
    # Start TTS
    latency_tracker.start_tts()
    
    # Send response text
    await websocket.send_json({
        "type": "response",
        "text": response,
    })
    
    latency_tracker.tts_first_audio()
    
    # Estimate TTS cost
    cost_tracker.add_tts_cost(len(response))
    
    # In production, we'd stream TTS audio here
    # For demo, just simulate completion
    await asyncio.sleep(0.1)  # Simulate TTS generation
    
    latency_tracker.end_tts()
    
    # Finish turn
    turn_latency = latency_tracker.finish_turn()
    turn_cost = cost_tracker.finish_turn()
    
    # Send turn metadata to client
    await websocket.send_json({
        "type": "turn_complete",
        "latency": turn_latency.to_dict(),
        "cost": turn_cost.to_dict(),
    })


async def simulate_llm_response(text: str, history: list) -> str:
    """
    Simulate LLM response for demo purposes.
    
    In production, this calls the Groq API.
    """
    # Simple response simulation
    text_lower = text.lower()
    
    if "hello" in text_lower or "hi" in text_lower:
        return "Hello! I'm your voice AI assistant. How can I help you today?"
    elif "how are you" in text_lower:
        return "I'm doing great, thank you for asking! I'm here and ready to help."
    elif "weather" in text_lower:
        return "I don't have access to real-time weather data, but I'd recommend checking a weather service for the most accurate forecast."
    elif "help" in text_lower:
        return "I can help with various tasks! You can ask me questions, have a conversation, or ask me to play audio by saying 'play sound'."
    elif "bye" in text_lower or "goodbye" in text_lower:
        return "Goodbye! It was nice talking with you. Have a great day!"
    else:
        return f"I heard you say: '{text}'. I'm a demo voice assistant. In production, I would use Groq's LLM for intelligent responses."


def main():
    """Run the FastAPI application."""
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
