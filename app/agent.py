"""
LiveKit Voice Agent Implementation

Implements a real-time voice AI agent using:
- Soniox STT for streaming speech-to-text
- Groq LLM (llama-3.3-70b-versatile) for conversation and tool calling
- ElevenLabs TTS for high-quality text-to-speech
"""

import asyncio
import os
from typing import Optional, Callable, Dict, Any
import structlog

from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents.voice import AgentOutput
from livekit.plugins import soniox, groq, elevenlabs, silero

from .cost_tracker import CostTracker
from .latency_tracker import LatencyTracker
from .tools import AudioPlaybackTool, get_tool_definitions

logger = structlog.get_logger()


class VoiceAgent:
    """
    Real-time voice AI agent with streaming STT, LLM, and TTS.
    
    Designed for ≤2s end-to-end latency with:
    - Streaming partial transcripts
    - First-token LLM streaming
    - Streaming TTS audio output
    """

    def __init__(
        self,
        session_id: str,
        cost_tracker: CostTracker,
        latency_tracker: LatencyTracker,
        on_audio_playback: Optional[Callable] = None,
    ):
        self.session_id = session_id
        self.cost_tracker = cost_tracker
        self.latency_tracker = latency_tracker
        self.audio_tool = AudioPlaybackTool(on_audio_playback)
        
        # Track token counts for cost estimation
        self._current_input_tokens = 0
        self._current_output_tokens = 0
        self._current_audio_duration = 0.0
        self._current_tts_chars = 0
        
        # Initialize providers
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize STT, LLM, and TTS providers."""
        # Soniox STT
        self.stt = soniox.STT(
            api_key=os.getenv("SONIOX_API_KEY"),
        )
        
        # Groq LLM with llama-3.3-70b-versatile
        self.llm = groq.LLM(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7,
        )
        
        # ElevenLabs TTS
        self.tts = elevenlabs.TTS(
            api_key=os.getenv("ELEVEN_API_KEY"),
            voice_id=os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),  # Rachel
            model="eleven_turbo_v2_5",  # Fastest model for low latency
        )
        
        # Voice Activity Detection
        self.vad = silero.VAD.load()

    def get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a helpful, friendly voice assistant. 

Your capabilities:
- Engage in natural conversation on any topic
- Answer questions accurately and concisely
- Use the play_audio tool when you need to play sounds or audio to the user

Guidelines:
- Keep responses concise since this is a voice conversation
- Be natural and conversational
- If you don't know something, say so
- When playing audio, briefly describe what you're playing

You have access to a play_audio tool that can play audio files from URLs."""

    async def create_agent_session(self, room: rtc.Room) -> AgentSession:
        """
        Create and configure an agent session for a room.
        
        Args:
            room: LiveKit room instance
            
        Returns:
            Configured AgentSession ready to handle voice interactions
        """
        logger.info("creating_agent_session", session_id=self.session_id)
        
        session = AgentSession(
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            vad=self.vad,
        )

        # Register event handlers for latency and cost tracking
        self._register_event_handlers(session)

        return session

    def _register_event_handlers(self, session: AgentSession) -> None:
        """Register handlers for tracking latency and cost."""
        
        @session.on("user_started_speaking")
        def on_user_started_speaking():
            logger.debug("user_started_speaking", session_id=self.session_id)
            self.latency_tracker.start_stt()
            self._current_audio_duration = 0.0

        @session.on("user_stopped_speaking")
        def on_user_stopped_speaking():
            logger.debug("user_stopped_speaking", session_id=self.session_id)

        @session.on("user_speech_committed")
        def on_user_speech_committed(text: str):
            logger.info("user_speech_committed", 
                       session_id=self.session_id, 
                       text=text[:100] if text else "")
            self.latency_tracker.end_stt()
            # Estimate audio duration based on speech (rough: ~150 words/min)
            word_count = len(text.split()) if text else 0
            self._current_audio_duration = (word_count / 150) * 60  # seconds
            self.cost_tracker.add_stt_cost(self._current_audio_duration)

        @session.on("agent_started_speaking")  
        def on_agent_started_speaking():
            logger.debug("agent_started_speaking", session_id=self.session_id)
            self.latency_tracker.tts_first_audio()

        @session.on("agent_stopped_speaking")
        def on_agent_stopped_speaking():
            logger.debug("agent_stopped_speaking", session_id=self.session_id)
            self.latency_tracker.end_tts()
            # Finalize the turn
            self.latency_tracker.finish_turn()
            self.cost_tracker.finish_turn()

    async def handle_function_call(
        self, 
        function_name: str, 
        arguments: Dict[str, Any]
    ) -> str:
        """
        Handle function calls from the LLM.
        
        Args:
            function_name: Name of the function to call
            arguments: Function arguments
            
        Returns:
            Result string to return to the LLM
        """
        self.latency_tracker.start_tool()
        
        try:
            if function_name == "play_audio":
                result = await self.audio_tool.play_audio(
                    audio_url=arguments.get("audio_url", ""),
                    description=arguments.get("description", "")
                )
            else:
                result = f"Unknown function: {function_name}"
                logger.warning("unknown_function_call", function_name=function_name)
        except Exception as e:
            result = f"Error executing {function_name}: {str(e)}"
            logger.exception("function_call_error", 
                           function_name=function_name,
                           error=str(e))
        finally:
            self.latency_tracker.end_tool()
        
        return result


async def run_agent(ctx: agents.JobContext) -> None:
    """
    Entry point for the LiveKit agent worker.
    
    This function is called by the LiveKit agents framework when
    a new room session starts.
    """
    logger.info("agent_job_started", room_name=ctx.room.name)
    
    # Create trackers for this session
    session_id = ctx.room.name
    cost_tracker = CostTracker(session_id)
    latency_tracker = LatencyTracker(session_id)
    
    # Create the voice agent
    agent = VoiceAgent(
        session_id=session_id,
        cost_tracker=cost_tracker,
        latency_tracker=latency_tracker,
    )
    
    # Create agent session
    session = await agent.create_agent_session(ctx.room)
    
    # Start the agent
    await session.start(
        room=ctx.room,
        agent=Agent(instructions=agent.get_system_prompt()),
        room_input_options=RoomInputOptions(
            # Process audio from all participants
        ),
    )
    
    # Keep running until the context is done
    await ctx.connect()
    
    logger.info("agent_job_completed",
               room_name=ctx.room.name,
               total_cost=cost_tracker.total_cost,
               turn_count=latency_tracker.turn_count)


def create_worker() -> agents.Worker:
    """Create and configure the LiveKit agents worker."""
    return agents.Worker(
        request_handler=run_agent,
    )
