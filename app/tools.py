"""
Tools for Voice AI Agent

Defines callable tools for the LLM agent:
- play_audio: Triggers audio playback to the client
"""

from typing import Callable, Optional
import asyncio
import structlog

logger = structlog.get_logger()


class AudioPlaybackTool:
    """
    Tool for triggering audio playback on the client.
    
    This tool is called by the LLM when it wants to play audio
    (e.g., a sound effect, notification, or pre-recorded message).
    """

    def __init__(self, playback_callback: Optional[Callable] = None):
        """
        Initialize the audio playback tool.
        
        Args:
            playback_callback: Async function to call when audio should be played.
                              Signature: async def callback(audio_url: str) -> bool
        """
        self._playback_callback = playback_callback

    def set_callback(self, callback: Callable) -> None:
        """Set the playback callback function."""
        self._playback_callback = callback

    async def play_audio(self, audio_url: str, description: str = "") -> str:
        """
        Play audio from a URL to the client.
        
        This is designed to be registered as an LLM-callable tool.
        
        Args:
            audio_url: URL of the audio file to play
            description: Optional description of what the audio is
            
        Returns:
            Status message indicating success or failure
        """
        logger.info("play_audio_tool_called",
                   audio_url=audio_url,
                   description=description)

        if not self._playback_callback:
            logger.warning("play_audio_no_callback")
            return "Audio playback requested but no playback handler is configured."

        try:
            # Call the registered callback to actually play the audio
            success = await self._playback_callback(audio_url)
            
            if success:
                message = f"Successfully started playing audio: {description or audio_url}"
                logger.info("play_audio_success", audio_url=audio_url)
            else:
                message = f"Failed to play audio: {audio_url}"
                logger.error("play_audio_failed", audio_url=audio_url)
            
            return message
            
        except Exception as e:
            error_msg = f"Error playing audio: {str(e)}"
            logger.exception("play_audio_error", audio_url=audio_url, error=str(e))
            return error_msg


def get_tool_definitions() -> list:
    """
    Get tool definitions for the LLM in OpenAI function calling format.
    
    Returns list of tool definitions compatible with Groq/OpenAI API.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "play_audio",
                "description": "Play an audio file to the user. Use this to play sound effects, notifications, music, or pre-recorded messages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "audio_url": {
                            "type": "string",
                            "description": "The URL of the audio file to play. Must be a valid, accessible audio URL (MP3, WAV, OGG, etc.)"
                        },
                        "description": {
                            "type": "string",
                            "description": "A brief description of what audio is being played (e.g., 'notification sound', 'welcome message')"
                        }
                    },
                    "required": ["audio_url"]
                }
            }
        }
    ]


# Example additional tools can be added here
SAMPLE_AUDIO_URLS = {
    "notification": "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
    "success": "https://assets.mixkit.co/active_storage/sfx/1435/1435-preview.mp3",
    "error": "https://assets.mixkit.co/active_storage/sfx/2955/2955-preview.mp3",
}
