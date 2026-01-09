"""
Cost Tracker for Voice AI Agent

Tracks and estimates costs for STT, LLM, and TTS usage per turn and conversation.

Pricing (as of 2025):
- Soniox STT: ~$0.002/minute (token-based: $2/1M audio tokens + $4/1M output tokens)
- Groq LLM (llama-3.3-70b-versatile): $0.59/1M input tokens, $0.79/1M output tokens
- ElevenLabs TTS: ~$0.24/1K characters (Pro tier)
"""

from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class TurnCost:
    """Cost breakdown for a single turn."""
    turn_id: int
    stt_cost: float = 0.0
    llm_input_cost: float = 0.0
    llm_output_cost: float = 0.0
    tts_cost: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def llm_cost(self) -> float:
        return self.llm_input_cost + self.llm_output_cost

    @property
    def total(self) -> float:
        return self.stt_cost + self.llm_cost + self.tts_cost

    def to_dict(self) -> Dict:
        return {
            "turn_id": self.turn_id,
            "stt_cost": round(self.stt_cost, 6),
            "llm_cost": round(self.llm_cost, 6),
            "tts_cost": round(self.tts_cost, 6),
            "total": round(self.total, 6),
        }


class CostTracker:
    """
    Tracks costs per turn and aggregates for entire conversation.
    
    Usage:
        tracker = CostTracker()
        tracker.add_stt_cost(audio_duration_seconds=30)
        tracker.add_llm_cost(input_tokens=100, output_tokens=50)
        tracker.add_tts_cost(characters=200)
        tracker.finish_turn()
    """

    # Pricing constants (USD)
    # Soniox: ~$0.12/hour = $0.002/minute
    SONIOX_COST_PER_MINUTE = 0.002
    
    # Groq: llama-3.3-70b-versatile
    GROQ_INPUT_COST_PER_MILLION = 0.59
    GROQ_OUTPUT_COST_PER_MILLION = 0.79
    
    # ElevenLabs: Pro tier
    ELEVENLABS_COST_PER_1K_CHARS = 0.24

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.turns: List[TurnCost] = []
        self._current_turn: TurnCost = TurnCost(turn_id=0)
        self._turn_counter = 0

    def add_stt_cost(self, audio_duration_seconds: float) -> None:
        """Add STT cost based on audio duration."""
        minutes = audio_duration_seconds / 60.0
        self._current_turn.stt_cost += minutes * self.SONIOX_COST_PER_MINUTE

    def add_llm_cost(self, input_tokens: int, output_tokens: int) -> None:
        """Add LLM cost based on token counts."""
        self._current_turn.llm_input_cost += (input_tokens / 1_000_000) * self.GROQ_INPUT_COST_PER_MILLION
        self._current_turn.llm_output_cost += (output_tokens / 1_000_000) * self.GROQ_OUTPUT_COST_PER_MILLION

    def add_tts_cost(self, characters: int) -> None:
        """Add TTS cost based on character count."""
        self._current_turn.tts_cost += (characters / 1000) * self.ELEVENLABS_COST_PER_1K_CHARS

    def finish_turn(self) -> TurnCost:
        """Finalize current turn and start a new one."""
        completed_turn = self._current_turn
        self.turns.append(completed_turn)
        self._turn_counter += 1
        self._current_turn = TurnCost(turn_id=self._turn_counter)
        return completed_turn

    @property
    def total_stt_cost(self) -> float:
        return sum(t.stt_cost for t in self.turns) + self._current_turn.stt_cost

    @property
    def total_llm_cost(self) -> float:
        return sum(t.llm_cost for t in self.turns) + self._current_turn.llm_cost

    @property
    def total_tts_cost(self) -> float:
        return sum(t.tts_cost for t in self.turns) + self._current_turn.tts_cost

    @property
    def total_cost(self) -> float:
        return self.total_stt_cost + self.total_llm_cost + self.total_tts_cost

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def average_cost_per_turn(self) -> float:
        if not self.turns:
            return 0.0
        return self.total_cost / len(self.turns)

    def get_summary(self) -> Dict:
        """Get cost summary for the conversation."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "costs": {
                "stt": round(self.total_stt_cost, 6),
                "llm": round(self.total_llm_cost, 6),
                "tts": round(self.total_tts_cost, 6),
                "total": round(self.total_cost, 6),
            },
            "average_per_turn": round(self.average_cost_per_turn, 6),
        }

    def get_last_turn(self) -> Dict:
        """Get the last completed turn's cost breakdown."""
        if not self.turns:
            return {}
        return self.turns[-1].to_dict()
