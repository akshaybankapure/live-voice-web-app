"""
Latency Tracker for Voice AI Agent

Tracks per-stage latency metrics for:
- STT (speech-to-text)
- LLM (language model inference)
- TTS (text-to-speech)
- Tool execution

Provides end-to-end latency measurements to ensure ≤2s target.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import structlog

logger = structlog.get_logger()


@dataclass
class StageLatency:
    """Latency measurements for a single processing stage."""
    stage: str
    start_time: Optional[float] = None
    first_result_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def time_to_first_result(self) -> Optional[float]:
        """Time from start to first partial result (ms)."""
        if self.start_time and self.first_result_time:
            return (self.first_result_time - self.start_time) * 1000
        return None

    @property
    def total_duration(self) -> Optional[float]:
        """Total duration from start to end (ms)."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage,
            "time_to_first_result_ms": round(self.time_to_first_result, 2) if self.time_to_first_result else None,
            "total_duration_ms": round(self.total_duration, 2) if self.total_duration else None,
        }


@dataclass
class TurnLatency:
    """Latency breakdown for a single conversation turn."""
    turn_id: int
    turn_start_time: float = field(default_factory=time.time)
    turn_end_time: Optional[float] = None
    stt: StageLatency = field(default_factory=lambda: StageLatency("stt"))
    llm: StageLatency = field(default_factory=lambda: StageLatency("llm"))
    tool: StageLatency = field(default_factory=lambda: StageLatency("tool"))
    tts: StageLatency = field(default_factory=lambda: StageLatency("tts"))

    @property
    def end_to_end_latency(self) -> Optional[float]:
        """End-to-end latency from audio input to first TTS output (ms)."""
        if self.stt.start_time and self.tts.first_result_time:
            return (self.tts.first_result_time - self.stt.start_time) * 1000
        return None

    @property
    def total_turn_duration(self) -> Optional[float]:
        """Total turn duration (ms)."""
        if self.turn_end_time:
            return (self.turn_end_time - self.turn_start_time) * 1000
        return None

    def to_dict(self) -> Dict:
        return {
            "turn_id": self.turn_id,
            "end_to_end_latency_ms": round(self.end_to_end_latency, 2) if self.end_to_end_latency else None,
            "total_turn_duration_ms": round(self.total_turn_duration, 2) if self.total_turn_duration else None,
            "stages": {
                "stt": self.stt.to_dict(),
                "llm": self.llm.to_dict(),
                "tool": self.tool.to_dict(),
                "tts": self.tts.to_dict(),
            }
        }


class LatencyTracker:
    """
    Tracks latency metrics for voice agent turns.
    
    Usage:
        tracker = LatencyTracker()
        tracker.start_stt()
        tracker.stt_first_result()
        tracker.end_stt()
        tracker.start_llm()
        # ... etc
        tracker.finish_turn()
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.turns: List[TurnLatency] = []
        self._current_turn: TurnLatency = TurnLatency(turn_id=0)
        self._turn_counter = 0

    def _log_stage(self, stage: str, event: str, latency_ms: Optional[float] = None) -> None:
        """Log stage events with structured logging."""
        log_data = {
            "session_id": self.session_id,
            "turn_id": self._current_turn.turn_id,
            "stage": stage,
        }
        if latency_ms is not None:
            log_data["latency_ms"] = round(latency_ms, 2)
        logger.info(f"{stage}_{event}", **log_data)

    # STT timing methods
    def start_stt(self) -> None:
        self._current_turn.stt.start_time = time.time()
        self._log_stage("stt", "started")

    def stt_first_result(self) -> None:
        self._current_turn.stt.first_result_time = time.time()
        self._log_stage("stt", "first_result", self._current_turn.stt.time_to_first_result)

    def end_stt(self) -> None:
        self._current_turn.stt.end_time = time.time()
        self._log_stage("stt", "completed", self._current_turn.stt.total_duration)

    # LLM timing methods
    def start_llm(self) -> None:
        self._current_turn.llm.start_time = time.time()
        self._log_stage("llm", "started")

    def llm_first_token(self) -> None:
        self._current_turn.llm.first_result_time = time.time()
        self._log_stage("llm", "first_token", self._current_turn.llm.time_to_first_result)

    def end_llm(self) -> None:
        self._current_turn.llm.end_time = time.time()
        self._log_stage("llm", "completed", self._current_turn.llm.total_duration)

    # Tool timing methods
    def start_tool(self) -> None:
        self._current_turn.tool.start_time = time.time()
        self._log_stage("tool", "started")

    def end_tool(self) -> None:
        self._current_turn.tool.end_time = time.time()
        self._current_turn.tool.first_result_time = self._current_turn.tool.end_time
        self._log_stage("tool", "completed", self._current_turn.tool.total_duration)

    # TTS timing methods
    def start_tts(self) -> None:
        self._current_turn.tts.start_time = time.time()
        self._log_stage("tts", "started")

    def tts_first_audio(self) -> None:
        self._current_turn.tts.first_result_time = time.time()
        self._log_stage("tts", "first_audio", self._current_turn.tts.time_to_first_result)
        # Log end-to-end latency when first audio is produced
        e2e = self._current_turn.end_to_end_latency
        if e2e:
            logger.info("end_to_end_latency", 
                       session_id=self.session_id,
                       turn_id=self._current_turn.turn_id,
                       latency_ms=round(e2e, 2),
                       target_met=e2e <= 2000)

    def end_tts(self) -> None:
        self._current_turn.tts.end_time = time.time()
        self._log_stage("tts", "completed", self._current_turn.tts.total_duration)

    def finish_turn(self) -> TurnLatency:
        """Finalize current turn and start a new one."""
        self._current_turn.turn_end_time = time.time()
        completed_turn = self._current_turn
        self.turns.append(completed_turn)
        
        logger.info("turn_completed",
                   session_id=self.session_id,
                   turn_id=completed_turn.turn_id,
                   total_duration_ms=round(completed_turn.total_turn_duration or 0, 2),
                   end_to_end_ms=round(completed_turn.end_to_end_latency or 0, 2))
        
        self._turn_counter += 1
        self._current_turn = TurnLatency(turn_id=self._turn_counter)
        return completed_turn

    @property
    def average_end_to_end_latency(self) -> Optional[float]:
        """Average end-to-end latency across all turns (ms)."""
        latencies = [t.end_to_end_latency for t in self.turns if t.end_to_end_latency]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def get_summary(self) -> Dict:
        """Get latency summary for the session."""
        avg_e2e = self.average_end_to_end_latency
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "average_end_to_end_latency_ms": round(avg_e2e, 2) if avg_e2e else None,
            "target_latency_ms": 2000,
            "target_met": avg_e2e is not None and avg_e2e <= 2000,
        }

    def get_last_turn(self) -> Dict:
        """Get the last completed turn's latency breakdown."""
        if not self.turns:
            return {}
        return self.turns[-1].to_dict()
