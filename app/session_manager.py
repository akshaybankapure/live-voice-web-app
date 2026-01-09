"""
Session Manager for Voice AI Agent

Manages concurrent voice sessions with:
- Session lifecycle (create, get, remove)
- Aggregate metrics across sessions
- Thread-safe operations
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import asyncio
from datetime import datetime
import structlog

from .cost_tracker import CostTracker
from .latency_tracker import LatencyTracker

logger = structlog.get_logger()


@dataclass
class Session:
    """Represents a single voice session."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    cost_tracker: CostTracker = field(default=None)
    latency_tracker: LatencyTracker = field(default=None)
    is_active: bool = True
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.cost_tracker is None:
            self.cost_tracker = CostTracker(self.session_id)
        if self.latency_tracker is None:
            self.latency_tracker = LatencyTracker(self.session_id)


class SessionManager:
    """
    Manages concurrent voice sessions.
    
    Thread-safe session management with aggregate metrics.
    
    Usage:
        manager = SessionManager()
        session = await manager.create_session("session-123")
        session.cost_tracker.add_stt_cost(...)
        await manager.remove_session("session-123")
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._total_sessions_created = 0

    async def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> Session:
        """Create a new session."""
        async with self._lock:
            if session_id in self._sessions:
                logger.warning("session_already_exists", session_id=session_id)
                return self._sessions[session_id]

            session = Session(
                session_id=session_id,
                metadata=metadata or {}
            )
            self._sessions[session_id] = session
            self._total_sessions_created += 1

            logger.info("session_created",
                       session_id=session_id,
                       active_sessions=len(self._sessions))
            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> Optional[Session]:
        """Remove and return a session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.is_active = False
                logger.info("session_removed",
                           session_id=session_id,
                           duration_seconds=(datetime.utcnow() - session.created_at).total_seconds(),
                           turns=session.latency_tracker.turn_count,
                           total_cost=session.cost_tracker.total_cost,
                           active_sessions=len(self._sessions))
            return session

    @property
    def active_session_count(self) -> int:
        """Number of currently active sessions."""
        return len(self._sessions)

    @property
    def total_sessions_created(self) -> int:
        """Total sessions created since startup."""
        return self._total_sessions_created

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self._sessions.keys())

    def get_aggregate_metrics(self) -> Dict:
        """Get aggregate metrics across all active sessions."""
        if not self._sessions:
            return {
                "active_sessions": 0,
                "total_sessions_created": self._total_sessions_created,
                "aggregate_cost": 0.0,
                "average_latency_ms": None,
            }

        total_cost = 0.0
        latencies = []
        total_turns = 0

        for session in self._sessions.values():
            total_cost += session.cost_tracker.total_cost
            total_turns += session.latency_tracker.turn_count
            avg_lat = session.latency_tracker.average_end_to_end_latency
            if avg_lat:
                latencies.append(avg_lat)

        avg_latency = sum(latencies) / len(latencies) if latencies else None

        return {
            "active_sessions": self.active_session_count,
            "total_sessions_created": self._total_sessions_created,
            "total_turns": total_turns,
            "aggregate_cost": {
                "total": round(total_cost, 6),
                "breakdown": self._get_cost_breakdown(),
            },
            "average_latency_ms": round(avg_latency, 2) if avg_latency else None,
            "target_latency_ms": 2000,
        }

    def _get_cost_breakdown(self) -> Dict:
        """Get aggregated cost breakdown across all sessions."""
        stt_cost = 0.0
        llm_cost = 0.0
        tts_cost = 0.0

        for session in self._sessions.values():
            stt_cost += session.cost_tracker.total_stt_cost
            llm_cost += session.cost_tracker.total_llm_cost
            tts_cost += session.cost_tracker.total_tts_cost

        return {
            "stt": round(stt_cost, 6),
            "llm": round(llm_cost, 6),
            "tts": round(tts_cost, 6),
        }

    def get_session_details(self, session_id: str) -> Optional[Dict]:
        """Get detailed info for a specific session."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "is_active": session.is_active,
            "cost": session.cost_tracker.get_summary(),
            "latency": session.latency_tracker.get_summary(),
            "metadata": session.metadata,
        }


# Global session manager instance
session_manager = SessionManager()
