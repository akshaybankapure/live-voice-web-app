"""
LiveKit Agent Worker Entry Point

Runs the LiveKit agents worker that handles voice sessions.
"""

import os
import asyncio
import structlog
from dotenv import load_dotenv

from livekit import agents

from app.agent import create_worker

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


def main():
    """Run the LiveKit agent worker."""
    logger.info("starting_livekit_worker",
               livekit_url=os.getenv("LIVEKIT_URL", "not_set"))
    
    # Create the worker
    worker = create_worker()
    
    # Run the worker
    agents.run_app(worker)


if __name__ == "__main__":
    main()
