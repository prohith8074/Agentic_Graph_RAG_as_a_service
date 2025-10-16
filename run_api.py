#!/usr/bin/env python3
"""
API server runner for the Lyzr Challenge RAG system.
Starts the FastAPI server with all endpoints.
"""

import uvicorn
import logging
from apis import create_app
from config.settings import settings
from database_adapters.database_factory import init_database_connections
from utils.helpers import setup_logging

def main():
    """Main entry point for API server."""
    # Setup logging
    setup_logging()

    # Validate configuration
    try:
        settings.validate()
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        return

    # Initialize database connections
    if not init_database_connections():
        logging.warning("Some database connections failed - system may have limited functionality")

    # Create FastAPI app
    app = create_app()

    # Configure server
    host = "0.0.0.0"
    port = 8000
    workers = 1  # Can be increased for production

    logging.info(f"Starting Lyzr Challenge RAG API server on {host}:{port}")

    # Start server
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()