"""
APIs module for external interfaces and SDKs.
Provides REST API, GraphQL, and programmatic interfaces.
"""

from fastapi import FastAPI
from typing import Optional

# Global API app instance
app: Optional[FastAPI] = None

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    global app
    if app is None:
        from .routes import router
        app = FastAPI(
            title="Lyzr Challenge RAG API",
            description="Agentic Graph RAG as a Service",
            version="1.0.0"
        )
        app.include_router(router)
    return app