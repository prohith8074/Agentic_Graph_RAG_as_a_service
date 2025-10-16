"""
API routes for the Lyzr Challenge RAG system.
Provides REST endpoints for querying, ontology management, and system operations.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json

from query.advanced_router import AdvancedQueryRouter
from knowledge_graph.ontology_editor import OntologyEditor
from database_adapters.database_factory import db_manager
from utils.opik_tracer import opik_tracer

router = APIRouter()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    streaming: bool = False

class OntologySuggestionRequest(BaseModel):
    focus_area: str = "general"

class OntologyEditRequest(BaseModel):
    suggestion: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    databases: Dict[str, Any]
    memory: Dict[str, Any]

@router.post("/query")
async def query_endpoint(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Main query endpoint with optional streaming support.
    """
    try:
        router = AdvancedQueryRouter()

        if request.streaming:
            # Return streaming response
            async def generate_stream():
                try:
                    async for result in router.route_query_advanced(request.query):
                        if result.metadata.get('intermediate'):
                            # Send intermediate status updates
                            yield f"data: {json.dumps({'type': 'status', 'data': result.metadata})}\n\n"
                        else:
                            # Send final result
                            yield f"data: {json.dumps({'type': 'result', 'data': result.dict()})}\n\n"
                            break
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )
        else:
            # Return regular JSON response
            result = None
            async for query_result in router.route_query_advanced(request.query):
                if not query_result.metadata.get('intermediate'):
                    result = query_result
                    break

            if result:
                return result.dict()
            else:
                raise HTTPException(status_code=500, detail="No result generated")

    except Exception as e:
        opik_tracer.log_error("api_query_error", str(e), query=request.query)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ontology/suggest")
async def suggest_ontology_improvements(request: OntologySuggestionRequest):
    """
    Get LLM suggestions for ontology improvements.
    """
    try:
        editor = OntologyEditor()
        # Load current ontology (would be from database in real implementation)
        suggestions = await editor.suggest_improvements(request.focus_area)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ontology/edit")
async def apply_ontology_edit(request: OntologyEditRequest):
    """
    Apply an ontology edit suggestion.
    """
    try:
        editor = OntologyEditor()
        success, message = await editor.apply_suggestion(request.suggestion)
        return {"success": success, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ontology/visualization")
async def get_ontology_visualization():
    """
    Get ontology data formatted for visualization.
    """
    try:
        editor = OntologyEditor()
        viz_data = editor.get_visualization_data()
        return viz_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    System health check endpoint.
    """
    try:
        # Check database health
        db_health = db_manager.health_check_all()

        # Check memory systems (placeholder for actual implementation)
        memory_health = {
            "redis_available": True,  # Would check actual Redis connection
            "cache_size": 0,  # Would get actual cache size
            "collections_count": 1  # Would get actual collection count
        }

        overall_status = "healthy"
        if any(db.get("status") != "healthy" for db in db_health.values()):
            overall_status = "degraded"

        # Get comprehensive memory statistics
        from utils.memory_manager import memory_manager
        memory_stats = memory_manager.get_memory_stats()

        return HealthResponse(
            status=overall_status,
            databases=db_health,
            memory=memory_stats
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            databases={},
            memory={"error": str(e)}
        )

@router.get("/database/stats")
async def get_database_stats():
    """
    Get statistics for all connected databases.
    """
    try:
        stats = {}
        for name, adapter in db_manager.adapters.items():
            stats[name] = adapter.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/database/switch/{db_type}")
async def switch_database(db_type: str):
    """
    Switch active database.
    """
    try:
        if db_manager.set_active_adapter(db_type):
            return {"success": True, "active_database": db_type}
        else:
            raise HTTPException(status_code=400, detail=f"Database {db_type} not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/database/available")
async def get_available_databases():
    """
    Get list of available database types.
    """
    from database_adapters.database_factory import DatabaseFactory
    return DatabaseFactory.get_available_adapters()

@router.delete("/cache")
async def clear_cache():
    """
    Clear query result cache.
    """
    try:
        from utils.memory_manager import memory_manager
        success = memory_manager.invalidate_cache()
        return {"success": success, "message": "Cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))