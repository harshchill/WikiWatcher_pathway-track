import os
import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Thread

from pipeline import get_pipeline_instance

# Check if GROQ_API_KEY is set
if not os.getenv("GROQ_API_KEY"):
    print("Warning: GROQ_API_KEY environment variable not set. Please set it before starting the pipeline.")

# Create FastAPI app
app = FastAPI(
    title="WikiWatch API",
    description="Real-time Wikipedia change Q&A bot API",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the pipeline instance
pipeline = get_pipeline_instance()

@app.post("/start_pipeline")
async def start_pipeline():
    """Start the Wikipedia RecentChanges SSE listener pipeline."""
    try:
        result = pipeline.start()
        if result:
            return {"status": "success", "message": "Pipeline started successfully"}
        else:
            return {"status": "info", "message": "Pipeline was already running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")

@app.post("/stop_pipeline")
async def stop_pipeline():
    """Stop the Wikipedia RecentChanges SSE listener pipeline."""
    try:
        result = pipeline.stop()
        if result:
            return {"status": "success", "message": "Pipeline stopped successfully"}
        else:
            return {"status": "info", "message": "Pipeline was not running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop pipeline: {str(e)}")

@app.get("/query")
async def query(q: str = Query(..., description="Question to ask about recent Wikipedia changes")):
    """
    Query the recent Wikipedia changes with a natural language question.
    
    This endpoint takes a question, performs vector search over the live recent_changes table,
    and uses the Groq chat completion endpoint to generate a concise answer based on the retrieved data.
    """
    if not pipeline.running:
        raise HTTPException(status_code=400, detail="Pipeline is not running. Start it first with /start_pipeline")
        
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty")
        
    try:
        answer = await pipeline.query(q)
        return {
            "status": "success", 
            "question": q, 
            "answer": answer,
            "changes_count": len(pipeline.recent_changes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/status")
async def status():
    """Get the current status of the pipeline."""
    try:
        return {
            "status": "success",
            "pipeline_running": pipeline.running,
            "changes_count": len(pipeline.recent_changes) if hasattr(pipeline, 'recent_changes') else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.get("/recent_changes")
async def get_recent_changes(limit: int = Query(10, description="Maximum number of changes to return")):
    """Get the most recent changes from the pipeline."""
    if not pipeline.running:
        raise HTTPException(status_code=400, detail="Pipeline is not running. Start it first with /start_pipeline")
        
    try:
        # Return the most recent changes, limited to the specified number
        changes = pipeline.recent_changes[-limit:] if hasattr(pipeline, 'recent_changes') else []
        
        # Format the changes for display
        formatted_changes = []
        for change in changes:
            formatted_changes.append({
                "change_id": change["change_id"],
                "title": change["title"],
                "user": change["user"],
                "comment": change["comment"],
                "timestamp": change["timestamp"],
                "diff_url": change["diff_url"]
            })
            
        return {
            "status": "success",
            "count": len(formatted_changes),
            "changes": formatted_changes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent changes: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
