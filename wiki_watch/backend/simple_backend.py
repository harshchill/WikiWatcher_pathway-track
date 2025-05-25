import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="WikiWatch API (Simple Version)",
    description="Simplified version of the WikiWatch API for UI demonstration",
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

# In-memory storage for demonstration
class SimplePipeline:
    def __init__(self):
        self.running = False
        self.recent_changes = []
        self.load_demo_data()
    
    def load_demo_data(self):
        """Load demonstration data for UI testing"""
        self.recent_changes = [
            {
                "change_id": 1,
                "title": "Python (programming language)",
                "user": "PythonFan123",
                "comment": "Updated version information and added new library references",
                "timestamp": time.time() - 3600,  # 1 hour ago
                "diff_url": "https://en.wikipedia.org/w/index.php?title=Python_(programming_language)&diff=1234567&oldid=1234566"
            },
            {
                "change_id": 2,
                "title": "Machine Learning",
                "user": "AIResearcher",
                "comment": "Added recent developments in transformer models",
                "timestamp": time.time() - 7200,  # 2 hours ago
                "diff_url": "https://en.wikipedia.org/w/index.php?title=Machine_learning&diff=1234568&oldid=1234565"
            },
            {
                "change_id": 3,
                "title": "Artificial Intelligence",
                "user": "TechEditor42",
                "comment": "Fixed references and updated ethical considerations section",
                "timestamp": time.time() - 10800,  # 3 hours ago
                "diff_url": "https://en.wikipedia.org/w/index.php?title=Artificial_intelligence&diff=1234569&oldid=1234564"
            },
            {
                "change_id": 4,
                "title": "ChatGPT",
                "user": "AIHistorian",
                "comment": "Updated with latest version information and capabilities",
                "timestamp": time.time() - 14400,  # 4 hours ago
                "diff_url": "https://en.wikipedia.org/w/index.php?title=ChatGPT&diff=1234570&oldid=1234563"
            },
            {
                "change_id": 5,
                "title": "Natural Language Processing",
                "user": "NLPExpert",
                "comment": "Added section on recent benchmarks and state-of-the-art models",
                "timestamp": time.time() - 18000,  # 5 hours ago
                "diff_url": "https://en.wikipedia.org/w/index.php?title=Natural_language_processing&diff=1234571&oldid=1234562"
            }
        ]
    
    def start(self):
        """Start the pipeline."""
        if self.running:
            return False
        self.running = True
        return True
    
    def stop(self):
        """Stop the pipeline."""
        if not self.running:
            return False
        self.running = False
        return True

    async def query(self, question: str) -> str:
        """Answer a question based on the recent changes."""
        # For demo purposes, generate a static response based on the question
        if "python" in question.lower():
            return f"Based on recent changes, the Python programming language article was updated by PythonFan123 with version information and library references. This suggests ongoing documentation improvements to keep the article current with the latest Python developments."
        
        elif "machine learning" in question.lower() or "ai" in question.lower():
            return f"There have been several recent edits to AI-related articles. AIResearcher updated the Machine Learning article with information about transformer models, while TechEditor42 revised the Artificial Intelligence article's ethical considerations section. AIHistorian also updated the ChatGPT article with the latest capabilities."
        
        elif "recent" in question.lower() or "latest" in question.lower():
            return f"The most recent Wikipedia changes in my database include updates to: Python (programming language), Machine Learning, Artificial Intelligence, ChatGPT, and Natural Language Processing articles. These changes primarily focus on updating technical information, adding recent developments, and improving references."
        
        else:
            return f"Based on the recent changes in my database, I don't have specific information about '{question}'. The most active editors recently include PythonFan123, AIResearcher, TechEditor42, AIHistorian, and NLPExpert, who have been updating technology-related articles."


# Create a singleton instance
pipeline = SimplePipeline()

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
    
    This simplified endpoint demonstrates the querying interface with pre-generated responses.
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
            "changes_count": len(pipeline.recent_changes)
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
        changes = pipeline.recent_changes[-limit:] if pipeline.recent_changes else []
        
        # For demo purposes, we're already using the right format
        return {
            "status": "success",
            "count": len(changes),
            "changes": changes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent changes: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("simple_backend:app", host="0.0.0.0", port=8000, reload=True)