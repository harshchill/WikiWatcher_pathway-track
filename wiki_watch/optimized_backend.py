import os
import json
import time
import asyncio
import traceback
from typing import Dict, List, Optional
import re

import aiohttp
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from groq import AsyncGroq

# Check if GROQ_API_KEY is set
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY environment variable not set. Please set it before starting the pipeline.")

# Create FastAPI app
app = FastAPI(
    title="WikiWatch API (Optimized)",
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

# Simple in-memory pipeline for faster startup
class OptimizedPipeline:
    def __init__(self):
        self.groq_api_key = GROQ_API_KEY
        self.groq_client = AsyncGroq(api_key=self.groq_api_key)
        self.recent_changes = []
        self.running = False
        self.sse_task = None
        self.embedding_model = "embedding-001"
        self.completion_model = "llama3-70b-8192"
        
    async def fetch_diff_content(self, diff_url: str) -> str:
        """Fetch the diff content from the diff URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(diff_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        # Extract the diff content from the HTML
                        diff_match = re.search(r'<table class="diff[^>]*>(.*?)</table>', html_content, re.DOTALL)
                        if diff_match:
                            diff_content = diff_match.group(1)
                            # Clean up HTML tags for better readability
                            diff_content = re.sub(r'<[^>]*>', ' ', diff_content)
                            diff_content = re.sub(r'\s+', ' ', diff_content).strip()
                            return diff_content
                        return "No diff content found"
                    else:
                        return f"Failed to fetch diff: HTTP {response.status}"
        except Exception as e:
            return f"Error fetching diff: {str(e)}"

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using Groq API."""
        try:
            response = await self.groq_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks of approximately the specified token size."""
        # Simple approximation: 1 token ≈ 4 characters
        char_limit = chunk_size * 4
        
        # Split by paragraphs first
        paragraphs = text.split("\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= char_limit:
                current_chunk += paragraph + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If paragraph is longer than chunk size, split it further
                if len(paragraph) > char_limit:
                    words = paragraph.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= char_limit:
                            current_chunk += word + " "
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = word + " "
                else:
                    current_chunk = paragraph + "\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    async def process_sse_event(self, event_data: str) -> Optional[Dict]:
        """Process a Server-Sent Event (SSE) from Wikipedia's RecentChanges stream."""
        try:
            # Parse the JSON event data
            data = json.loads(event_data)
            
            # Extract the required fields
            if data.get("type") == "edit":
                change_id = data.get("id", 0)
                title = data.get("title", "")
                user = data.get("user", "")
                comment = data.get("comment", "")
                timestamp = data.get("timestamp", 0)
                
                # Construct the diff URL
                revid = data.get("revision", {}).get("new", 0)
                oldid = data.get("revision", {}).get("old", 0)
                if revid and oldid:
                    diff_url = f"https://en.wikipedia.org/w/index.php?title={title.replace(' ', '_')}&diff={revid}&oldid={oldid}"
                    
                    # Fetch the diff content
                    diff_content = await self.fetch_diff_content(diff_url)
                    
                    # Create the record
                    return {
                        "change_id": change_id,
                        "title": title,
                        "user": user,
                        "comment": comment,
                        "timestamp": timestamp,
                        "diff_url": diff_url,
                        "diff_content": diff_content
                    }
            
            return None
        except Exception as e:
            print(f"Error processing SSE event: {str(e)}")
            traceback.print_exc()
            return None

    async def start_sse_listener(self):
        """Connect to Wikipedia's RecentChanges SSE stream and process events."""
        sse_url = "https://stream.wikimedia.org/v2/stream/recentchange"
        
        print(f"Connecting to Wikipedia's RecentChanges stream at {sse_url}")
        
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(sse_url, headers={"Accept": "text/event-stream"}) as response:
                        if response.status != 200:
                            print(f"Failed to connect to SSE stream: HTTP {response.status}")
                            await asyncio.sleep(5)  # Wait before retrying
                            continue
                        
                        print("Connected to Wikipedia's RecentChanges stream")
                        
                        # Process the SSE stream
                        buffer = ""
                        async for line in response.content:
                            if not self.running:
                                break
                            
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                buffer += line[6:]  # Remove 'data: ' prefix
                            elif line.strip() == '':  # Empty line indicates end of event
                                if buffer:
                                    event_data = buffer.strip()
                                    buffer = ""
                                    
                                    record = await self.process_sse_event(event_data)
                                    if record:
                                        # Process the diff content
                                        diff_content = record.get("diff_content", "")
                                        if len(diff_content) > 0:
                                            # If the diff is large, split it into chunks
                                            chunks = self.chunk_text(diff_content)
                                            
                                            # For simplicity, we'll just use the first chunk for now
                                            if chunks:
                                                record["diff_content"] = chunks[0]
                                                
                                                # Generate embedding for the chunk
                                                embedding = await self.generate_embedding(chunks[0])
                                                record["embedding"] = embedding
                                                
                                                # Add to recent changes
                                                self.recent_changes.append(record)
                                                print(f"Processed change: {record['title']} by {record['user']}")
                                                
                                                # Keep only the latest 100 changes
                                                if len(self.recent_changes) > 100:
                                                    self.recent_changes = self.recent_changes[-100:]
            except Exception as e:
                print(f"Error in SSE listener: {str(e)}")
                traceback.print_exc()
                
                # Wait before reconnecting
                await asyncio.sleep(5)

    def start(self):
        """Start the pipeline."""
        if self.running:
            return False
            
        self.running = True
        
        # Start the SSE listener as a background task
        self.sse_task = asyncio.create_task(self.start_sse_listener())
        
        return True
        
    def stop(self):
        """Stop the pipeline."""
        if not self.running:
            return False
            
        self.running = False
        
        # Cancel the background task if it exists
        if self.sse_task:
            self.sse_task.cancel()
            self.sse_task = None
            
        return True

    async def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)

    async def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform vector search on the recent changes using the query."""
        try:
            # Generate embedding for the query
            query_embedding = await self.generate_embedding(query)
            
            if not query_embedding:
                return []
                
            # Simple in-memory vector search
            results = []
            for change in self.recent_changes:
                if change.get("embedding"):
                    # Compute cosine similarity
                    similarity = await self.cosine_similarity(query_embedding, change["embedding"])
                    results.append((similarity, change))
            
            # Sort by similarity and take top_k
            results.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in results[:top_k]]
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            traceback.print_exc()
            return []

    async def query(self, question: str) -> str:
        """Answer a question based on the recent changes."""
        try:
            # Search for relevant changes
            relevant_changes = await self.vector_search(question)
            
            if not relevant_changes:
                return "I don't have enough information to answer that question based on recent Wikipedia changes."
                
            # Format the context for Groq
            context = "Here are some recent Wikipedia changes:\n\n"
            for i, change in enumerate(relevant_changes, 1):
                context += f"Change {i}:\n"
                context += f"Page: {change['title']}\n"
                context += f"Editor: {change['user']}\n"
                context += f"Edit comment: {change['comment']}\n"
                context += f"Timestamp: {time.ctime(change['timestamp'])}\n"
                context += f"Diff content: {change['diff_content']}\n\n"
                
            # Generate a response using Groq
            prompt = f"""You are WikiWatch, an AI assistant that monitors and answers questions about recent Wikipedia changes.
            
Based on the following recent Wikipedia changes, please answer this question: "{question}"

{context}

Answer the question concisely and accurately based only on the information provided above. If you cannot answer the question based on the provided changes, state that clearly.
"""
            
            response = await self.groq_client.chat.completions.create(
                model=self.completion_model,
                messages=[
                    {"role": "system", "content": "You are WikiWatch, an AI assistant that monitors and answers questions about recent Wikipedia changes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            traceback.print_exc()
            return f"Error processing your question: {str(e)}"

# Create pipeline instance
pipeline = OptimizedPipeline()

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
    uvicorn.run("optimized_backend:app", host="0.0.0.0", port=8000, reload=True)