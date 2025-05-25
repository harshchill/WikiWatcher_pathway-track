import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
import asyncio
import pathway as pw
import requests
from groq import AsyncGroq
from pathway import Table


# Define schema for the RecentChanges table
@dataclass
class RecentChange:
    change_id: int
    title: str
    user: str
    comment: str
    timestamp: float
    diff_url: str
    diff_content: Optional[str] = None
    embedding: Optional[List[float]] = None


class WikiWatchPipeline:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.groq_client = AsyncGroq(api_key=self.groq_api_key)
        self.recent_changes_table = None
        self.running = False
        self.pipeline_thread = None
        self.embedding_model = "embedding-001"
        self.completion_model = "llama3-70b-8192"
        self.pathway_app = None
        
    async def fetch_diff_content(self, diff_url: str) -> str:
        """Fetch the diff content from the diff URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(diff_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        # Extract the diff content from the HTML
                        # This is a simple regex-based extraction, might need adjustment
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
                                            # In a production system, you might want to store all chunks
                                            if chunks:
                                                record["diff_content"] = chunks[0]
                                                
                                                # Generate embedding for the chunk
                                                embedding = await self.generate_embedding(chunks[0])
                                                record["embedding"] = embedding
                                                
                                                # Write to the Pathway table
                                                # Note: In a real implementation, you would need to properly handle
                                                # writing to Pathway tables from an async context
                                                print(f"Processed change: {record['title']} by {record['user']}")
                                                
                                                # Add to in-memory store for now (would use Pathway in production)
                                                self.recent_changes.append(record)
            except Exception as e:
                print(f"Error in SSE listener: {str(e)}")
                traceback.print_exc()
                
                # Wait before reconnecting
                await asyncio.sleep(5)

    def setup_pathway_pipeline(self):
        """Set up the Pathway data processing pipeline."""
        class WikiChangeInput(pw.io.python.ConnectorSubject):
            def __init__(self):
                super().__init__()
                self.changes = []
                
            def add_change(self, change):
                self.changes.append(change)
                self.on_next(change)
                
            def get_changes(self):
                return self.changes
                
        # Create a subject to feed data into Pathway
        self.input_subject = WikiChangeInput()
        
        # Define the input connector
        input_connector = pw.io.python.read(self.input_subject, schema=RecentChange)
        
        # Store the table for querying
        self.recent_changes_table = input_connector
        
        # Set up the Pathway app
        self.pathway_app = pw.App()
        self.pathway_app.run()
        
    async def add_to_pathway(self, record):
        """Add a record to the Pathway table."""
        change = RecentChange(
            change_id=record["change_id"],
            title=record["title"],
            user=record["user"],
            comment=record["comment"],
            timestamp=record["timestamp"],
            diff_url=record["diff_url"],
            diff_content=record["diff_content"],
            embedding=record["embedding"]
        )
        self.input_subject.add_change(change)
        
    def start(self):
        """Start the pipeline."""
        if self.running:
            return False
            
        self.running = True
        self.recent_changes = []
        
        # Set up the Pathway pipeline
        self.setup_pathway_pipeline()
        
        # Start the SSE listener in a separate thread
        self.pipeline_thread = asyncio.new_event_loop()
        asyncio.run_coroutine_threadsafe(self.start_sse_listener(), self.pipeline_thread)
        
        return True
        
    def stop(self):
        """Stop the pipeline."""
        self.running = False
        return True
        
    async def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform vector search on the recent changes using the query."""
        try:
            # Generate embedding for the query
            query_embedding = await self.generate_embedding(query)
            
            if not query_embedding:
                return []
                
            # In a real implementation, you would use Pathway's vector search capabilities
            # For this simplified version, we'll do a basic similarity computation
            results = []
            for change in self.recent_changes:
                if change.get("embedding"):
                    # Compute cosine similarity
                    similarity = self.cosine_similarity(query_embedding, change["embedding"])
                    results.append((similarity, change))
            
            # Sort by similarity and take top_k
            results.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in results[:top_k]]
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            traceback.print_exc()
            return []
            
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
        
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


# Singleton instance
pipeline_instance = None

def get_pipeline_instance():
    """Get or create the pipeline instance."""
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = WikiWatchPipeline()
    return pipeline_instance


if __name__ == "__main__":
    # Simple test
    async def test():
        pipeline = WikiWatchPipeline()
        pipeline.start()
        await asyncio.sleep(60)  # Run for 60 seconds
        print(f"Collected {len(pipeline.recent_changes)} changes")
        pipeline.stop()
        
    asyncio.run(test())
