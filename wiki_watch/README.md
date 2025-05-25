# WikiWatch: Real-time Wikipedia Change Q&A Bot

WikiWatch is a real-time application that monitors Wikipedia's recent changes, processes them, and allows users to ask natural language questions about what's happening on Wikipedia right now.

## Features

- Connect to Wikipedia's RecentChanges SSE stream to get live updates
- Process and store change data including titles, users, comments, and diff content
- Generate embeddings for efficient semantic search
- Query the collected data using natural language questions
- View recent changes in real-time through a Streamlit frontend

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
