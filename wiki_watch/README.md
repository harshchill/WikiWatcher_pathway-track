# WikiWatch: Real-time Wikipedia Change Q&A Bot

WikiWatch is a real-time application that monitors Wikipedia's recent changes, processes them, and allows users to ask natural language questions about what's happening on Wikipedia right now.

🔗 **Live Demo**: [https://wikiwatcher-frontend.onrender.com](https://wikiwatcher-frontend.onrender.com)

## Project Structure
```
wiki_watch/
├── backend/
│   ├── app.py              # Main FastAPI application
│   ├── optimized_backend.py # Optimized backend implementation
│   ├── pipeline.py         # Wikipedia SSE stream processing
│   ├── simple_backend.py   # Simple backend implementation
│   └── Procfile           # Backend deployment configuration
├── frontend/
│   ├── frontend.py        # Streamlit frontend application
│   ├── Procfile          # Frontend deployment configuration
│   └── .streamlit/       # Streamlit configuration
├── config/
│   └── .env.template     # Environment variables template
├── requirements.txt      # Python dependencies
├── runtime.txt          # Python version specification
└── README.md           # Project documentation
```

## Features

- Connect to Wikipedia's RecentChanges SSE stream to get live updates
- Process and store change data including titles, users, comments, and diff content
- Generate embeddings for efficient semantic search
- Query the collected data using natural language questions
- View recent changes in real-time through a Streamlit frontend

## Local Development Setup

1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Set up environment variables
```bash
cp .env.template .env
# Edit .env and add your GROQ_API_KEY
```

3. Run the backend server
```bash
cd backend
python -m uvicorn optimized_backend:app --host 0.0.0.0 --port 8000
```

4. Run the frontend (in a separate terminal)
```bash
cd frontend
streamlit run frontend.py --server.port 5000
```

## Deployment on Render

### Backend Deployment

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +"
3. Select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - Name: wikiwatch-backend
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python -m uvicorn optimized_backend:app --host 0.0.0.0 --port $PORT`
6. Add environment variable:
   - Key: GROQ_API_KEY
   - Value: Your Groq API key

### Frontend Deployment

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +"
3. Select "Web Service"
4. Connect your GitHub repository
5. Configure the service:
   - Name: wikiwatch-frontend
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run frontend.py --server.port $PORT`
6. Add environment variable:
   - Key: API_URL
   - Value: Your backend URL (e.g., https://wikiwatch-backend.onrender.com)


## Usage

1. Visit your deployed frontend URL
2. Click "Start" to begin collecting Wikipedia changes
3. Ask questions about recent changes in the search box
4. View real-time updates in the Recent Wikipedia Activity section

## API Endpoints

- `POST /start_pipeline`: Start monitoring Wikipedia changes
- `POST /stop_pipeline`: Stop the monitoring pipeline
- `GET /query?q=<question>`: Ask questions about recent changes
- `GET /status`: Get pipeline status
- `GET /recent_changes`: Get list of recent changes
