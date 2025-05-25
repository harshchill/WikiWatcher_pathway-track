# WikiWatch: Real-time Wikipedia Change Q&A Bot

WikiWatch is a real-time application that monitors Wikipedia's recent changes, processes them, and allows users to ask natural language questions about what's happening on Wikipedia right now.

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
python -m uvicorn optimized_backend:app --host 0.0.0.0 --port 8000
```

4. Run the frontend (in a separate terminal)
```bash
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

### Important Notes for Free Tier

1. Services will spin down after 15 minutes of inactivity
2. Maximum of 750 hours per month of runtime
3. Limited to 512 MB RAM and 0.1 CPU
4. Automatic HTTPS/SSL provided
5. Cold starts may occur after inactivity

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
