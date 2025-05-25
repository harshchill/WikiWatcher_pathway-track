import time
import requests
import streamlit as st
from datetime import datetime

# Set up the Streamlit page
st.set_page_config(
    page_title="WikiWatch - Real-time Wikipedia Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define the API URL
API_URL = "http://localhost:8000"

def start_pipeline():
    """Start the Wikipedia RecentChanges pipeline."""
    try:
        response = requests.post(f"{API_URL}/start_pipeline")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to start pipeline: {str(e)}")
        return None

def stop_pipeline():
    """Stop the Wikipedia RecentChanges pipeline."""
    try:
        response = requests.post(f"{API_URL}/stop_pipeline")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to stop pipeline: {str(e)}")
        return None

def get_status():
    """Get the current status of the pipeline."""
    try:
        response = requests.get(f"{API_URL}/status")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get status: {str(e)}")
        return {"status": "error", "pipeline_running": False, "changes_count": 0}

def get_recent_changes(limit=10):
    """Get the most recent changes from the pipeline."""
    try:
        response = requests.get(f"{API_URL}/recent_changes?limit={limit}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get recent changes: {str(e)}")
        return {"status": "error", "count": 0, "changes": []}

def query(question):
    """Query the pipeline with a natural language question."""
    try:
        response = requests.get(f"{API_URL}/query?q={question}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to query: {str(e)}")
        return {"status": "error", "question": question, "answer": f"Error: {str(e)}"}

# Title and introduction
st.title("WikiWatch: Real-time Wikipedia Q&A")
st.markdown("""
This application monitors Wikipedia's recent changes in real-time and allows you to ask
questions about what's happening on Wikipedia right now.

1. Start the pipeline to begin collecting recent changes
2. Ask questions about the recent edits
3. View the latest changes as they happen
""")

# Sidebar for controls and status
with st.sidebar:
    st.header("Pipeline Controls")
    
    # Pipeline control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Pipeline"):
            with st.spinner("Starting pipeline..."):
                result = start_pipeline()
                if result and result["status"] in ["success", "info"]:
                    st.success(result["message"])
                    time.sleep(1)
                    st.rerun()
    
    with col2:
        if st.button("Stop Pipeline"):
            with st.spinner("Stopping pipeline..."):
                result = stop_pipeline()
                if result and result["status"] in ["success", "info"]:
                    st.success(result["message"])
                    time.sleep(1)
                    st.rerun()
    
    # Status information
    st.header("Status")
    status = get_status()
    
    if status["pipeline_running"]:
        st.success("Pipeline is running")
    else:
        st.error("Pipeline is not running")
    
    st.metric("Changes collected", status.get("changes_count", 0))
    
    # Refresh button
    if st.button("Refresh Status"):
        st.rerun()

# Main content area
st.header("Ask about recent Wikipedia changes")

# User query input
question = st.text_input("Ask a question about recent Wikipedia changes:", 
                        placeholder="E.g., What recent changes were made to articles about Python?")

# Query button
if st.button("Ask") and question:
    with st.spinner("Analyzing recent changes..."):
        result = query(question)
        
        if result["status"] == "success":
            st.subheader("Answer")
            st.write(result["answer"])
            st.caption(f"Based on {result.get('changes_count', 0)} recent changes")
        else:
            st.error("Failed to get an answer. Make sure the pipeline is running.")

# Recent changes section
st.header("Recent Wikipedia Changes")
if st.button("Refresh Changes"):
    st.rerun()

# Display the recent changes
changes_data = get_recent_changes(limit=20)
if changes_data["status"] == "success" and changes_data["count"] > 0:
    for change in changes_data["changes"]:
        with st.expander(f"{change['title']} - by {change['user']}"):
            st.write(f"**Page**: {change['title']}")
            st.write(f"**Editor**: {change['user']}")
            st.write(f"**Comment**: {change['comment']}")
            st.write(f"**Time**: {datetime.fromtimestamp(change['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Diff URL**: [{change['diff_url']}]({change['diff_url']})")
else:
    if status["pipeline_running"]:
        st.info("Waiting for changes to be collected... This may take a few moments.")
    else:
        st.warning("No changes available. Please start the pipeline first.")

# Footer
st.markdown("---")
st.caption("WikiWatch - Real-time Wikipedia monitoring and Q&A")
