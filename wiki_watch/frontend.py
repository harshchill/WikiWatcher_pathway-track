import time
import requests
import streamlit as st
from datetime import datetime

# Set up the Streamlit page
st.set_page_config(
    page_title="WikiWatch",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    .highlight {
        color: #FFC107;
        font-weight: 600;
    }
    .subtitle {
        font-size: 1rem;
        opacity: 0.8;
        margin-bottom: 2rem;
    }
    .stButton>button {
        border-radius: 4px;
    }
    .sidebar-header {
        font-weight: 600;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Define the API URL
API_URL = "http://localhost:8000"

# Check if the API is available
def is_api_available():
    """Check if the backend API is available."""
    try:
        response = requests.get(f"{API_URL}/status", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_pipeline():
    """Start the Wikipedia RecentChanges pipeline."""
    if not is_api_available():
        return {"status": "error", "message": "Backend service is not available"}
    
    try:
        response = requests.post(f"{API_URL}/start_pipeline", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Failed to start pipeline: {str(e)}"}

def stop_pipeline():
    """Stop the Wikipedia RecentChanges pipeline."""
    if not is_api_available():
        return {"status": "error", "message": "Backend service is not available"}
    
    try:
        response = requests.post(f"{API_URL}/stop_pipeline", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Failed to stop pipeline: {str(e)}"}

def get_status():
    """Get the current status of the pipeline."""
    if not is_api_available():
        return {"status": "error", "pipeline_running": False, "changes_count": 0, "message": "Backend service is not available"}
    
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "pipeline_running": False, "changes_count": 0, "message": f"Failed to get status: {str(e)}"}

def get_recent_changes(limit=10):
    """Get the most recent changes from the pipeline."""
    if not is_api_available():
        return {"status": "error", "count": 0, "changes": [], "message": "Backend service is not available"}
    
    try:
        response = requests.get(f"{API_URL}/recent_changes?limit={limit}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "count": 0, "changes": [], "message": f"Failed to get recent changes: {str(e)}"}

def query(question):
    """Query the pipeline with a natural language question."""
    if not is_api_available():
        return {"status": "error", "question": question, "answer": "Backend service is not available. Please start the backend server first."}
    
    try:
        # URL encode the question to handle special characters
        encoded_question = requests.utils.quote(question)
        response = requests.get(f"{API_URL}/query?q={encoded_question}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "question": question, "answer": f"Error: {str(e)}"}

# Title and Logo
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Wikipedia%27s_W.svg/240px-Wikipedia%27s_W.svg.png", width=80)
with col2:
    st.markdown('<p class="main-title">Wiki<span class="highlight">Watch</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Live monitoring & intelligent insights for Wikipedia edits</p>', unsafe_allow_html=True)

# Check API availability and show a prominent message if it's down
api_available = is_api_available()
if not api_available:
    st.warning("""
    ⚠️ **Backend service is not available**
    
    The WikiWatch backend service is not running. Features like starting the pipeline, querying, 
    and viewing recent changes will not work until the backend is started.
    
    To start the backend, run: `cd wiki_watch && uvicorn app:app --reload --host 0.0.0.0 --port 8000`
    """)
    st.markdown("---")

# Sidebar for controls and status
with st.sidebar:
    st.markdown('<p class="sidebar-header">Pipeline Controls</p>', unsafe_allow_html=True)
    
    # Pipeline control buttons with improved styling
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶ Start", type="primary", disabled=not api_available)
        if start_button:
            with st.spinner("Starting pipeline..."):
                result = start_pipeline()
                if result["status"] in ["success", "info"]:
                    st.success(result["message"])
                    time.sleep(1)
                    st.rerun()
                elif result["status"] == "error":
                    st.error(result["message"])
    
    with col2:
        stop_button = st.button("⏹ Stop", type="secondary", disabled=not api_available)
        if stop_button:
            with st.spinner("Stopping pipeline..."):
                result = stop_pipeline()
                if result["status"] in ["success", "info"]:
                    st.success(result["message"])
                    time.sleep(1)
                    st.rerun()
                elif result["status"] == "error":
                    st.error(result["message"])
    
    # Status information with visual enhancements
    st.markdown('<p class="sidebar-header">Status</p>', unsafe_allow_html=True)
    status = get_status()
    
    # Backend connection status
    if not api_available:
        st.error("🔌 Backend disconnected")
    elif status["pipeline_running"]:
        st.success("🟢 Pipeline active")
    else:
        st.warning("🟠 Pipeline inactive")
    
    # Changes count
    st.metric("Changes collected", status.get("changes_count", 0))
    
    # Refresh button with icon
    if st.button("🔄 Refresh"):
        st.rerun()
        
    # Add system information
    with st.expander("System Info"):
        st.markdown(f"""
        - **Backend URL**: {API_URL}
        - **Backend Status**: {"Connected" if api_available else "Disconnected"}
        - **Frontend Version**: 1.0.0
        - **Last Updated**: {datetime.now().strftime("%Y-%m-%d")}
        """)

# Main content area - Query section
st.markdown('<h3>Ask a question about Wikipedia activity</h3>', unsafe_allow_html=True)

# Create a card-like container for the query section
query_container = st.container()
with query_container:
    st.markdown("""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; margin-bottom: 1rem;">
        <p style="font-size: 0.9rem; color: #6c757d;">
            Ask anything about recent Wikipedia edits. For example:
            <span style="color: #FFC107; font-style: italic;">
                "What recent changes were made to articles about Python?"
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # User query input with improved styling
    question = st.text_input("", 
                          placeholder="Type your question here...",
                          label_visibility="collapsed")
    
    # Query button with better styling
    if st.button("🔍 Search Wikipedia Changes", type="primary") and question:
        with st.spinner("Analyzing recent changes..."):
            result = query(question)
            
            if result["status"] == "success":
                st.markdown("""
                <div style="padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #FFC107; 
                            background-color: #f8f9fa; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                st.markdown(f"**Your answer:**", unsafe_allow_html=True)
                st.write(result["answer"])
                st.caption(f"Based on {result.get('changes_count', 0)} recent Wikipedia changes")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Failed to get an answer. Make sure the pipeline is running.")

# Recent changes section with improved styling
st.markdown('<h3>Recent Wikipedia Activity</h3>', unsafe_allow_html=True)

# Add refresh button with better styling
col1, col2 = st.columns([1, 6])
with col1:
    if st.button("🔄 Refresh", key="refresh_changes"):
        st.rerun()

# Display the recent changes with improved styling
changes_data = get_recent_changes(limit=20)
if changes_data["status"] == "success" and changes_data["count"] > 0:
    st.markdown(f"<p style='color:#6c757d;font-size:0.9rem;'>Showing latest {changes_data['count']} edits</p>", unsafe_allow_html=True)
    
    # Create a container for all changes
    changes_container = st.container()
    with changes_container:
        for i, change in enumerate(changes_data["changes"]):
            # Create a cleaner expander with highlight
            expander_title = f"{change['title']} - by {change['user']}"
            with st.expander(expander_title):
                cols = st.columns([1, 1])
                with cols[0]:
                    st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Page</span>: {change['title']}", unsafe_allow_html=True)
                    st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Editor</span>: {change['user']}", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Time</span>: {datetime.fromtimestamp(change['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}", unsafe_allow_html=True)
                    st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Diff URL</span>: [{change['diff_url'].split('/')[-1]}]({change['diff_url']})", unsafe_allow_html=True)
                
                # Comment section with special styling if available
                if change['comment']:
                    st.markdown("<div style='margin-top:0.5rem;padding:0.5rem;background-color:#f8f9fa;border-radius:0.3rem;'>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Comment</span>: {change['comment']}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
else:
    if status["pipeline_running"]:
        st.info("⏳ Waiting for changes to be collected... This may take a few moments.")
    else:
        st.warning("⚠️ No changes available. Please start the pipeline first.")

# Footer with better styling
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;padding:1rem 0;">
    <span style="color:#6c757d;font-size:0.8rem;">WikiWatch • Real-time Wikipedia monitoring</span>
    <span style="color:#6c757d;font-size:0.8rem;">Updated: {}</span>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
