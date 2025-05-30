import os
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

# Define the API URL - Configure for your environment
# Set to False to use the real backend
DEMO_MODE = False
API_URL = os.getenv("API_URL", "http://localhost:8000")  # Default API URL for local development

# Demo data for the self-contained mode
DEMO_DATA = {
    "pipeline_running": False,
    "changes": [
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
}

# Check if the API is available
def is_api_available():
    """Check if the backend API is available."""
    if DEMO_MODE:
        return True
    
    try:
        response = requests.get(f"{API_URL}/status", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_pipeline():
    """Start the Wikipedia RecentChanges pipeline."""
    if DEMO_MODE:
        DEMO_DATA["pipeline_running"] = True
        return {"status": "success", "message": "Pipeline started successfully"}
    
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
    if DEMO_MODE:
        DEMO_DATA["pipeline_running"] = False
        return {"status": "success", "message": "Pipeline stopped successfully"}
    
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
    if DEMO_MODE:
        return {
            "status": "success",
            "pipeline_running": DEMO_DATA["pipeline_running"],
            "changes_count": len(DEMO_DATA["changes"])
        }
    
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
    if DEMO_MODE:
        changes = DEMO_DATA["changes"][:limit]
        return {
            "status": "success",
            "count": len(changes),
            "changes": changes
        }
    
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
    if DEMO_MODE:
        # Generate demo responses based on the question
        question_lower = question.lower()
        
        if "python" in question_lower:
            answer = "Based on recent changes, the Python programming language article was updated by PythonFan123 with version information and library references. This suggests ongoing documentation improvements to keep the article current with the latest Python developments."
        
        elif "machine learning" in question_lower or "ai" in question_lower:
            answer = "There have been several recent edits to AI-related articles. AIResearcher updated the Machine Learning article with information about transformer models, while TechEditor42 revised the Artificial Intelligence article's ethical considerations section. AIHistorian also updated the ChatGPT article with the latest capabilities."
        
        elif "recent" in question_lower or "latest" in question_lower:
            answer = "The most recent Wikipedia changes in my database include updates to: Python (programming language), Machine Learning, Artificial Intelligence, ChatGPT, and Natural Language Processing articles. These changes primarily focus on updating technical information, adding recent developments, and improving references."
        
        else:
            answer = f"Based on the recent changes in my database, I don't have specific information about '{question}'. The most active editors recently include PythonFan123, AIResearcher, TechEditor42, AIHistorian, and NLPExpert, who have been updating technology-related articles."
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "changes_count": len(DEMO_DATA["changes"])
        }
    
    if not is_api_available():
        return {"status": "error", "question": question, "answer": "Backend service is not available. Please start the backend server first."}
    
    try:
        # URL encode the question to handle special characters
        from urllib.parse import quote
        encoded_question = quote(question)
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

# Check if we're in demo mode
api_available = is_api_available() 
if DEMO_MODE:
    st.info("""
    ℹ️ **Demo Mode Active**
    
    WikiWatch is currently running in demonstration mode with sample data.
    Start the pipeline to begin exploring the interface and try asking questions 
    about recent Wikipedia changes.
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
    question = st.text_input("Your question", 
                          placeholder="Type your question here...")
    
    # Query button with better styling
    search_button = st.button("🔍 Search Wikipedia Changes", type="primary", disabled=not api_available)
    
    if search_button and question:
        if not api_available:
            st.error("Cannot process your question. Backend service is not available.")
        else:
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
                    st.error(result.get("answer", "Failed to get an answer. Make sure the pipeline is running."))

# Recent changes section with improved styling
st.markdown('<h3>Recent Wikipedia Activity</h3>', unsafe_allow_html=True)

# Add refresh button with better styling
col1, col2 = st.columns([1, 6])
with col1:
    refresh_button = st.button("🔄 Refresh", key="refresh_changes", disabled=not api_available)
    if refresh_button:
        st.rerun()

# Display the recent changes with improved styling
if not api_available:
    st.warning("Backend service is not available. Cannot display recent changes.")
    
    # Show placeholder data visualization to demonstrate UI
    st.markdown("""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; margin-top: 1rem;">
        <p style="text-align: center; color: #6c757d;">
            <i>Preview of how recent changes will appear when backend is connected</i>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display some example data just for UI demonstration
    with st.expander("Example: Python (programming language) - by WikiUser123"):
        cols = st.columns([1, 1])
        with cols[0]:
            st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Page</span>: Python (programming language)", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Editor</span>: WikiUser123", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Time</span>: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Diff URL</span>: [View diff](https://en.wikipedia.org)", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top:0.5rem;padding:0.5rem;background-color:#f8f9fa;border-radius:0.3rem;'>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#FFC107;font-weight:500;'>Comment</span>: Updated version information and added new library references", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
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
