import streamlit as st
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('auditrol')

# Set page configuration
st.set_page_config(
    page_title="Auditrol Testing Arena",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Main page header and description
    st.title("🔍 Auditrol Testing Arena")
    
    # Introduction section
    st.markdown("""
    ## Welcome to Auditrol Testing Arena
    
    This platform provides tools for testing and evaluating audit trail and compliance monitoring solutions.
    
    ### Available Modules:
    - **👀 Duplicate Detection Module**: Find semantic duplicates in your data
    - **🚀 Run Code**: Execute and test custom code snippets
    
    Use the sidebar navigation to explore different modules and features.
    """)
    
    # Features overview
    with st.container():
        st.markdown("## 🛠️ Features")
        
        # Display features in a 2x2 grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👀 Duplicate Detection")
            st.markdown("""
            - Find semantic duplicates in data
            - Supports multiple embedding models
            - Visualize similarity clusters
            - Export results for analysis
            """)
            
        with col2:
            st.markdown("### 🚀 Code Execution")
            st.markdown("""
            - Test custom code snippets
            - Run data analysis workflows
            - Interactive Python environment
            - View execution results in real-time
            """)
    
    # About section
    with st.expander("ℹ️ About This Project"):
        st.markdown("""
        ### Auditrol Testing Arena
        
        A platform for testing and evaluating audit trail and compliance monitoring solutions.
        
        This platform provides tools for semantic duplicate detection, code testing, and more.
        
        For more information, check the documentation in the README files.
        """)

if __name__ == "__main__":
    main()
