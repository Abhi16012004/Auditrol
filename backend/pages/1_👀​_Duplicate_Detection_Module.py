import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import logging
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('duplicate_detector')

# Configure page settings
st.set_page_config(page_title="Semantic Duplicate Detection", layout="wide")


class DuplicateDetectionService:
    # Available models with their dimensions and performance characteristics
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "sts_score": 84,
            "description": "Fast, efficient model for prototyping and production (384 dimensions)",
            "provider": "sentence-transformers"
        },
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "sts_score": 86,
            "description": "Better semantic accuracy with moderate compute (768 dimensions)",
            "provider": "sentence-transformers"
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "dimensions": 384,
            "sts_score": 83,
            "description": "Multilingual support with good performance (384 dimensions)",
            "provider": "sentence-transformers"
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "sts_score": 91,
            "description": "State-of-the-art embeddings, highest accuracy (3072 dimensions)",
            "provider": "openai"
        },
        "text-embedding-3-small": {
            "dimensions": 1024,
            "sts_score": 89,
            "description": "Good balance of performance and speed (1024 dimensions)",
            "provider": "openai"
        }
    }
    
    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.75,
        embedding_strategy: str = "combined",  # 'combined' or 'separate'
        openai_api_key: str = st.secrets["OPENAI_API_KEY"]
    ):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.model_name = model_name
        self.threshold = threshold
        self.embedding_strategy = embedding_strategy
        
        # Initialize the appropriate embedding model
        if model_name in self.AVAILABLE_MODELS:
            provider = self.AVAILABLE_MODELS[model_name].get("provider", "sentence-transformers")
            logger.info(f"Initializing model: {model_name} with provider: {provider}")
            
            if provider == "sentence-transformers":
                logger.info(f"Loading SentenceTransformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.embedding_provider = "sentence-transformers"
            elif provider == "openai":
                if not openai_api_key:
                    logger.error("OpenAI API key is required but not provided")
                    raise ValueError("OpenAI API key is required for OpenAI models")
                
                logger.info(f"Using OpenAI embedding model: {model_name}")
                self.embedding_provider = "openai"
                # Store API key for later use
                self.openai_api_key = openai_api_key
                logger.info(f"{self.openai_api_key}")
                
                # Check if project ID is included in API key format (sk-proj-*)
                if openai_api_key and openai_api_key.startswith('sk-proj-'):
                    logger.info("Using OpenAI project-based API key")
                    
                # Note: We'll initialize the client when needed in get_embeddings
        else:
            # Default to SentenceTransformer
            logger.info(f"Loading default SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_provider = "sentence-transformers"

    def combine_text_columns(
        self, df: pd.DataFrame, columns: List[str]
    ) -> List[str]:
        """Combine multiple columns into a single text for embedding."""
        return df[columns].astype(str).agg(" | ".join, axis=1).tolist()

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            logger.warning("Empty texts list provided for embedding")
            return np.array([])
            
        start_time = time.time()
        
        if self.embedding_provider == "sentence-transformers":
            logger.info(f"Getting embeddings for {len(texts)} texts with SentenceTransformer")
            embeddings = self.model.encode(texts)
            logger.info(f"Embeddings generated in {time.time() - start_time:.2f} seconds")
            return embeddings
        elif self.embedding_provider == "openai":
            # Use OpenAI's embedding API
            try:
                logger.info(f"Getting embeddings for {len(texts)} texts with OpenAI API")
                
                # Configure OpenAI client with project ID if needed
                client = OpenAI(api_key=self.openai_api_key)
                
                # Set up the project ID if provided
                if hasattr(self, 'project_id') and self.project_id:
                    logger.info(f"Using project ID: {self.project_id}")
                    # For project-specific API, this is handled through the API key itself
                    # Project-based API keys (sk-proj-*) already have the project context
                
                # Break texts into smaller batches if needed
                max_batch_size = 100  # OpenAI recommends smaller batches
                all_embeddings = []
                
                for i in range(0, len(texts), max_batch_size):
                    batch = texts[i:i + max_batch_size]
                    logger.info(f"Processing batch {i//max_batch_size + 1} with {len(batch)} texts")
                    
                    response = client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                logger.info(f"OpenAI embeddings generated in {time.time() - start_time:.2f} seconds")
                return np.array(all_embeddings)
                
            except Exception as e:
                # Check for specific OpenAI errors
                error_msg = str(e)
                logger.error(f"Error with OpenAI embeddings: {error_msg}")
                
                # Show appropriate error messages based on error type
                if "insufficient_quota" in error_msg:
                    logger.error("OpenAI quota exceeded. Check billing details or upgrade plan.")
                    st.error("""
                    ### ⚠️ OpenAI API Quota Exceeded
                    
                    Your OpenAI API key doesn't have sufficient quota for making embedding requests.
                    
                    **Possible solutions:**
                    1. Check your [OpenAI billing page](https://platform.openai.com/account/billing/overview)
                    2. Add a payment method if you're on a free tier
                    3. Use a different model (SentenceTransformer models are free)
                    
                    Using SentenceTransformer fallback model instead...
                    """)
                elif "rate limit" in error_msg.lower():
                    logger.error("OpenAI API rate limit exceeded.")
                    st.warning("⚠️ OpenAI API rate limit exceeded. Using fallback model...")
                else:
                    st.error(f"⚠️ OpenAI API error: {error_msg}")
                    
                # Fallback to SentenceTransformer if OpenAI fails
                logger.warning("Falling back to default SentenceTransformer model")
                
                # Use all-MiniLM-L6-v2 as it's a good balance of performance and speed
                fallback_model_name = "all-MiniLM-L6-v2"
                logger.info(f"Loading fallback model: {fallback_model_name}")
                
                try:
                    fallback_model = SentenceTransformer(fallback_model_name)
                    fallback_embeddings = fallback_model.encode(texts)
                    logger.info(f"Successfully generated fallback embeddings")
                    return fallback_embeddings
                except Exception as fallback_error:
                    logger.error(f"Error with fallback model: {str(fallback_error)}")
                    st.error("Failed to use fallback model. Please try again later.")
                    # Return empty array as last resort
                    return np.zeros((len(texts), 384))  # MiniLM has 384 dimensions

    def create_clusters(
        self, texts: List[str], embeddings: np.ndarray
    ) -> List[tuple]:
        clusters = []
        for idx, embedding in enumerate(embeddings):
            found_cluster = False
            for cluster in clusters:
                sim_score = cosine_similarity([embedding], [cluster[0]]).flatten()[0]
                if sim_score >= self.threshold:
                    cluster[1].append(texts[idx])
                    found_cluster = True
                    break
            if not found_cluster:
                clusters.append((embedding, [texts[idx]]))
        return clusters

    def compute_group_percentages(self, df: pd.DataFrame) -> Dict:
        group_counts = df["hc_group_id"].value_counts().to_dict()
        total_records = len(df)
        return {
            gid: round((count / total_records) * 100, 2)
            for gid, count in group_counts.items()
        }

    def compute_group_centroids(self, df: pd.DataFrame) -> Dict:
        group_centroids = {}
        for gid in df["hc_group_id"].unique():
            group_vectors = np.vstack(
                df[df["hc_group_id"] == gid]["embedding"].values
            )
            centroid = np.mean(group_vectors, axis=0)
            group_centroids[gid] = centroid
        return group_centroids

    def compute_relative_similarity(
        self, embedding: np.ndarray, group_id: int, group_centroids: Dict
    ) -> float:
        centroid = group_centroids[group_id]
        sim = cosine_similarity([embedding], [centroid]).flatten()[0]
        return round(sim * 100, 2)

    def get_column_embeddings(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
        """Generate embeddings for each column separately with metadata."""
        column_embeddings = {}
        
        for column in columns:
            if column in df.columns:
                # Clean text data
                texts = df[column].astype(str).tolist()
                # Generate embeddings
                embeddings = self.get_embeddings(texts)
                # Store with metadata
                column_embeddings[column] = {
                    'texts': texts,
                    'embeddings': embeddings,
                    'weight': 1.0  # Default weight, could be adjusted by column importance
                }
                
        return column_embeddings
    
    def calculate_similarity_matrix(self, embeddings1: np.ndarray, embeddings2: np.ndarray = None) -> np.ndarray:
        """Calculate similarity matrix between two sets of embeddings."""
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        # Calculate cosine similarity
        return cosine_similarity(embeddings1, embeddings2)
    
    def find_duplicates_from_separate_embeddings(
        self, column_embeddings: Dict[str, Dict], df: pd.DataFrame
    ) -> Dict:
        """Find duplicates using separate column embeddings."""
        # Initialize results
        row_similarities = np.zeros((len(df), len(df)))
        
        # For each column, calculate similarities and update overall matrix
        for column, data in column_embeddings.items():
            # Calculate similarity for this column
            sim_matrix = self.calculate_similarity_matrix(data['embeddings'])
            
            # Add to overall similarity
            row_similarities += sim_matrix
        
        # Normalize by number of columns
        if len(column_embeddings) > 0:
            row_similarities /= len(column_embeddings)
        
        # Find clusters
        clusters = []
        processed = set()
        
        for i in range(len(df)):
            if i in processed:
                continue
                
            # Find rows similar to this one
            similar_indices = []
            for j in range(len(df)):
                if i != j and row_similarities[i, j] >= self.threshold:
                    similar_indices.append(j)
            
            # If we found similar rows, create a cluster
            if similar_indices:
                cluster = [i] + similar_indices
                clusters.append(cluster)
                processed.update(cluster)
            else:
                clusters.append([i])
                processed.add(i)
        
        # Map rows to group IDs
        row_to_group = {}
        for group_id, cluster in enumerate(clusters, start=1):
            for row_idx in cluster:
                row_to_group[row_idx] = group_id
        
        return {
            'row_similarities': row_similarities, 
            'clusters': clusters, 
            'row_to_group': row_to_group
        }
        
    def detect_duplicates(
        self, df: pd.DataFrame, columns: List[str] = None
    ) -> List[Dict]:
        """Detect duplicates in a dataframe using either combined or separate embedding strategies."""
        if columns is None:
            columns = ["hc_type", "hc_name", "hc_description", "hc_class"]

        # Filter out None values and use only available columns
        available_columns = [col for col in columns if col in df.columns]
        if not available_columns:
            raise ValueError(
                "None of the specified columns found in DataFrame"
            )

        if self.embedding_strategy == 'separate':
            # Strategy: Embed each column separately
            column_embeddings = self.get_column_embeddings(df, available_columns)
            
            # Find duplicates using separate embeddings
            duplicate_data = self.find_duplicates_from_separate_embeddings(
                column_embeddings, df
            )
            
            # Assign group IDs
            df["hc_group_id"] = df.index.map(lambda i: duplicate_data['row_to_group'].get(i, -1))
            
            # Create combined text for reporting purposes
            df["combined_text"] = self.combine_text_columns(df, available_columns)
            
            # Get embeddings for similarity calculations
            df["embedding"] = list(self.get_embeddings(df["combined_text"].tolist()))
            
        else:
            # Default strategy: Combine columns and get embeddings
            combined_texts = self.combine_text_columns(df, available_columns)
            unique_texts = list(set(combined_texts))
            unique_embeddings = self.get_embeddings(unique_texts)

            # Create clusters
            clusters = self.create_clusters(unique_texts, unique_embeddings)

            # Map combined text to group_id
            text_to_group = {}
            for group_id, cluster in enumerate(clusters, start=1):
                for text in cluster[1]:
                    text_to_group[text] = group_id

            # Assign group IDs
            df["combined_text"] = combined_texts
            df["hc_group_id"] = df["combined_text"].map(text_to_group)

            # Get embeddings for all rows
            df["embedding"] = list(self.get_embeddings(combined_texts))

        # Calculate group percentages
        group_percentages = self.compute_group_percentages(df)

        # Calculate group centroids
        group_centroids = self.compute_group_centroids(df)

        results = []
        for idx, row in df.iterrows():
            group_id = row["hc_group_id"]
            relative_similarity = self.compute_relative_similarity(
                row["embedding"], group_id, group_centroids
            )
            group_percentage = group_percentages[group_id]

            results.append({
                "index": idx,
                "hc_group_id": group_id,
                "hc_group_percentage": group_percentage,
                "relative_similarity": relative_similarity,
                "match_percentage": relative_similarity,
            })

        # Clean up temporary columns
        df.drop(columns=["embedding", "combined_text"], inplace=True)

        return results


def render_similarity_badge(value):
    """Render a colored badge for the similarity percentage"""
    if value >= 90:
        return f"🟢 {value}%"  # Green circle for high similarity
    elif value >= 80:
        return f"🔵 {value}%"  # Blue circle for good similarity
    elif value >= 70:
        return f"🟡 {value}%"  # Yellow circle for moderate similarity
    else:
        return f"🔴 {value}%"  # Red circle for low similarity


def generate_group_comparison_chart(df):
    """Generate a bar chart comparing the sizes of different groups"""
    group_counts = df.groupby('hc_group_id').size().reset_index(name='count')
    group_counts = group_counts.sort_values('count', ascending=False)
    
    fig = px.bar(
        group_counts, 
        x='hc_group_id', 
        y='count',
        labels={'count': 'Number of Records', 'hc_group_id': 'Group ID'},
        title='Group Size Comparison',
        color='count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis=dict(type='category'),
        width=800,
        height=500
    )
    
    return fig


def generate_similarity_heatmap(df, group_id=None, model_name="all-MiniLM-L6-v2"):
    """Generate a heatmap of similarities within a group"""
    # Filter by group if specified
    if group_id is not None:
        df = df[df['hc_group_id'] == group_id]
    
    if len(df) <= 1:
        return None
    
    # We need embeddings for the heatmap
    service = DuplicateDetectionService(model_name=model_name)
    if 'combined_text' not in df.columns:
        # Use visible columns for similarity
        text_cols = [c for c in df.columns if df[c].dtype == 'object']
        combined_texts = service.combine_text_columns(df, text_cols)
        embeddings = service.get_embeddings(combined_texts)
    else:
        embeddings = service.get_embeddings(df['combined_text'].tolist())
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Sample if too many items
    n = sim_matrix.shape[0]
    max_items = 50
    if n > max_items:
        indices = np.random.choice(n, size=max_items, replace=False)
        sim_subset = sim_matrix[np.ix_(indices, indices)]
        labels = [f"Item {i}" for i in indices]
    else:
        sim_subset = sim_matrix
        labels = [f"Item {i}" for i in range(n)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sim_subset,
        x=labels,
        y=labels,
        hoverongaps=False,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Similarity')
    ))
    
    fig.update_layout(
        title=f"Similarity Heatmap {f'(Group {group_id})' if group_id else ''}",
        xaxis_title="Items",
        yaxis_title="Items",
        width=800,
        height=800
    )
    
    return fig


def main():
    st.title("Semantic Duplicate Detection")
    
    with st.expander("ℹ️ About this module", expanded=False):
        st.markdown("""
        ### Semantic Duplicate Detection
        
        This module finds semantic duplicates in your data based on the meanings of text, not just exact matches.
        
        #### How it works:
        1. **Upload your CSV file** with text data
        2. **Select columns** to analyze for duplicates
        3. **Choose embedding model and strategy**
        4. **Adjust similarity threshold** (higher = stricter matching)
        5. **Process** and view groups of similar items
        
        The system uses sentence embeddings and cosine similarity to identify semantically similar content.
        """)
    
        # Model selection sidebar
    with st.sidebar:
        st.header("Model Settings")
        
        # Get available models
        service = DuplicateDetectionService()
        model_options = list(service.AVAILABLE_MODELS.keys())
        
        # Add provider info to display name
        model_display_options = []
        for model in model_options:
            provider = service.AVAILABLE_MODELS[model].get("provider", "")
            sts_score = service.AVAILABLE_MODELS[model].get("sts_score", "")
            dimensions = service.AVAILABLE_MODELS[model].get("dimensions", "")
            
            # Add free/paid label
            cost_label = "💰 Paid" if provider == "openai" else "✅ Free"
            
            display_name = f"{model} | STS: {sts_score} | {dimensions}d | {cost_label}"
            model_display_options.append(display_name)
        
        # Create a mapping from display name to actual model name
        model_name_map = {display: model for display, model in zip(model_display_options, model_options)}
        
        selected_model_display = st.selectbox(
            "Select Embedding Model",
            options=model_display_options,
            index=0,
            help="Choose a model for generating embeddings. Higher STS score = better accuracy."
        )
        
        # Get the actual model name from the display name
        selected_model = model_name_map[selected_model_display]
        
        # Provider-specific info
        provider = service.AVAILABLE_MODELS[selected_model].get("provider", "")
        if provider == "openai":
            st.warning("⚠️ This is a paid model requiring OpenAI API credits")
        else:
            st.success("✅ This is a free model with no API key required")
        
        # Show model details
        st.info(f"**{selected_model}**\n\n{service.AVAILABLE_MODELS[selected_model]['description']}")        # Check if we need to ask for OpenAI API key
        if service.AVAILABLE_MODELS[selected_model].get("provider") == "openai":
            # Initialize session state for API key and project ID
            if "openai_api_key" not in st.session_state:
                st.session_state.openai_api_key = ""
            
            if "openai_project_id" not in st.session_state:
                st.session_state.openai_project_id = ""
                
            # Add project ID input field
            st.markdown("### OpenAI API Settings")
                
            openai_api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                value=st.session_state.openai_api_key,
                help="Required for using OpenAI's embedding models",
                key="openai_api_key_sidebar"
            )
            
            project_id = st.text_input(
                "OpenAI Project ID (Optional)",
                value=st.session_state.openai_project_id,
                help="Project ID if using project-scoped API keys",
                key="openai_project_id_sidebar"
            )
            
            # Store in session state for persistence
            st.session_state.openai_api_key = openai_api_key
            st.session_state.openai_project_id = project_id
            
            if openai_api_key:
                st.success("✅ API key provided")
                
                # Show model cost estimate
                dimensions = service.AVAILABLE_MODELS[selected_model].get("dimensions", 0)
                st.info(f"""
                **Model Information:**
                - Dimensions: {dimensions}
                - STS Score: {service.AVAILABLE_MODELS[selected_model].get('sts_score', 'N/A')}
                - Approx. Cost: {'$$$' if dimensions > 1000 else '$$' if dimensions > 500 else '$'}
                """)
            else:
                st.warning("⚠️ API key required for OpenAI models")
        else:
            openai_api_key = None
        
        # Embedding strategy selection
        embedding_strategy = st.radio(
            "Embedding Strategy",
            options=["combined", "separate"],
            index=0,
            help="Combined: Join all columns for one embedding. Separate: Create embeddings for each column separately."
        )
        
        st.divider()
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded file with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Display sample of the data
            with st.expander("Preview uploaded data", expanded=True):
                st.dataframe(df.head(5))
            
            # Column selection
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            if not text_columns:
                st.error("No text columns found in the uploaded data. Please upload a CSV with text columns.")
            else:
                # Allow for multi-column selection
                col_selection_type = st.radio(
                    "Column selection mode:",
                    ["Single Column", "Multiple Columns"],
                    help="Choose whether to analyze a single column or combine multiple columns"
                )
                
                if col_selection_type == "Single Column":
                    selected_columns = [st.selectbox(
                        "Select column for duplicate analysis:", 
                        options=text_columns,
                        index=0
                    )]
                else:
                    selected_columns = st.multiselect(
                        "Select columns for duplicate analysis (will be combined):",
                        options=text_columns,
                        default=[text_columns[0]] if text_columns else []
                    )
                    
                    if not selected_columns:
                        st.warning("Please select at least one column for analysis")
                        st.stop()
                
                # Advanced options section
                with st.expander("Advanced Options", expanded=False):
                    # Model selection
                    st.markdown("#### Model Selection")
                    models = list(DuplicateDetectionService.AVAILABLE_MODELS.keys())
                    model_descriptions = [
                        f"{model} - {DuplicateDetectionService.AVAILABLE_MODELS[model]['description']}" 
                        for model in models
                    ]
                    
                    selected_model_idx = st.selectbox(
                        "Select Embedding Model:", 
                        range(len(models)),
                        format_func=lambda i: model_descriptions[i],
                        help="Choose the model used for generating embeddings"
                    )
                    selected_model = models[selected_model_idx]
                    
                    # Embedding strategy
                    st.markdown("#### Embedding Strategy")
                    embedding_strategy = st.radio(
                        "Embedding Strategy:",
                        ["combined", "separate"],
                        index=0,
                        help=(
                            "Combined: Create single embedding for all columns. "
                            "Separate: Create separate embeddings for each column."
                        ),
                        format_func=lambda x: {
                            "combined": "Combined Columns (Faster)",
                            "separate": "Separate Columns (More Accurate)"
                        }.get(x, x)
                    )
                    
                    # Show model details
                    st.markdown("#### Selected Model Details")
                    model_details = DuplicateDetectionService.AVAILABLE_MODELS[selected_model]
                    model_cols = st.columns(3)
                    with model_cols[0]:
                        st.metric("Dimensions", model_details["dimensions"])
                    with model_cols[1]:
                        st.metric("STS Score", model_details["sts_score"])
                    with model_cols[2]:
                        if embedding_strategy == "separate":
                            processing_speed = "Slower"
                        else:
                            processing_speed = "Standard"
                        st.metric("Processing Speed", processing_speed)
                
                # Threshold slider
                threshold = st.slider(
                    "Similarity Threshold", 
                    min_value=0.5, 
                    max_value=1.0, 
                    value=0.75, 
                    step=0.01,
                    help="Higher values create more groups with higher similarity within each group"
                )
                
                # Process button
                if st.button("Detect Duplicates"):
                    with st.spinner("Processing duplicates..."):
                        # Initialize service with selected parameters
                        with st.spinner(f"Loading {selected_model} model..."):
                            # Check if OpenAI API key is needed
                            model_provider = DuplicateDetectionService.AVAILABLE_MODELS.get(selected_model, {}).get("provider", "")
                            logger.info(f"Selected model provider: {model_provider}")
                            
                            if model_provider == "openai":
                                st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
                                st.session_state.openai_project_id = "proj_1vC4wtwdBffN4rNNfnECigmk"
                                # Use API key from session state (set in sidebar)
                                openai_api_key = st.session_state.get("openai_api_key", "")
                                project_id = st.session_state.get("openai_project_id", "")
                                
                                if not openai_api_key:
                                    logger.warning("OpenAI API key not found in session state")
                                    st.error("⚠️ Please enter your OpenAI API key in the sidebar first")
                                    st.stop()
                                else:
                                    logger.info("Using OpenAI API key from session state")
                                    if project_id:
                                        logger.info(f"Using project ID: {project_id}")
                            else:
                                openai_api_key = None
                                project_id = None
                                logger.info("No OpenAI API key needed for this model")
                                
                            try:
                                # Create service with appropriate parameters
                                service_kwargs = {
                                    "threshold": threshold,
                                    "model_name": selected_model,
                                    "embedding_strategy": embedding_strategy
                                }
                                
                                if openai_api_key:
                                    service_kwargs["openai_api_key"] = openai_api_key
                                    
                                service = DuplicateDetectionService(**service_kwargs)
                                
                                # Add project ID if provided (for OpenAI API calls)
                                if project_id and hasattr(service, 'embedding_provider') and service.embedding_provider == "openai":
                                    service.project_id = project_id
                            
                            except Exception as e:
                                logger.error(f"Error initializing service: {str(e)}")
                                st.error(f"⚠️ Error initializing service: {str(e)}")
                                st.stop()
                        
                        # Process the data
                        try:
                            # Make a copy to avoid modifying the original
                            df_copy = df.copy()
                            
                            # Detect duplicates
                            with st.spinner(f"Processing using {embedding_strategy} strategy..."):
                                results = service.detect_duplicates(df_copy, selected_columns)
                            
                            # Add results back to the dataframe
                            result_df = df.copy()
                            for res in results:
                                idx = res["index"]
                                result_df.loc[idx, "hc_group_id"] = res["hc_group_id"]
                                result_df.loc[idx, "hc_group_percentage"] = res["hc_group_percentage"]
                                result_df.loc[idx, "relative_similarity"] = res["relative_similarity"]
                            
                            # Store results in session state
                            st.session_state['duplicate_results'] = result_df
                            
                            # Show success message
                            st.success(f"Processing complete! Found {result_df['hc_group_id'].nunique()} groups.")
                            
                            # Create tabs for different sections
                            result_tabs = st.tabs(["Results", "Visualizations", "Export"])
                            
                            with result_tabs[0]:  # Results Tab
                                st.subheader("Duplicate Detection Results")
                                
                                # Summary metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Records", len(result_df))
                                with col2:
                                    st.metric("Number of Groups", result_df["hc_group_id"].nunique())
                                with col3:
                                    avg_similarity = round(result_df["relative_similarity"].mean(), 2)
                                    st.metric("Average Similarity", f"{avg_similarity}%")
                                
                                # Display the results with group info
                                st.markdown("### Grouped Data")
                                
                                # Format the columns for display
                                display_df = result_df.copy()
                                
                                # Format similarity badges
                                display_df["relative_similarity"] = display_df["relative_similarity"].apply(
                                    render_similarity_badge
                                )
                                
                                # Add group size column
                                group_sizes = result_df["hc_group_id"].value_counts().to_dict()
                                display_df["group_size"] = display_df["hc_group_id"].map(group_sizes)
                                
                                # Reorder columns for better display
                                key_cols = [
                                    "hc_group_id", "group_size", "hc_group_percentage", "relative_similarity"
                                ]
                                # Display selected columns first, then other columns
                                display_cols = key_cols + selected_columns
                                other_cols = [c for c in display_df.columns if c not in display_cols]
                                display_cols.extend(other_cols)
                                
                                # Display interactive table with highlighted groups
                                st.dataframe(
                                    display_df[display_cols].style.apply(
                                        lambda x: ["background-color: #f0f8ff" if i % 2 == 0 else "" 
                                                for i in range(len(x))], 
                                        axis=0
                                    ),
                                    use_container_width=True,
                                    height=500
                                )
                                
                                # Group Summary
                                st.markdown("### Group Summary")
                                count_column = selected_columns[0]
                                group_summary = result_df.groupby("hc_group_id").agg(
                                    count=pd.NamedAgg(column=count_column, aggfunc="count"),
                                    percentage=pd.NamedAgg(column="hc_group_percentage", aggfunc="first"),
                                    avg_similarity=pd.NamedAgg(column="relative_similarity", aggfunc="mean")
                                ).sort_values(by="count", ascending=False).reset_index()
                                
                                group_summary["avg_similarity"] = group_summary["avg_similarity"].round(2).apply(
                                    lambda x: render_similarity_badge(x)
                                )
                                
                                # Add sample values from each group
                                def get_sample_values(group_id):
                                    values = result_df[result_df["hc_group_id"] == group_id][selected_columns[0]].unique()
                                    return ", ".join([str(v) for v in values[:3]]) + ("..." if len(values) > 3 else "")
                                
                                group_summary["sample_values"] = group_summary["hc_group_id"].apply(get_sample_values)
                                
                                st.dataframe(
                                    group_summary.rename(columns={
                                        "count": "Group Size",
                                        "percentage": "% of Total",
                                        "avg_similarity": "Avg Similarity",
                                        "sample_values": "Sample Values"
                                    }),
                                    use_container_width=True,
                                    height=400
                                )
                            
                            with result_tabs[1]:  # Visualizations Tab
                                st.subheader("Visualizations")
                                
                                # Group comparison chart
                                st.markdown("### Group Size Comparison")
                                group_chart = generate_group_comparison_chart(result_df)
                                st.plotly_chart(group_chart, use_container_width=True)
                                
                                # Group-specific heatmap
                                st.markdown("### Group Similarity Heatmap")
                                groups = sorted(result_df["hc_group_id"].unique())
                                group_labels = [
                                    f"Group {g} ({group_sizes[g]} items)" 
                                    for g in groups
                                ]
                                
                                selected_idx = st.selectbox(
                                    "Select group to visualize:", 
                                    range(len(groups)),
                                    format_func=lambda i: group_labels[i]
                                )
                                
                                selected_group = groups[selected_idx]
                                with st.spinner(f"Generating heatmap for Group {selected_group}..."):
                                    heatmap = generate_similarity_heatmap(
                                        result_df, 
                                        selected_group,
                                        model_name=selected_model
                                    )
                                    if heatmap:
                                        st.plotly_chart(heatmap, use_container_width=True)
                                    else:
                                        st.info(f"Group {selected_group} has too few items for visualization")
                            
                            with result_tabs[2]:  # Export Tab
                                st.subheader("Export Results")
                                
                                # Export options
                                csv_export = result_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Results CSV",
                                    data=csv_export,
                                    file_name="duplicate_detection_results.csv",
                                    mime="text/csv"
                                )
                                
                                # Export group summary
                                summary_export = group_summary.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Group Summary CSV",
                                    data=summary_export,
                                    file_name="group_summary.csv",
                                    mime="text/csv"
                                )
                        
                        except Exception as e:
                            st.error(f"Error processing duplicates: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        # Example when no file is uploaded
        st.markdown("""
        ### Upload a CSV file to begin
        
        Your file should contain at least one text column that you want to analyze for duplicates.
        
        #### Example format:
        | ID | Description | Category | Value |
        |----|-------------|----------|-------|
        | 1  | KYC Check   | Finance  | 1000  |
        | 2  | Know Your Customer | Finance | 1000 |
        | 3  | AML Check   | Security | 2000  |
        | 4  | Transaction Monitoring | Security | 3000 |
        """)


if __name__ == "__main__":
    main()
