import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import train_ranker
from src.pipeline import WardrobePipeline

# Add src to PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page Config
st.set_page_config(
    page_title="Model 4 Demonstration",
    layout="wide"
)

# Custom Styles
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .outfit-score {
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_resource
def load_model():
    """Initializes the pipeline only once to avoid reloading CLIP every time."""
    with st.spinner("Loading AI Model & Wardrobe (this may take a moment)..."):
        pipeline = WardrobePipeline()
    return pipeline

def draw_features_chart(features):
    """Draws a simple bar chart for the 4 key metrics."""
    labels = ['Style Fit', 'Coherence', 'Color', 'Balance']
    fig, ax = plt.subplots(figsize=(4, 2))
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    bars = ax.bar(labels, features, color=colors, zorder=3)
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(zorder=0, linestyle="--", alpha=0.75, linewidth=1)
    return fig

# Main App

st.title("Model 4 Demonstration")

# Initialize Pipeline
try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# Sidebar: Configuration
st.sidebar.header("Configuration")

# BLOCK 1: MODEL & WEIGHTS
st.sidebar.subheader("Model & Weights")

# Training Button
if st.sidebar.button("Train New Weights", type="primary", help="Retrain model on data/user/items"):
    with st.spinner("Training model..."):
        try:
            train_ranker.train_ranker()
            st.cache_resource.clear()
            pipeline = load_model()
            st.sidebar.success("Training Complete!")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

# Weights Display
if hasattr(pipeline, 'weights'):
    st.sidebar.caption("Current Learned Priorities:")
    weights = pipeline.weights
    w_vals = weights[:4] if len(weights) >= 4 else [0.25]*4
    
    weight_df = pd.DataFrame({
        'Metric': ['Style', 'Coherence', 'Color', 'Balance'],
        'Weight': w_vals
    })
    st.sidebar.dataframe(weight_df, hide_index=True, use_container_width=True)

st.sidebar.markdown("---")

# BLOCK 2: RANKING SETTINGS
st.sidebar.subheader("Ranking Options")

diversity_lambda = st.sidebar.slider(
    "Diversity (MMR Lambda)", 
    min_value=0.0, max_value=1.0, value=0.6,
    help="Higher = Focus on Score. Lower = Focus on Uniqueness."
)

top_k = st.sidebar.slider("Number of Outfits", 1, 10, 3)


# Tabs
tab1, tab2 = st.tabs(["Outfit Generator", "Wardrobe Browser"])

# TAB 1: GENERATOR
with tab1:
    st.subheader("Generate Recommendations")
    with st.form("generation_form"):
        st.write("Click below to run the AI ranking on your wardrobe.")
        submitted = st.form_submit_button("Run AI Ranking", type="primary")
        
    if submitted:
        with st.spinner("Analyzing combinations..."):
            results = pipeline.run_ranking(top_k=top_k, diversity_lambda=diversity_lambda)
        
        if not results:
            st.warning("No outfits found. Please check if 'data/test/items' has images.")
        
        for i, outfit in enumerate(results):
            score = outfit['score']
            items = outfit['items'] 
            features = outfit['features'] 
            
            st.markdown("---")
            cols = st.columns([1, 3, 2])
            
            with cols[0]:
                st.markdown(f"### Rank #{i+1}")
                st.markdown(f"<div class='outfit-score'>Score: {score:.3f}</div>", unsafe_allow_html=True)
            
            with cols[1]:
                img_cols = st.columns(3)
                for idx, item in enumerate(items):
                    with img_cols[idx]:
                        path = item['path']
                        if os.path.exists(path):
                            st.image(path, caption=item['id'].split('.')[0], use_container_width=True)
                        else:
                            st.info(f"Missing File:\n{item['id']}")
            
            with cols[2]:
                st.caption("Aesthetic Metrics")
                st.pyplot(draw_features_chart(features))
                with st.expander("View Details"):
                    st.write(f"**Coherence:** {features[1]:.2f}")
                    st.write(f"**Color Harmony:** {features[2]:.2f}")
                    st.write(f"**Visual Balance:** {features[3]:.2f}")

# TAB 2: WARDROBE
with tab2:
    st.subheader("Current Wardrobe Inventory")
    st.caption(f"Loading from: data/test/items/")
    
    categories = ['top', 'bottom', 'shoes']
    for cat in categories:
        st.markdown(f"### {cat.capitalize()}s")
        items = pipeline.wardrobe.get(cat, [])
        
        if not items:
            st.info(f"No {cat}s found.")
            continue
            
        # Display first 5 items
        visible_items = items[:5]
        cols = st.columns(5)
        for i, item in enumerate(visible_items):
            with cols[i]:
                if os.path.exists(item['path']):
                    st.image(item['path'], caption=item['id'], use_container_width=True)
                else:
                    st.text(f"Mock: {item['id']}")

        # Display the rest in an expander
        remaining_items = items[5:]
        if remaining_items:
            with st.expander(f"Show {len(remaining_items)} more {cat}s..."):
                cols_hidden = st.columns(5)
                for j, item in enumerate(remaining_items):
                    with cols_hidden[j % 5]:
                        if os.path.exists(item['path']):
                            st.image(item['path'], caption=item['id'], use_container_width=True)
                        else:
                            st.text(f"Mock: {item['id']}")