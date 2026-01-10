import sys
import os
import torch
import pandas as pd
import streamlit as st
import gc  # Garbage collection for RAM management

# --- PATH SETUP ---
# Ensure we can import from src/glassbox
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# ------------------

from dashboard.components import header
from glassbox.model_loader import ModelWrapper
from glassbox.tracers import get_attention_data, get_logit_lens_data
from glassbox.steering import get_steering_vector, generate_steered_response
from glassbox.evaluate import run_steering_eval
from glassbox.sae import (
    load_sae,
    get_top_features_for_text,
    get_feature_activations_for_text,
    get_sae_feature_vector
)

# 1. Page Config
st.set_page_config(page_title="Glass Box AI", layout="wide", page_icon="🧠")

# 2. Header
header()

# --- SIDEBAR: GLOBAL CONFIGURATION ---
st.sidebar.header("⚙️ Model Settings")

# A. Model Selector (Supports Small, Medium, Large)
model_options = {
    "gpt2": 12,  # Small (124M)
    "gpt2-medium": 24,  # Medium (355M)
    "gpt2-large": 36,  # Large (774M)
}
# Default to gpt2-medium if available
model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()), index=0)

# B. Dynamic Layer Slider
# The slider max value updates based on the model choice
max_layers = model_options[model_name]
sae_layer = st.sidebar.number_input(
    "SAE Layer",
    min_value=0,
    max_value=max_layers - 1,
    value=int(max_layers / 2),  # Defaults to middle layer
    help=f"Select a layer between 0 and {max_layers - 1}"
)

# C. Path Logic
# Matches the naming convention: sae_{model}_{layer}.pt
default_ckpt = f"data/sae_{model_name}_layer{sae_layer}.pt"
st.sidebar.markdown("---")
st.sidebar.header("🧬 SAE Config")
sae_path_global = st.sidebar.text_input("SAE Path", value=default_ckpt)

# D. Model Loading with RAM Management
# We check if the model changed to avoid reloading it unnecessarily
if "model" not in st.session_state or st.session_state.get("model_name") != model_name:
    with st.spinner(f"🧠 Loading {model_name}... (Please wait)"):
        # 1. Clear RAM if switching models
        if "model" in st.session_state:
            del st.session_state.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        # 2. Load New Model
        st.session_state.model = ModelWrapper.load(model_name)
        st.session_state.model_name = model_name

# Create a local reference for easier use
model = st.session_state.model

# 3. Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔥 Attention Maps", "🧐 Logit Lens", "🎮 Activation Steering", "🧬 Feature Dictionary"]
)

# --- TAB 1: ATTENTION ---
with tab1:
    st.subheader("Visualizing Attention Heads")
    text_input = st.text_input("Enter text to analyze:", "The cat sat on the mat.", key="att_input")
    if text_input:
        try:
            html_viz = get_attention_data(text_input, model_name)
            st.components.v1.html(html_viz, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Error: {e}")

# --- TAB 2: LOGIT LENS ---
with tab2:
    st.subheader("Layer-by-Layer Mind Reader")
    ll_input = st.text_input("Enter text to decode:", "The Eiffel Tower is located in", key="ll_input")
    if ll_input:
        try:
            df = get_logit_lens_data(ll_input, model_name)
            # Use columns specific to formatting
            st.dataframe(df.style.background_gradient(subset=["Top Prob"], cmap="Greens"), height=600)
        except Exception as e:
            st.error(f"Error: {e}")

# --- TAB 3: STEERING ---
with tab3:
    st.subheader("🎮 Surgical Activation Steering")
    st.markdown("Intervene on the model using raw concepts or discovered SAE features.")

    # Steering Mode Selection
    steering_mode = st.radio(
        "Steering Source:",
        ["Manual Concept (Text)", "Discovered Feature (SAE)"],
        horizontal=True
    )

    vector = None
    target_layer_final = sae_layer  # Default to SAE layer

    # MODE A: MANUAL
    if steering_mode == "Manual Concept (Text)":
        col1, col2 = st.columns(2)
        with col1:
            pos_concept = st.text_input("Positive Concept (+)", "Love")
            neg_concept = st.text_input("Negative Concept (-)", "Hate")
        with col2:
            # Here we can choose ANY layer
            manual_layer = st.number_input(
                "Injection Layer",
                0, model.cfg.n_layers - 1,
                int(sae_layer),
                key="steer_layer_manual"
            )
            multiplier = st.slider("Strength", -10.0, 10.0, 0.0, 0.5, key="steer_mult_manual")
            target_layer_final = int(manual_layer)

    # MODE B: SAE FEATURE
    else:
        col1, col2 = st.columns(2)
        with col1:
            feature_idx = st.number_input(
                "SAE Feature Index",
                min_value=0,
                value=0,  # Default 0, change to 334 etc.
                help="Enter the ID of a feature you found in the Dictionary tab."
            )
        with col2:
            # Here we MUST use the SAE layer from the sidebar
            st.info(f"Steering on Layer {sae_layer} (Defined in Sidebar)")

            # --- FIX: INCREASED RANGE TO +/- 1000.0 ---
            multiplier = st.slider(
                "Clamping Strength",
                min_value=-1000.0,
                max_value=1000.0,
                value=0.0,
                step=10.0,
                key="steer_mult_sae"
            )
            # ------------------------------------------

            target_layer_final = int(sae_layer)

    st.markdown("---")
    steer_input = st.text_input("Prompt to Steer:", "I am going to the store to buy")

    if st.button("🔴 Run Experiment & Evaluate"):
        with st.spinner("Running A/B Test..."):
            try:
                # 1. Get Vector
                if steering_mode == "Manual Concept (Text)":
                    vector = get_steering_vector(pos_concept, neg_concept, model_name, target_layer_final)
                else:
                    # Pass the Sidebar Path so it knows where to load from
                    vector = get_sae_feature_vector(feature_idx, model_name, target_layer_final,
                                                    sae_path=sae_path_global)

                # 2. Run Evaluation (The new function)
                results = run_steering_eval(steer_input, vector, multiplier, model_name, target_layer_final)

                # 3. Display Text Side-by-Side
                st.markdown("### 📝 Qualitative Results (Text)")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Control (No Steering)")
                    st.info(results["control_text"])
                with c2:
                    st.caption(f"Treatment (Strength {multiplier})")
                    st.warning(results["steered_text"])

                # 4. Display Metrics (The Scorecard)
                st.markdown("### 📊 Quantitative Results (Metrics)")
                metrics = results["metrics"]

                m1, m2, m3 = st.columns(3)

                # Metric 1: Sentiment Shift
                with m1:
                    st.metric(
                        "Sentiment Shift",
                        f"{metrics['sentiment_steered']:.2f}",
                        delta=f"{metrics['sentiment_delta']:.2f}"
                    )

                # Metric 2: Subjectivity Shift
                with m2:
                    st.metric(
                        "Subjectivity",
                        f"{metrics['subjectivity_steered']:.2f}",
                        delta=f"{metrics['subjectivity_steered'] - metrics['subjectivity_control']:.2f}",
                        delta_color="off"
                    )

                # Metric 3: Word Count Change
                with m3:
                    len_diff = len(results['steered_text']) - len(results['control_text'])
                    st.metric("Length Change", f"{len(results['steered_text'])} chars", delta=f"{len_diff}")

            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 4: FEATURE DICTIONARY ---
# --- TAB 4: FEATURE DICTIONARY ---
with tab4:
    st.header("Sparse Autoencoder Features")
    
    # 1. Load the SAE safely
    try:
        # Try to load, but prepare for it to fail if file is missing
        sae = load_sae(model_name, int(sae_layer), path=sae_path_global) 
    except Exception:
        sae = None

    # 2. THE FIX: Check if it exists before using it
    if sae is None:
        # If missing, show a warning instead of crashing
        st.warning("⚠️ SAE Model not found.")
        st.info(f"Could not find weights at: `{sae_path_global}`. Please upload the .pt file to src/glassbox/.")
    
    else:
        # 3. Only run this code if 'sae' is real
        text_input = st.text_input("Enter text:", "The Eiffel Tower is in Paris")
        
        if st.button("Analyze"):
            # This line was crashing before because sae was None!
            top_feats = get_top_features_for_text(text_input, model, int(sae_layer), sae)
            st.write(top_feats)

    st.markdown("---")
    st.markdown("#### 🔍 Inspect Corpus")

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        feat_id_inspect = st.number_input("Feature ID", 0, value=0, key="feat_inspect")
    with c_col2:
        corpus_path_ui = st.text_input("Corpus Path", "data/sae_corpus.txt")

    if st.button("Find Top Sentences"):
        try:
            # We assume sae is loaded or load it here
            sae = load_sae(model_name, int(sae_layer), path=sae_path_global)

            # Simple check if file exists
            if not os.path.exists(corpus_path_ui):
                st.error("Corpus file not found! Create one in data/sae_corpus.txt")
            else:
                with open(corpus_path_ui, "r") as f:
                    lines = [l.strip() for l in f if l.strip()]

                rows = []
                # Quick scan of first 50 lines for demo
                scan_limit = 50
                for i, txt in enumerate(lines):
                    if i >= scan_limit: break
                    vec = get_feature_activations_for_text(txt, model_name, int(sae_layer), sae)
                    if feat_id_inspect < vec.shape[0]:
                        val = vec[feat_id_inspect].item()
                        if val > 0.1:  # Only show non-zero
                            rows.append({"text": txt, "activation": val})

                if rows:
                    rows.sort(key=lambda x: x["activation"], reverse=True)
                    st.table(rows[:10])
                else:
                    st.info("No activation found in the first 50 lines of corpus.")

        except Exception as e:
            st.error(f"Error: {e}")
