import sys
import os
import torch
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components  # <--- FIXED: Required for HTML components
import gc  # Garbage collection for RAM management

# --- PATH SETUP ---
# Ensure we can import from src/glassbox
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- IMPORTS ---
# We wrap these in try/except to prevent the app from dying if a file is moved
try:
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
except ImportError as e:
    st.error(f"Setup Error: {e}. Please check your 'glassbox' folder.")
    st.stop()

# 1. Page Config
st.set_page_config(page_title="Glass Box AI", layout="wide", page_icon="🧠")

# 2. Header
header()

# --- SIDEBAR: GLOBAL CONFIGURATION ---
st.sidebar.header("⚙️ Model Settings")

# A. Model Selector
# NOTE: Streamlit Cloud Free Tier only supports 'gpt2' (Small). 
# Medium/Large will crash due to memory (OOM). 
model_options = {
    "gpt2": 12,          # Small (124M) - SAFE
    "gpt2-medium": 24,   # Medium (355M) - RISK OF CRASH
    "gpt2-large": 36,    # Large (774M) - WILL CRASH ON FREE TIER
}

# Default to index=0 ("gpt2") to prevent startup crashes
model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()), index=0)

if model_name != "gpt2":
    st.sidebar.warning("⚠️ 'Medium'/'Large' models may crash on Streamlit Cloud (Free Tier).")

# B. Dynamic Layer Slider
max_layers = model_options[model_name]
sae_layer = st.sidebar.number_input(
    "SAE Layer",
    min_value=0,
    max_value=max_layers - 1,
    value=int(max_layers / 2),
    help=f"Select a layer between 0 and {max_layers - 1}"
)

# C. Path Logic
default_ckpt = f"src/glassbox/sae_{model_name}_layer{sae_layer}.pt" # Fixed path to src/glassbox
st.sidebar.markdown("---")
st.sidebar.header("🧬 SAE Config")
sae_path_global = st.sidebar.text_input("SAE Path", value=default_ckpt)

# D. Model Loading with RAM Management
if "model" not in st.session_state or st.session_state.get("model_name") != model_name:
    with st.spinner(f"🧠 Loading {model_name}... (Please wait)"):
        # 1. Clear RAM
        if "model" in st.session_state:
            del st.session_state.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        # 2. Load New Model
        try:
            # Reverted to standard constructor to be safe
            st.session_state.model = ModelWrapper(model_name)
            st.session_state.model_name = model_name
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

# Local reference
model = st.session_state.model

# 3. Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔥 Attention Maps", "🧐 Logit Lens", "🎮 Activation Steering", "🧬 Feature Dictionary"]
)

# --- TAB 1: ATTENTION ---
with tab1:
    st.subheader("Visualizing Attention Heads")
    text_input = st.text_input("Enter text to analyze:", "The cat sat on the mat.", key="att_input")
    if st.button("Run Attention", key="btn_att"):
        with st.spinner("Tracing Attention..."):
            try:
                # Assuming get_attention_data returns HTML string or object
                html_viz = get_attention_data(model, text_input) # Passed 'model' object, not name
                components.html(str(html_viz), height=600, scrolling=True)
            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 2: LOGIT LENS ---
with tab2:
    st.subheader("Layer-by-Layer Mind Reader")
    ll_input = st.text_input("Enter text to decode:", "The Eiffel Tower is located in", key="ll_input")
    if st.button("Run Logit Lens", key="btn_ll"):
        with st.spinner("Decoding..."):
            try:
                df = get_logit_lens_data(model, ll_input) # Passed 'model' object
                st.dataframe(df.style.background_gradient(axis=0), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# --- TAB 3: STEERING ---
with tab3:
    st.subheader("🎮 Surgical Activation Steering")
    
    steering_mode = st.radio(
        "Steering Source:",
        ["Manual Concept (Text)", "Discovered Feature (SAE)"],
        horizontal=True
    )

    vector = None
    target_layer_final = sae_layer 

    # MODE A: MANUAL
    if steering_mode == "Manual Concept (Text)":
        col1, col2 = st.columns(2)
        with col1:
            pos_concept = st.text_input("Positive Concept (+)", "Love")
            neg_concept = st.text_input("Negative Concept (-)", "Hate")
        with col2:
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
            feature_idx = st.number_input("SAE Feature Index", min_value=0, value=0)
        with col2:
            st.info(f"Steering on Layer {sae_layer}")
            multiplier = st.slider("Strength", -100.0, 100.0, 0.0, 1.0, key="steer_mult_sae")
            target_layer_final = int(sae_layer)

    st.markdown("---")
    steer_input = st.text_input("Prompt to Steer:", "I am going to the store to buy")

    if st.button("🔴 Run Experiment"):
        with st.spinner("Running A/B Test..."):
            try:
                # 1. Get Vector
                if steering_mode == "Manual Concept (Text)":
                    vector = get_steering_vector(model, pos_concept, neg_concept, target_layer_final)
                else:
                    # SAE Loading Safety Check
                    try:
                        vector = get_sae_feature_vector(feature_idx, model_name, target_layer_final, sae_path=sae_path_global)
                    except Exception as e:
                        st.error(f"Could not load SAE vector: {e}")
                        vector = None

                # 2. Run Evaluation
                if vector is not None:
                    results = run_steering_eval(steer_input, vector, multiplier, model, target_layer_final)
                    
                    st.markdown("### 📝 Results")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Control**")
                        st.info(results["control_text"])
                    with c2:
                        st.markdown(f"**Steered (Strength {multiplier})**")
                        st.warning(results["steered_text"])
            except Exception as e:
                st.error(f"Steering Error: {e}")

# --- TAB 4: FEATURE DICTIONARY ---
with tab4:
    st.subheader("🧬 Feature Dictionary")
    st.markdown(f"Inspecting SAE at **Layer {sae_layer}**")

    # 1. Safety Loading
    try:
        # Pass path explicitly so it doesn't fail if file missing
        sae = load_sae(model_name, int(sae_layer), path=sae_path_global)
    except Exception:
        sae = None

    if sae is None:
        st.warning(f"⚠️ SAE weights not found at `{sae_path_global}`.")
        st.info("Upload the .pt file to `src/glassbox/` or train a new SAE.")
    else:
        dict_input = st.text_input("Text to analyze:", "I went to Paris and London", key="dict_input")
        top_k = st.slider("Top-K features", 5, 50, 10)

        if st.button("Analyze Features"):
            with st.spinner("Analyzing..."):
                try:
                    top_feats = get_top_features_for_text(dict_input, model, int(sae_layer), sae, top_k=top_k)
                    if top_feats:
                        df_feats = pd.DataFrame(top_feats)
                        st.dataframe(df_feats.style.background_gradient(subset=["activation"], cmap="Blues"))
                    else:
                        st.warning("No active features found.")
                except Exception as e:
                    st.error(f"Analysis Error: {e}")
