import sys
import os

# --- PATH SETUP (Must be first) ---
# This points Python to the 'src' folder so it can find 'glassbox' and 'dashboard'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # This gets us to 'src'
sys.path.append(parent_dir)
# ----------------------------------

import streamlit as st
import pandas as pd

from dashboard.components import header, sidebar_config
from glassbox.model_loader import ModelWrapper
from glassbox.tracers import get_attention_data, get_logit_lens_data
from glassbox.steering import get_steering_vector, generate_steered_response
from glassbox.sae import (
    load_sae,
    get_top_features_for_text,
    default_sae_path,
    get_feature_activations_for_text,
)

# 1. Page Config
st.set_page_config(page_title="Glass Box AI", layout="wide")

# 2. Sidebar & Header
header()
model_name, layer_idx = sidebar_config()

# Load Model (Singleton)
with st.spinner(f"🧠 Loading {model_name}... (this may take a moment)"):
    model = ModelWrapper.load(model_name)

# 3. Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔥 Attention Maps", "🧐 Logit Lens", "🎮 Activation Steering", "🧬 Feature Dictionary"]
)

# --- TAB 1: ATTENTION ---
with tab1:
    st.subheader("Visualizing Attention Heads")
    text_input = st.text_input(
        "Enter text to analyze:",
        "The cat sat on the mat.",
        key="att_input",
    )
    if text_input:
        try:
            html_viz = get_attention_data(text_input, model_name)
            st.components.v1.html(html_viz, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Error visualizing attention: {e}")

# --- TAB 2: LOGIT LENS ---
with tab2:
    st.subheader("Layer-by-Layer Mind Reader")
    ll_input = st.text_input(
        "Enter text to decode:",
        "The Eiffel Tower is located in",
        key="ll_input",
    )
    if ll_input:
        try:
            df = get_logit_lens_data(ll_input, model_name)
            st.dataframe(
                df.style.background_gradient(subset=["Top Prob"], cmap="Greens"),
                height=600,
            )
        except Exception as e:
            st.error(f"Error running logit lens: {e}")

# --- TAB 3: STEERING ---
with tab3:
    st.subheader("🧠 Activation Steering (Mind Control)")
    st.markdown("Define a concept vector and force the model to think about it.")

    col1, col2 = st.columns(2)
    with col1:
        pos_concept = st.text_input("Positive Concept (+)", "Love")
        neg_concept = st.text_input("Negative Concept (-)", "Hate")

    with col2:
        target_layer = st.number_input(
            "Injection Layer",
            min_value=0,
            max_value=model.cfg.n_layers - 1,
            value=6,
        )
        multiplier = st.slider(
            "Steering Strength",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
        )

    steer_input = st.text_input("Prompt to Steer:", "I think that you are")

    if st.button("🔴 Run Experiment"):
        with st.spinner("Calculating Vector & Generating..."):
            try:
                # 1. Get the Vector
                vector = get_steering_vector(
                    pos_concept, neg_concept, model_name, int(target_layer)
                )

                # 2. Generate NORMAL response (Control)
                control_out = generate_steered_response(
                    steer_input, vector, 0.0, model_name, int(target_layer)
                )

                # 3. Generate STEERED response (Intervention)
                steered_out = generate_steered_response(
                    steer_input, vector, multiplier, model_name, int(target_layer)
                )

                # 4. Display Results
                st.markdown("### 🧪 Results")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.info(f"**Original Output:**\n\n{control_out}")
                with res_col2:
                    if multiplier > 0:
                        st.success(f"**Steered Output (+{multiplier}):**\n\n{steered_out}")
                    else:
                        st.warning(f"**Steered Output ({multiplier}):**\n\n{steered_out}")
            except Exception as e:
                st.error(f"Error during steering experiment: {e}")

# --- TAB 4: FEATURE DICTIONARY (SAE) ---
with tab4:
    st.subheader("🧬 Feature Dictionary (Sparse Autoencoder)")
    st.markdown(
        "Discover which internal **features** (SAE neurons) fire for your text. "
        "This turns the model into a kind of concept dictionary."
    )

    # Config for which layer & checkpoint to use
    dict_layer = st.number_input(
        "Dictionary Layer (SAE trained here)",
        min_value=0,
        max_value=model.cfg.n_layers - 1,
        value=6,
        step=1,
    )

    default_ckpt = default_sae_path(model_name, int(dict_layer))
    sae_path = st.text_input(
        "SAE Checkpoint Path",
        value=default_ckpt,
        help="Path to the trained SAE checkpoint file.",
    )

    dict_input = st.text_input(
        "Text to analyze (features will be extracted from this):",
        "I went to Paris and London and wrote some Python code.",
        key="dict_input",
    )

    top_k = st.slider("Top-K features to show", min_value=5, max_value=50, value=20, step=5)

    if st.button("Analyze Features", key="analyze_features"):
        with st.spinner("Loading SAE and extracting features..."):
            try:
                sae = load_sae(model_name, int(dict_layer), path=sae_path)
                top_feats = get_top_features_for_text(
                    dict_input,
                    model_name,
                    int(dict_layer),
                    sae,
                    top_k=top_k,
                )
                if not top_feats:
                    st.info("No features found (this usually means the SAE is not well-trained yet).")
                else:
                    df_feats = pd.DataFrame(top_feats)
                    st.markdown("**Top feature activations for this text:**")
                    st.dataframe(
                        df_feats.style.background_gradient(
                            subset=["activation"], cmap="Blues"
                        ),
                        height=400,
                    )
            except FileNotFoundError:
                st.error(
                    f"Could not find SAE checkpoint at `{sae_path}`. "
                    "Make sure you've trained it (e.g. `python -m scripts.train_sae`)."
                )
            except Exception as e:
                st.error(f"Error running Feature Dictionary: {e}")

    st.markdown("---")
    st.markdown("#### 🔍 Inspect a Single Feature on the SAE Corpus")

    feat_id = st.number_input(
        "Feature ID to inspect",
        min_value=0,
        value=0,
        step=1,
        help="Pick a feature index (from the table above) to see which corpus sentences activate it most.",
    )
    corpus_path_ui = st.text_input(
        "Corpus file path",
        value="data/sae_corpus.txt",
        help="Corpus used for SAE training (one sentence per line).",
    )
    top_k_ui = st.slider(
        "Top-K sentences to show",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        key="top_k_sentences",
    )

    if st.button("Show top sentences for this feature", key="show_feature_sentences"):
        try:
            sae = load_sae(model_name, int(dict_layer), path=sae_path)
            if not os.path.exists(corpus_path_ui):
                st.error(f"Corpus file not found at `{corpus_path_ui}`.")
            else:
                with open(corpus_path_ui, "r", encoding="utf-8") as f:
                    corpus_prompts = [ln.strip() for ln in f if ln.strip()]

                rows = []
                for text in corpus_prompts:
                    vec = get_feature_activations_for_text(
                        text,
                        model_name,
                        int(dict_layer),
                        sae,
                    )
                    if feat_id < 0 or feat_id >= vec.shape[0]:
                        st.error(
                            f"Feature index {int(feat_id)} is out of range (d_hidden={vec.shape[0]})."
                        )
                        break
                    score = float(vec[int(feat_id)].item())
                    rows.append({"text": text, "activation": score})

                rows.sort(key=lambda r: r["activation"], reverse=True)
                rows = rows[: top_k_ui]

                if rows:
                    st.markdown("**Top sentences for this feature (sorted by activation):**")
                    for i, row in enumerate(rows, start=1):
                        st.write(f"**#{i}** · Activation: `{row['activation']:.4f}`")
                        st.write(row["text"])
                        st.write("---")
                else:
                    st.info("No sentences found in corpus or all activations are zero.")
        except Exception as e:
            st.error(f"Error inspecting feature: {e}")
