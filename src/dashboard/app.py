import sys
import os

# --- PATH SETUP (Must be first) ---
# This points Python to the 'src' folder so it can find 'glassbox' and 'dashboard'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # This gets us to 'src'
sys.path.append(parent_dir)
# ----------------------------------

# Now your imports will work!
import streamlit as st
import pandas as pd
from dashboard.components import header, sidebar_config
from glassbox.model_loader import ModelWrapper
from glassbox.tracers import get_attention_data, get_logit_lens_data
from glassbox.steering import get_steering_vector, generate_steered_response

# 1. Page Config
st.set_page_config(page_title="Glass Box AI", layout="wide")

# 2. Sidebar & Header
header()
model_name, layer_idx = sidebar_config()

# Load Model (Singleton)
with st.spinner(f"🧠 Loading {model_name}... (this may take a moment)"):
    model = ModelWrapper.load(model_name)

# 3. Main Tabs
tab1, tab2, tab3 = st.tabs(["🔥 Attention Maps", "🧐 Logit Lens", "🎮 Activation Steering"])

# --- TAB 1: ATTENTION ---
with tab1:
    st.subheader("Visualizing Attention Heads")
    text_input = st.text_input("Enter text to analyze:", "The cat sat on the mat.", key="att_input")
    if text_input:
        try:
            html_viz = get_attention_data(text_input, model_name)
            st.components.v1.html(html_viz, height=600, scrolling=True)
        except Exception as e:
            st.error(f"Error visualizing attention: {e}")

# --- TAB 2: LOGIT LENS ---
with tab2:
    st.subheader("Layer-by-Layer Mind Reader")
    ll_input = st.text_input("Enter text to decode:", "The Eiffel Tower is located in", key="ll_input")
    if ll_input:
        df = get_logit_lens_data(ll_input, model_name)
        st.dataframe(df.style.background_gradient(subset=["Top Prob"], cmap="Greens"), height=600)

# --- TAB 3: STEERING (NEW!) ---
with tab3:
    st.subheader("🧠 Activation Steering (Mind Control)")
    st.markdown("Define a concept vector and force the model to think about it.")

    col1, col2 = st.columns(2)
    with col1:
        pos_concept = st.text_input("Positive Concept (+)", "Love")
        neg_concept = st.text_input("Negative Concept (-)", "Hate")

    with col2:
        target_layer = st.number_input("Injection Layer", min_value=0, max_value=model.cfg.n_layers - 1, value=6)
        multiplier = st.slider("Steering Strength", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)

    steer_input = st.text_input("Prompt to Steer:", "I think that you are")

    if st.button("🔴 Run Experiment"):
        with st.spinner("Calculating Vector & Generating..."):
            # 1. Get the Vector
            vector = get_steering_vector(pos_concept, neg_concept, model_name, target_layer)

            # 2. Generate NORMAL response (Control)
            control_out = generate_steered_response(steer_input, vector, 0.0, model_name, target_layer)

            # 3. Generate STEERED response (Intervention)
            steered_out = generate_steered_response(steer_input, vector, multiplier, model_name, target_layer)

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