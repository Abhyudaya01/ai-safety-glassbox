import streamlit as st


def header():
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>🧠 The Glass-Box AI Monitor</h1>
            <p style='font-size: 1.1rem;'>
                Real-time Mechanistic Interpretability. > Type a sentence below to visualize the internal attention patterns of the LLM. 
                This tool exposes <i>why</i> the model generates specific outputs.
            </p>
        </div>
    """, unsafe_allow_html=True)


def sidebar_config():
    st.sidebar.header("⚙️ Model Controls")

    # --- CORRECTED MODEL NAMES (Removed the extra dash) ---
    model_options = [
        "gpt2",  # 117M params (Small)
        "gpt2-medium",  # 345M params (Medium)
        "gpt2-large",  # 774M params (Large)
    ]

    selected_model = st.sidebar.selectbox(
        "Select Model",
        model_options,
        index=0
    )

    st.sidebar.markdown("---")

    # Dynamic Layer Slider based on model choice
    if "medium" in selected_model:
        n_layers = 24
    elif "large" in selected_model:
        n_layers = 36
    else:  # gpt2 (small)
        n_layers = 12

    layer_idx = st.sidebar.slider(
        "Select Network Layer",
        min_value=0,
        max_value=n_layers - 1,
        value=5,
        help="Layer 0 is close to the input (grammar). Higher layers are closer to the output (abstract facts)."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **How to read this:**

        * **Attention:** The lines show which words the model is 'looking at'.
        * **Logit Lens:** The table shows the model's top guess at every layer.
        * **Steering:** Control the model's output with math.
        """
    )

    return selected_model, layer_idx