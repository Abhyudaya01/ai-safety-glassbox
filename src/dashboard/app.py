import sys
import os
from dotenv import load_dotenv  # <--- ADD THIS

load_dotenv()
import torch
import pandas as pd
import streamlit as st
import gc  # Garbage collection for RAM management

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dashboard.components import header
from glassbox.model_loader import ModelWrapper

# --- TRACERS ---
from glassbox.tracers import get_attention_data, get_logit_lens_data
try:
    from glassbox.steering import get_steering_vector, run_steering_eval
except ImportError:
    from glassbox.steering import get_steering_vector
    from glassbox.evaluate import run_steering_eval

from glassbox.sae import (
    load_sae,
    get_top_features_for_text,
    get_feature_activations_for_text,
    get_sae_feature_vector
)


# --- HELPER FUNCTION: ROBUST FEATURE SEARCH (Algorithm v7) ---
def find_best_feature(text, model, sae, layer_name, top_k=5):
    """
    Algorithm v7: Ratio Filtering (Target / Neutral).
    """
    neutral_text = "The wall is made of concrete and stone."

    with torch.no_grad():
        _, cache_target = model.run_with_cache(text)
        _, cache_neutral = model.run_with_cache(neutral_text)

    hook_point = f"blocks.{layer_name}.hook_resid_pre"
    if hook_point not in cache_target:
        return []

    # Get SAE Activations
    act_target = sae(cache_target[hook_point])
    if isinstance(act_target, tuple): act_target = act_target[0]
    act_target = act_target.squeeze(0).max(dim=0).values

    act_neutral = sae(cache_neutral[hook_point])
    if isinstance(act_neutral, tuple): act_neutral = act_neutral[0]
    act_neutral = act_neutral.squeeze(0).max(dim=0).values

    # Ratio Filter
    specificity_ratio = act_target / (act_neutral + 0.1)
    valid_mask = (specificity_ratio > 3.0) & (act_target > 1.0)
    final_scores = torch.where(valid_mask, act_target, torch.tensor(-1.0, device=act_target.device))

    top_vals, top_indices = torch.topk(final_scores, k=min(top_k * 5, final_scores.shape[0]))

    results = []
    found_count = 0

    for i in range(len(top_indices)):
        if found_count >= top_k: break
        idx = top_indices[i].item()
        score = top_vals[i].item()

        if score < 0: continue

        noise_level = act_neutral[idx].item()
        results.append({
            "Rank": found_count + 1,
            "Feature ID": idx,
            "Activation": score,
            "Noise Level": noise_level,
            "Ratio": score / (noise_level + 0.01)
        })
        found_count += 1

    return results


# ------------------------------------------
# MAIN UI SETUP
# ------------------------------------------

st.set_page_config(page_title="Glass Box AI", layout="wide", page_icon="🧠")
header()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Model Settings")
model_options = {"gpt2": 12, "gpt2-medium": 24, "gpt2-large": 36}
model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()), index=0)

max_layers = model_options[model_name]
sae_layer = st.sidebar.number_input("SAE Layer", 0, max_layers - 1, int(max_layers / 2))

default_ckpt = f"data/sae_{model_name}_layer{sae_layer}.pt"
st.sidebar.markdown("---")
sae_path_global = st.sidebar.text_input("SAE Path", value=default_ckpt)

if "model" not in st.session_state or st.session_state.get("model_name") != model_name:
    with st.spinner(f"🧠 Loading {model_name}..."):
        try:
            if "model" in st.session_state:
                del st.session_state.model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            st.session_state.model = ModelWrapper.load(model_name)
            st.session_state.model_name = model_name
        except Exception as e:
            st.error(f"Could not load {model_name}: {e}")
            st.info("Make sure you are connected to the internet for the first run, or that the model is already cached locally by Hugging Face.")
            st.stop()

model = st.session_state.model

# --- TABS ---
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
            st.dataframe(df.style.background_gradient(subset=["Top Prob"], cmap="Greens"), height=600)
        except Exception as e:
            st.error(f"Error: {e}")

# --- TAB 3: STEERING ---
with tab3:
    st.subheader("🎮 Surgical Activation Steering")

    col_setup, col_params = st.columns([1, 1])

    with col_setup:
        steering_mode = st.radio("Source", ["Manual (Text)", "SAE Feature"], horizontal=True)
        if steering_mode == "Manual (Text)":
            pos = st.text_input("Positive Concept", "Love")
            neg = st.text_input("Negative Concept", "Hate")
            steer_layer = st.number_input("Layer", 0, model.cfg.n_layers - 1, int(sae_layer))
            target_layer_final = int(steer_layer)
        else:
            default_feat = st.session_state.get('feature_idx', 0)
            feat_idx = st.number_input("Feature ID", 0, value=default_feat, key="steer_feat_id")
            st.caption(f"Steering on Layer {sae_layer}")
            target_layer_final = int(sae_layer)

    with col_params:
        mult = st.slider("Strength", -10.0, 10.0, 5.0, key="steer_mult")

    steer_prompt = st.text_input("Prompt:", "I am going to the store to buy")

    if st.button("🔴 Run Experiment"):
        with st.spinner("Steering..."):
            try:
                if steering_mode == "Manual (Text)":
                    vec = get_steering_vector(pos, neg, model_name, target_layer_final)
                else:
                    vec = get_sae_feature_vector(feat_idx, model_name, target_layer_final, sae_path=sae_path_global)

                if vec is None:
                    st.error("Could not load the requested SAE feature vector. Check the SAE path, layer, model, and feature ID.")
                    st.stop()

                res = run_steering_eval(steer_prompt, vec, mult, model_name, target_layer_final)

                c1, c2 = st.columns(2)
                c1.info(f"**Original:**\n\n{res['control_text']}")
                c2.warning(f"**Steered:**\n\n{res['steered_text']}")
            except Exception as e:
                st.error(f"Steering Error: {e}")

# --- TAB 4: FEATURE DICTIONARY ---
with tab4:
    st.subheader("🧬 Feature Dictionary & Search")
    st.markdown(f"Inspecting SAE at **Layer {sae_layer}**")

    # --- SECTION A: AUTO-SEARCH ---
    st.info("🤖 **Auto-Search:** Find the neuron that spikes for your concept.")
    col_search_1, col_search_2 = st.columns([3, 1])

    with col_search_1:
        search_query = st.text_input("Concept to find:", "She is a woman and a girl.")
    with col_search_2:
        auto_top_k = st.slider("Candidates", 1, 20, 5)

    if st.button("Auto-Detect Feature", use_container_width=True):
        with st.spinner("Scanning neurons..."):
            try:
                sae_search = load_sae(model_name, int(sae_layer), path=sae_path_global)
                if sae_search:
                    results = find_best_feature(search_query, model, sae_search, sae_layer, top_k=auto_top_k)
                    if results:
                        winner = results[0]
                        st.session_state['feature_idx'] = winner["Feature ID"]
                        st.session_state['feat_inspect'] = winner["Feature ID"]
                        st.success(f"Best Match: Feature #{winner['Feature ID']}")
                        st.dataframe(pd.DataFrame(results)[["Rank", "Feature ID", "Activation", "Ratio"]],
                                     use_container_width=True)
                    else:
                        st.warning("No specific features found.")
                else:
                    st.error("SAE not found.")
            except Exception as e:
                st.error(f"Search Error: {e}")

    st.markdown("---")

    # --- SECTION B: MANUAL SEARCH (FIXED) ---
    st.info("🔬 **Manual Analysis:** See all features active in this text.")
    c_man1, c_man2 = st.columns([3, 1])
    with c_man1:
        dict_input = st.text_input("Text to analyze:", "The doctor called the nurse.")
    with c_man2:
        top_k = st.slider("Top-K", 5, 50, 10)

    if st.button("Analyze Features"):
        sae = load_sae(model_name, int(sae_layer), path=sae_path_global)
        if sae:
            top_feats = get_top_features_for_text(dict_input, model_name, int(sae_layer), sae, top_k=top_k)
            if top_feats:
                # FIX: Do not hardcode column names (avoids KeyError)
                df = pd.DataFrame(top_feats)
                st.dataframe(df.style.background_gradient(cmap="Blues"), use_container_width=True)
            else:
                st.warning("No active features found.")

    st.markdown("---")

    # --- SECTION C: INSPECTOR & CORPUS SCANNER (FIXED) ---
    st.subheader("🔍 Feature Inspector")

    col_insp_1, col_insp_2 = st.columns([1, 2])

    with col_insp_1:
        # Input ID manually or take from Auto-Detect
        f_id = st.number_input("Feature ID", min_value=0, value=st.session_state.get('feature_idx', 0),
                               key="feat_inspect")

        # GEMINI LABELER
        # Check if key is already loaded from .env
        env_key = os.getenv("GEMINI_API_KEY")

        # If key exists in .env, pre-fill the box (masked) or hide it
        if env_key:
            st.success("✅ API Key loaded from .env")
        else:
            # Only show input if .env is missing
            api_key_input = st.text_input("Gemini API Key", type="password")
            if api_key_input:
                os.environ["GEMINI_API_KEY"] = api_key_input
        if st.button("✨ Auto-Label"):
            sae = load_sae(model_name, int(sae_layer), path=sae_path_global)
            if sae and os.getenv("GEMINI_API_KEY"):
                from glassbox.auto_interp import generate_feature_label

                label = generate_feature_label(f_id, sae, model, int(sae_layer), None)
                st.success("Done!")
                st.markdown(f"### {label}")
            else:
                st.error("Missing SAE or API Key.")

    with col_insp_2:
        # FIXED CORPUS SCANNER
        st.markdown("#### 📖 Top Sentences from Corpus")

        # 1. Create Dummy Corpus if missing
        default_corpus = "data/sae_corpus.txt"
        if not os.path.exists("data"): os.makedirs("data")

        if not os.path.exists(default_corpus):
            with open(default_corpus, "w") as f:
                # Add diverse sentences so the scanner always finds something
                f.write("The cat sat on the mat.\n")
                f.write("She is a doctor and a mother.\n")
                f.write("He is a construction worker.\n")
                f.write("Python is a programming language.\n")
                f.write("The Eiffel Tower is in Paris.\n")
                f.write("I love you so much.\n")
                f.write("I hate this situation.\n")
                f.write("The integral of x squared is x cubed over three.\n")

        corpus_path_ui = st.text_input("Corpus Path", default_corpus)
        scan_limit = st.slider("Scan Limit (Lines)", 50, 1000, 200)

        if st.button("Find Top Sentences"):
            sae = load_sae(model_name, int(sae_layer), path=sae_path_global)
            if sae and os.path.exists(corpus_path_ui):
                with st.spinner(f"Scanning {scan_limit} lines..."):
                    with open(corpus_path_ui, "r") as f:
                        lines = [l.strip() for l in f if l.strip()]

                    rows = []
                    found_any = False
                    for i, txt in enumerate(lines):
                        if i >= scan_limit: break
                        vec = get_feature_activations_for_text(txt, model_name, int(sae_layer), sae)
                        if f_id < vec.shape[0]:
                            val = vec[f_id].item()
                            # FIX: LOWER THRESHOLD to 0.0001 (catches rare features)
                            if val > 0.0001:
                                rows.append({"Sentence": txt, "Activation": val})
                                found_any = True

                    if found_any:
                        rows.sort(key=lambda x: x["Activation"], reverse=True)
                        st.table(pd.DataFrame(rows[:10]))
                    else:
                        st.warning(
                            "No activations found. Try increasing Scan Limit or analyzing a feature found in Manual Search first.")
