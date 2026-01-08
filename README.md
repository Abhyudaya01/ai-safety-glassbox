# Glass Box AI – LLM Interpretability & Activation Steering Dashboard

This project is a **hands-on interpretability lab for GPT-2**.  
It lets you inspect what a transformer is doing inside its layers, steer its internal activations, and discover **interpretable features** using a **Sparse Autoencoder (SAE)**.

Built with:

- **Python, PyTorch**
- **[TransformerLens](https://github.com/neelnanda-io/TransformerLens)** for hookable GPT-2
- **Streamlit** for an interactive dashboard
- **Sparse Autoencoder (SAE)** for concept discovery

> Deployed as a public Streamlit app so interviewers can explore **attention, logit lens, activation steering, and learned concepts** directly in the browser.

---

## 🔍 High-level Overview

The project is organized into “phases” that map directly to the code:

1. **Phase 1 – Interpretability Engine (`glassbox/`)**
   - Load GPT-2 with hooks.
   - Inspect attention patterns and layer-wise token predictions (logit lens).

2. **Phase 2 – Activation Steering (`glassbox/steering.py`, dashboard tab)**
   - Compute **steering vectors** from text pairs (e.g., `"Love"` – `"Hate"`).
   - Inject these vectors into a chosen residual stream layer during generation.
   - See how the model’s **tone** and behavior change while factual answers stay intact.

3. **Phase 3 – Feature Dictionary with Sparse Autoencoder (`glassbox/sae.py`)**
   - Train a **Sparse Autoencoder (SAE)** on mid-layer GPT-2 activations.
   - Turn raw 768-dim activations into **512 sparse features**.
   - Explore which features fire for a given text, and which texts activate a given feature.
   - Effectively treat the layer as a **dictionary of concepts** instead of a black box.

Planned later phases (e.g., MCP server, embedded deployment) are scaffolded but optional.

---

## 🧠 Dashboard Features

The main UI lives in `src/dashboard/app.py` and is served via Streamlit.

### 1. 🔥 Attention Maps

**Tab:** “Attention Maps”  
**Code:** `glassbox/tracers.py`, `glassbox/visualizers.py`

- Enter any text and visualize **which tokens attend to which** via CircuitsVis.
- Helps answer questions like:
  - “Which word is the model focusing on when predicting the next token?”
  - “Are there heads that specialize in syntax vs. long-range dependencies?”

---

### 2. 🧐 Logit Lens

**Tab:** “Logit Lens”  
**Code:** `glassbox/tracers.py`

- Enter a prefix like:  
  > `"The capital of France is"`
- For each layer, decode the residual stream and show:
  - **Top predicted token**
  - **Probability**
- You can watch “Paris” emerge as the top candidate layer-by-layer.
- Useful for seeing **where factual knowledge appears** in the network.

---

### 3. 🎮 Activation Steering (Concept Injection)

**Tab:** “Activation Steering”  
**Code:** `glassbox/steering.py`

- Define a **concept pair**, e.g.:
  - Positive (+): `"Love"`
  - Negative (−): `"Hate"`
- Compute a steering vector:  
  > `v = activations("Love") – activations("Hate")`
- During generation, inject `multiplier * v` into a chosen layer’s residual stream.
- Run A/B experiments:
  - **Control:** baseline GPT-2
  - **Intervention:** GPT-2 + steering vector
- Example use:
  - For the prompt `"I hate you because"`, a strong positive `"Love"` steering can push the model to **refuse toxic completions** and shift toward more positive / safe responses.
- Shows how we can **change tone and sentiment** without changing underlying facts (e.g., “the capital of France is Paris”).

---

### 4. 🧬 Phase 3 – Sparse Autoencoder Feature Dictionary

**Tab:** “Feature Dictionary”  
**Code:** `glassbox/sae.py`, `scripts/train_sae.py`, `scripts/explore_feature.py`

This is the **“dictionary of concepts”** phase.

#### Training the SAE

1. Collect activations from GPT-2 at **layer 6**:
   - Hook: `blocks.6.hook_resid_post`
   - Corpus: `data/sae_corpus.txt` (one sentence per line)
2. Train a Sparse Autoencoder:
   - Encoder: **768 → 512** features
   - Decoder: **512 → 768**
   - Loss = reconstruction (MSE) + **L1 sparsity** on the feature activations.
3. L1 sparsity encourages **most features to be OFF** for a given token:
   - Only a small subset of features fire strongly.
   - That makes those features **more interpretable**.

Command to train:

```bash
python -m scripts.train_sae --model gpt2 --layer 6 --corpus_path data/sae_corpus.txt --hidden 512 --epochs 10 --l1 1e-2
