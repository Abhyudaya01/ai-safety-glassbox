# 🧠 Glass Box AI: LLM Interpretability & Steering

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![TransformerLens](https://img.shields.io/badge/Library-TransformerLens-orange.svg)](https://github.com/neelnanda-io/TransformerLens)
[![Status](https://img.shields.io/badge/Status-Live_Deployment-green.svg)]()

> **"Don't just read the output. Read the mind."**

## 📋 Overview
**Glass Box AI** is a full-stack interpretability lab for GPT-2. It allows researchers to visualize internal model states, discover interpretable features using **Sparse Autoencoders (SAE)**, and perform **Activation Steering** to manipulate model behavior in real-time.

Unlike standard LLM interfaces, this tool hooks into the residual stream to expose *how* the model thinks, not just what it says.

**🔗 [View Live Dashboard](https://ai-safety-glassbox-ffpfd7caqoxnrnalcyddnb.streamlit.app/)**

---

## 🚀 Key Features (The "Phases")

### 1. 🧐 Logit Lens (Mind Reading)
* **The Logic:** Decodes the residual stream at every layer to see the model's "subconscious" predictions.
* **The Insight:** Watch concepts like "Paris" emerge in Layer 15, well before the final output generation.
* **Tech:** `glassbox/tracers.py`

### 2. 🎮 Activation Steering (Behavioral Control)
* **The Logic:** We inject a "steering vector" (calculated from concept pairs like `Love - Hate`) directly into the residual stream during inference.
* **The Result:** We can force the model to be "happier," "angrier," or "more factual" without retraining a single weight.
* **Tech:** `glassbox/steering.py`

### 3. 🧬 Dictionary Learning (Sparse Autoencoders)
* **The Logic:** Neural networks process information in "superposition" (dense, polysemantic vectors).
* **The Solution:** Trained a **Sparse Autoencoder (SAE)** on Layer 6 activations to decompose these dense vectors into **512 human-interpretable features**.
* **Tech:** `glassbox/sae.py`

---

## 🛠️ Tech Stack

* **Core Model:** `GPT-2 Small/Medium`
* **Interpretability Lib:** `TransformerLens` (Hook points & caching)
* **Frontend:** `Streamlit` (Real-time visualization)
* **Math:** `PyTorch`, `Einops` (Tensor manipulation)
* **Visualization:** `CircuitsVis`, `Plotly`

---

## 🧪 Featured Experiment: "The Rome Inception"

One of the core validation tests for this dashboard was **Concept Replacement**.

* **Goal:** Force GPT-2 (which knows the Eiffel Tower is in Paris) to believe it is in **Rome**.
* **Method:** We intercept the forward pass at **Layer 10** and inject the vector `(Rome - Paris)`.
* **Result:**
    > *Control:* "The Eiffel Tower is located in **Paris**."
    >
    > *Steered:* "The Eiffel Tower is located in **Rome**, in the city of Rome..."

---

## 💻 Installation & Usage

### 1. Setup
```bash
git clone https://github.com/Abhyudaya01/ai-safety-glassbox.git
cd ai-safety-glassbox
pip install -r requirements.txt

-->Run the Dashboard

python -m streamlit run src/dashboard/app.py

-->Train the Sparse Autoencoder (SAE)

python -m scripts.train_sae \
  --model gpt2 \
  --layer 6 \
  --corpus_path data/sae_corpus.txt \
  --hidden 512 \
  --epochs 10 \
  --l1 1e-2

-->Repo Structure

glass-box-ai/
├── glassbox/               # Core Interpretability Engine
│   ├── tracers.py          # Logit Lens & Attention Hooks
│   ├── steering.py         # Vector Arithmetic Logic
│   └── sae.py              # Sparse Autoencoder Architecture
├── src/
│   └── dashboard/          # Streamlit UI Components
│       └── app.py          # Main Application Entry
├── scripts/                # Training Scripts
│   └── train_sae.py        # SAE Training Loop
├── data/                   # Activations & Model Weights
└── requirements.txt        # Dependencies
