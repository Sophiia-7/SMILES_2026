# SOLUTION.md

## SMILES-2026 Hallucination Detection Solution

**Author:** Sofiia Buianova
**Task:** Detect hallucinated responses from Qwen2.5-0.5B using internal hidden states

---

## 1. Reproducibility Instructions

### 1.1 Environment Setup

```bash
git clone https://github.com/Sophiia-7/SMILES_2026
cd SMILES_2026

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate.bat  # Windows

pip install -r requirements.txt
```

### 1.2 Run Solution

```bash
python solution.py
```

### 1.3 Expected Output

* `predictions.csv` — predictions for test set
* `results.json` — cross-validation metrics averaged over folds

### 1.4 Implementation Details

| Setting            | Value                            |
| ------------------ | -------------------------------- |
| Model              | Qwen/Qwen2.5-0.5B                |
| CV strategy        | Stratified 5-fold                |
| Random seed        | 42 (used in splitting)           |
| Geometric features | Enabled (`USE_GEOMETRIC = True`) |
| Feature type       | Hidden-state based aggregation   |

### 1.5 Hardware

* Minimum: CPU (slow)
* Recommended: GPU (T4 / similar)

---

## 2. Final Solution Description

### 2.1 Modified Components

| File             | Changes                                                   |
| ---------------- | --------------------------------------------------------- |
| `aggregation.py` | Multi-layer hidden state aggregation + geometric features |
| `probe.py`       | MLP classifier + StandardScaler + PCA + weighted loss     |
| `splitting.py`   | Stratified 5-fold cross-validation                        |

---

### 2.2 Final Pipeline

```
Input: prompt + response (ChatML format)
        ↓
Qwen2.5-0.5B forward pass
        ↓
Hidden states extraction (24 layers)
        ↓
Select layers (-1, -2, -4)
        ↓
Token pooling:
    - mean pooling
    - last-token representation
        ↓
Concatenation → feature vector (~5500 dims)
        ↓
Geometric features (norm stats, layer drift, etc.)
        ↓
StandardScaler normalization
        ↓
MLP classifier:
    128 → 32 → 1
        ↓
Sigmoid → hallucination probability
```

---

## 2.3 Key Design Choices

### Multi-layer representations (-1, -2, -4)

Different layers encode:

* final semantic output (-1)
* intermediate reasoning (-2)
* early refinement signals (-4)

This improves robustness compared to single-layer probing.

---

### Token pooling strategy

We use:

* mean pooling (global representation)
* last token representation (generation endpoint signal)

This captures both:

* overall response consistency
* final decision state of the model

---

### Geometric features

We include statistics of hidden states:

* token norm mean/std/max/min per layer
* layer-to-layer cosine similarity
* representation drift between layers
* sequence length signal

These features capture **uncertainty and instability in generation**, which correlates with hallucination.

---

### PCA (dimensionality reduction)

Applied after scaling to reduce redundancy in ~5k+ dimensional feature space and improve generalization on small dataset (689 samples).

---

### MLP classifier

A small neural probe is used instead of linear models:

```
Linear → ReLU → Dropout
Linear → ReLU → Dropout
Linear → sigmoid
```

This allows non-linear separation between truthful and hallucinated representations.

---

### Class-weighted loss

Due to imbalance (more hallucinated samples), we use:

* `BCEWithLogitsLoss(pos_weight=neg/pos)`

---

## 2.4 What Improved Performance Most

| Component               | Effect                             |
| ----------------------- | ---------------------------------- |
| Multi-layer aggregation | strong improvement in stability    |
| Geometric features      | better AUROC separation            |
| Class weighting         | improved recall of hallucinations  |
| MLP vs linear probe     | better nonlinear decision boundary |
| PCA                     | reduced overfitting                |

---

## 3. Experiments and Failed Attempts

### Full 24-layer concatenation

* Result: extreme overfitting
* Feature space too large for dataset size
* discarded

---

### Single final layer only

* too weak signal
* underperformed multi-layer setup
* discarded

---

### Attention-weighted pooling

* unstable and noisy
* worse than simple mean pooling
* discarded

---

### SVM (linear and RBF)

* linear: underfit
* RBF: slow, no consistent gain
* discarded

---

### Max pooling

* lost distributional information
* worse than mean pooling
* discarded

---

### Response-only encoding

* removed critical context
* strong performance drop
* discarded

---

### Removing class weighting

* caused bias toward majority class
* worse F1 and recall
* kept weighting

---

## 4. Final Remarks

### Strengths

* robust to overfitting via regularization + PCA
* uses multi-layer semantic signal
* interpretable geometric features
* stable across folds

### Limitations

* still sensitive to dataset size (689 samples is small)
* limited ability to generalize to other LLM families

### Possible future improvements

* contrastive learning between truthful/hallucinated pairs
* attention over layers instead of fixed selection
* calibration (temperature scaling) for probabilities

---

## Submission

**Link to predictions.csv:** https://drive.google.com/drive/folders/1aZo9OpF226_mgPx7C5VTy7rtTNiIZE9H?usp=drive_link
