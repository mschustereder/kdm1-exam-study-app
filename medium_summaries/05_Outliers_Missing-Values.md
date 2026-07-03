# 05 — Outliers & Missing Values

## 1. Motivation

Both too much data (outliers) and too little data (missing values) can distort analysis. The goal is to understand their generative nature and learn common strategies.

## 2. Recap (Prerequisites)

- **KDE**: Estimates a PDF from discrete observations using a tunable kernel.
- **I.I.D. Assumption**: Independence + identical distribution; common for tabular, less for time series.
- **Law of Large Numbers**: More i.i.d. data improves estimates and predictions if the model fits.

## 3. Outliers (Anomaly Detection)

### 3.1 Core Idea

Even one low-probability point can skew estimates (e.g., the mean of a normal distribution). Outliers are **noise to remove** or the **target to detect** in fraud detection, intrusion detection, and medical diagnosis. Key challenges: highly skewed datasets, missing ground truth (open class), and high-dimensional heterogeneous data.

Application areas include intrusion detection, fraud detection, industrial fault monitoring, infrastructure monitoring, stock market surveillance, novelty detection, and medical diagnosis.

### 3.2 Two Perspectives

- **Data-driven (low-probability)**: Points in low-density regions of a fitted distribution.
- **Generative**: Points from a separate process (e.g., "Martians" vs. humans).

### 3.3 Formal Definition

Normal distribution \(P_+\) with density \(p_+\). Anomaly set: \(A = \{x \in X \mid p_+(x) \le \tau\}\). Classic definitions (Grubbs, Barnett, Hawkins) agree: outliers deviate markedly from the rest.

### 3.4 Anomaly Taxonomy

1. **Point**: Single deviating point.
2. **Contextual**: Deviates only within a context (\(P_+ \equiv P_+(X \mid T)\)).
3. **Collective**: A related set anomalous together.
4. **Low-level sensory**: Pixel/signal-level deviation.
5. **High-level semantic**: Normality conditioned on latent variables.

### 3.5 Terminology

**Anomaly** (distinct distribution) vs. **outlier** (rare instance from \(P_+\)) vs. **novelty** (new mode of evolving \(P_+\)).

### 3.6 Detection Settings

**Unsupervised** (no labels), **semi-supervised** (partial labels), **supervised** (full labels — rare for novelty). KDE and OC-SVM implicitly assume a uniform anomaly distribution.

### 3.7 Key Assumptions

- **Concentration**: Normal data is bounded; anomalies in low-density areas.
- **Manifold**: Normal data lives on a lower-dimensional embedded manifold.
- **Prototype**: Normal data is characterised by finite prototypes.

### 3.8 Approach Types

**Probabilistic** (needs threshold \(\alpha\)), **score-based** (ranking), **binary** (hard assignment), **distance-based**, **domain-informed** (heuristics).

### 3.9 Algorithm Families

- **Density-based (generative)**: Gaussian, KDE, Real NVP — full density estimation.
- **One-class classification (discriminative)**: Boundary around normals (hypersphere, OC-SVM). Deep variants are flexible but data-hungry.
- **Reconstruction**: Encoder + decoder; score = \(\|x - \psi_d(\psi_e(x))\|^2\) (PCA, autoencoders, VAEs).
- **LOF**: k-NN local density; sparse regions get higher anomaly scores.
- **Isolation Forest (iForest)**: Fewer splits needed to isolate anomalies; no normalisation required.
- **Depth-based**: Convex-hull layers; points at shallow depth are outliers.
- **ABOD**: Lower angle variance — more stable than distances in high dimensions.
- **Grid-based (Aggarwal 2001)**: Sparsity coefficient \(S(C) = \frac{n(C) - n \Phi^{-k}}{\sqrt{n \Phi^{-k} (1 - \Phi^{-k})}}\) flags underpopulated cells.

## 4. Missing Values

### 4.1 Implications & Strategies

Loss of efficiency, bias, method incompatibility. Three strategies: (1) native-handling methods (e.g., decision trees), (2) drop rows (safe only under MCAR), (3) imputation.

### 4.2 Imputation

**Univariate**: Constant, mode, mean/median.
**Multivariate**: Predict from other features (e.g., via ML), modelling inter-feature dependencies.

### 4.3 Missingness Mechanisms (Little & Rubin 1987)

- **MCAR**: No dependency on observed or missing data → dropping rows is safe.
- **MAR**: Depends on observed data only → predictive imputation works.
- **NMAR**: Depends on the missing value itself → imputation may introduce bias.

Validity hinges on understanding *why* values are missing, often unverifiable from data alone.

## 5. Learning Objectives

- Define outliers from low-probability and generative-mechanism perspectives.
- Distinguish point, contextual, collective, sensory, and semantic anomalies.
- Compare probabilistic, score-based, distance-based, and domain-informed approaches.
- Explain density-based, one-class, reconstruction, LOF, iForest, ABOD, and grid-based methods.
- Categorise missingness as MCAR/MAR/NMAR and select appropriate strategies.
- Validate that assumptions (I.I.D., concentration, manifold, missingness) hold for sound results.

---

*Source: KDDM1 — Roman Kern, V1.1.0, TU Graz. Key refs: Chandola et al. (2009), Ruff et al. (2021), Little & Rubin (1987).*
