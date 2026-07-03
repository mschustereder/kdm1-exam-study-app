# 07 — Dimensionality Reduction (Medium Summary)

## Overview & Motivation

Dimensionality reduction transforms a high-dimensional feature space into a lower-dimensional **embedding space**. It is motivated by the **curse of dimensionality** (data becomes sparse, distances lose meaning, algorithms degrade) and by the expectation that many features are correlated or redundant. Applications include visualisation, preprocessing, compression, and noise removal.

---

## Regularisation & Feature Transformation (Slides 8–14)

- **Regularisation** penalises model complexity by adding a term to the cost function: `cost(f) = −l(f) + regularizer(f)`. Common regularisers: L0 (count of non-zero features), L1 (sum of absolute feature values, induces sparsity).
- **Feature Transformation** maps features into a higher-dimensional space to make problems linearly separable (e.g., solving XOR with a single-layer perceptron). The **kernel trick** performs this mapping on-the-fly via a kernel function φ(x, y) without explicitly constructing the high-dimensional space — used in SVMs.

---

## Principal Component Analysis (PCA) (Slides 19–26)

PCA is the classic linear dimensionality reduction method:

1. **Centre** the data matrix D by subtracting the mean → matrix X.
2. Compute the **sample covariance matrix**: `S = (1/n) X^T X`. This is square, symmetric, and positive semi-definite; its eigenvalues are real and eigenvectors are orthonormal.
3. The eigenvectors form a new orthonormal basis (principal components). Each axis is ordered by **descending variance** — the first component captures the most variance.
4. Retain only the first k components, discarding low-variance dimensions. This achieves **decorrelation** and compression.

**Key properties & limitations:**
- Assumes linear relationships only.
- Variance implies importance, so normalisation (e.g., z-scoring) strongly affects results.
- Sensitive to outliers.
- Choosing k: use a variance threshold (e.g., 95%), knee-point/elbow detection, or fix k=2–3 for visualisation.
- For Gaussian data, PCA is information-theoretically optimal (maximal information content in top components).

---

## Singular Value Decomposition (SVD) (Slides 27–31)

SVD generalises PCA to **non-square matrices**. Any matrix M ∈ ℝ^(n×m) of rank r can be factored as `M = U Σ V^T`, where:
- U ∈ ℝ^(n×r) — column-orthonormal (left singular vectors)
- V ∈ ℝ^(m×r) — column-orthonormal (right singular vectors)
- Σ ∈ ℝ^(r×r) — diagonal matrix of singular values (sorted descending)

**Interpretation:** SVD uncovers a small number of latent "concepts" linking rows and columns. Retaining only the top k singular values (Thin SVD) yields the best low-rank approximation. A classic application is **Latent Semantic Indexing** for text retrieval.

---

## t-Distributed Stochastic Neighbor Embedding (t-SNE) (Slides 32–34)

t-SNE is a **non-linear** method primarily used for visualisation. Algorithm steps:
1. Compute pairwise probabilities p_ij in the high-dimensional space (Gaussian, perplexity controls the per-point standard deviation).
2. Randomly initialise points in low-dimensional space.
3. Compute probabilities q_ij in low-dimensional space (Student t-distribution).
4. Minimise the Kullback–Leibler divergence between p and q via gradient descent.

**Reflections:** t-SNE preserves **local structure** (neighbourhoods) well but can distort global distances. **UMAP** is a newer alternative that better preserves global structure and scales better.

---

## Autoencoders (Slides 35–36)

Autoencoders use neural networks for non-linear dimensionality reduction. An **encoder** compresses the input into a low-dimensional **bottleneck layer** (the embedding), and a **decoder** reconstructs the original input. Training is **self-supervised** — the objective is reconstruction error.

---

## Conclusions & Learning Objectives

- All methods require **numeric features** and a meaningful **distance function**.
- Preprocessing (normalisation, centering) is critical.
- Key techniques to know: PCA (linear, global variance), SVD (generalises PCA, latent concepts), t-SNE (non-linear, local), and autoencoders (deep learning).

*Source: KDM1 — Dimensionality Reduction, R. Kern & D. Helic, v2.1.1, TU Graz.*
