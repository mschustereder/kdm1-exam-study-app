# 07 Dimensionality Reduction — Concise Summary

## Learning Objectives

- Understand the motivation for reducing dimensionality: improve runtime, avoid the curse of dimensionality, remove redundancies, and decorrelate features.
- Grasp the distinctions between feature selection, regularisation, feature transformation (increasing dimensions), and dimensionality reduction (decreasing dimensions).
- Apply PCA, SVD, t-SNE, and Autoencoders for different dimensionality reduction tasks.
- Select the appropriate technique based on data type, linearity assumptions, and downstream goal (visualisation, compression, preprocessing).

## Core Definitions & Concepts

- **Dimensionality Reduction**: Transformation of a high-dimensional feature space into a lower-dimensional embedding space, removing redundancies and correlations.
- **Curse of Dimensionality**: As dimensions increase, data becomes sparse, distance functions lose meaning, and algorithm performance degrades.
- **Feature Engineering Process**: Remove unnecessary/redundant features, create or transform features, modify feature types/values.
- **Regularisation**: Penalising model complexity by integrating a regulariser into the cost function (e.g., `cost(f) = -l(f) + regularizer(f)`). Common regularisers include L0 (number of non-zero features) and L1 (sum of feature values).
- **Feature Transformation (Kernel Trick)**: Mapping features into a higher-dimensional space so that non-linearly separable problems become linearly separable, performed on-the-fly via kernel functions (e.g., Gaussian kernel).
- **Distance Function**: Must match the data; normalisation is often required to equalise feature value ranges.
- **Self-Supervised Learning**: Autoencoders are technically unsupervised but use a supervised training objective (reconstruction).

## Key Algorithms & Techniques

### Principal Component Analysis (PCA)

- Linearly transforms the centered data matrix into a new orthonormal basis (eigenvectors of the covariance matrix).
- Axes (principal components) are ordered by descending variance; the first k components capture the most variance.
- Sample covariance matrix: `S = (1/n) X^T X` — square, symmetric, positive semi-definite.
- Limitations: only linear relationships; sensitive to normalisation, range, and outliers; selecting k requires knee/elbow point detection or a variance threshold.
- Can be interpreted as optimal compression for Gaussian data (information-theoretic optimality).

### Singular Value Decomposition (SVD)

- Generalisation of PCA to non-square matrices: `M = U Σ V^T`, where U and V are column-orthonormal and Σ is diagonal with singular values.
- Rank r of matrix M determines the number of non-zero singular values.
- **Thin SVD**: choosing k < r singular values for approximation.
- Assumes a small number of latent "concepts" connect rows and columns, ranked by importance.
- Application: **Latent Semantic Indexing (LSI)** — reduces high-dimensional text data for information retrieval in embedding space.

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

- Non-linear dimensionality reduction primarily used for visualisation.
- Preserves local neighbourhood structure by matching pairwise probabilities between high-dimensional (Gaussian) and low-dimensional (Student t-distribution) spaces.
- Algorithm steps: (1) compute high-dim probabilities p_ij with perplexity, (2) initialise low-dim space randomly, (3) compute low-dim probabilities q_ij, (4) optimise via gradient descent on KL divergence between p and q.
- Reflections: preserves local structure well; PCA captures global structure; Euclidean distance in high-dim space suffers from curse of dimensionality.
- **UMAP** (Uniform Manifold Approximation and Projection) is an extension that better preserves global structures and is often preferred as a preprocessing step before clustering.

### Autoencoder

- Deep learning approach: encoder compresses input to a bottleneck layer (low-dim embedding), decoder reconstructs the original data.
- Training objective: reproduce the dataset while restricting capacity via the bottleneck (self-supervised).
- Example architecture: 5 inputs → bottleneck of 2 latent neurons → 5 outputs.

## Important Formulas & Diagrams

- **Cost function with regulariser**: `cost(f) = -l(f) + regularizer(f)`
- **Kernel trick**: Original scalar product `<x, y>` becomes `φ(x, y)` with a kernel function.
- **Sample Covariance Matrix**: `S = (1/n) X^T X`
- **SVD Decomposition**: `M = U Σ V^T`, where M ∈ R^(n×m), U ∈ R^(n×r), V ∈ R^(m×r), Σ ∈ R^(r×r).
- **Diagram**: Original non-linearly separable data (left) vs. transformed high-dimensional data that is linearly separable (right).
- **PCA visualisation example**: Ancient human genomes projected onto first two principal components, showing clustering by geography/ancestry.
- **PCA bias example**: Same ancient DNA with different modern reference populations yields different clusters, illustrating that PCA results "may not be reliable, robust, or replicable."
- **Autoencoder diagram**: Input → encoder → bottleneck (latent space) → decoder → output.
