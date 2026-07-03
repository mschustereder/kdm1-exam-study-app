# KDDM1 — Dimensionality Reduction (Full Summary)

**Lecturers:** Roman Kern & Denis Helic (TU Graz, ISDS) — Version 2.1.1  
**Source:** `07_Dim_Reduction.pdf` (38 slides)

---

## 1. Motivation & Core Goals

- **Problem:** Real-world datasets often exhibit complex interdependencies among features, leading to correlations and redundancy. High-dimensional feature spaces harm runtime performance, reduce representativeness (curse of dimensionality), and degrade algorithm quality (e.g., DBSCAN clustering).
- **Goal:** Reduce redundancies via three complementary strategies — regularisation (penalise complexity), dimensionality reduction (compress/decorrelate), and clustering (group similar instances).
- **Idea:** Transform the original features into a new, smaller representation that removes redundancies and decorrelates features while preserving as much information as possible.

---

## 2. Recap — Foundations

### 2.1 Distance Functions
- Many algorithms (explicitly or implicitly) rely on a distance function.
- The distance function must match the data; pre-processing (e.g., normalisation) is often required to equalise feature value ranges.

### 2.2 Feature Engineering Process
1. Remove unnecessary features.
2. Remove redundant features.
3. Create new features — combine existing ones, transform them, use contextual or external sources.
4. Modify feature types (e.g., binary → numeric) and feature values.

### 2.3 Example — Splitting a Feature
- A dataset with male/female participants includes a single `height` feature.
- Height distributions for each sex are individually normal, but the joint distribution is not normal.
- It may be beneficial to split `height` into two separate features (`male_height`, `female_height`) to better capture group-specific patterns.

### 2.4 Feature Selection Approaches
- **Heuristics:** Rule-of-thumb strategies.
- **Filter methods:** Evaluate features independently of a model (e.g., compute information gain between a feature and the target).
- **Wrapper methods:** Search the feature subset space using a predictive model — **forward selection** (add features greedily) and **backward elimination** (remove features greedily).

---

## 3. Regularisation & Feature Transformation

### 3.1 Regularisation (Implicit Feature Selection)
- **Basic idea:** Introduce a penalty term for model complexity into the cost/loss function.
- Complexity grows with the number of features (especially problematic when features exceed observations).
- **Formula (example with negative log-likelihood):** `cost(f) = -l(f) + regularizer(f)`
- **Common regularisers:**
  - **L0:** Count of non-zero features (subset selection).
  - **L1 (Lasso):** Sum of absolute feature values (drives some weights to zero).
  - **L2 (Ridge):** Sum of squared feature values (shrinks weights but does not zero them).

### 3.2 Feature Transformation (to Higher Dimensions)
- **Motivation:** Some algorithms (e.g., single-layer perceptrons) can only solve linearly separable problems — e.g., the XOR problem is not linearly separable in the original space.
- **Solution:** Map the original features into a **higher-dimensional** space. More dimensions increase the likelihood that the problem becomes linearly separable.
- **Kernel Trick:** For algorithms relying on scalar products (e.g., SVMs), the transformation can be computed "on-the-fly" via a kernel function `phi(x, y)` instead of explicitly constructing the high-dimensional space.
- **Examples:** Gaussian (RBF) kernel, polynomial kernel — these typically introduce tunable parameters.

### 3.3 Key Contrast
- Regularisation penalises complexity and implicitly selects features.
- Feature transformation (with kernels) deliberately creates more features to aid linear separability, accepting increased dimensionality.

---

## 4. Dimensionality Reduction — Overview

### 4.1 Motivation (Detailed)
- High dimensions hurt runtime, dilute representativeness, and degrade algorithm performance.
- Features often exhibit dependencies → correlations. Dimensionality reduction aims to find a **low-dimensional embedding** that captures the essential structure.

### 4.2 Applications
- **Visualisation:** Scatter plots in 2D/3D for human inspection and interaction.
- **Preprocessing:** Reduce noise and improve downstream model performance.
- **Compression:** Store and process data more efficiently.
- **Noise removal:** Discard low-variance dimensions that are assumed to contain noise.

### 4.3 Initial Mathematical Setup
- Given a data matrix **D** with `n` dimensions (features) and `m` observations.
- **Centering:** Subtract the mean of each feature to obtain matrix **X** (zero-mean columns). This centres the data at the origin.

### 4.4 Sample Covariance Matrix
- Compute: **S** = (1/n) **X^T X**
- Each element S_ij is the covariance between dimension `i` and dimension `j` (variance on the diagonal).
- **Properties:** Symmetric, positive semi-definite, all eigenvalues are real, eigenvectors are orthonormal → they form an eigenbasis of the n-dimensional space.

---

## 5. Principal Component Analysis (PCA)

### 5.1 Core Concept
- PCA transforms the original (centered) data matrix **D** onto a new orthonormal basis.
- **Principal Components (PCs):** Axes of the new basis, each is an eigenvector of the covariance matrix, ordered by decreasing variance.
- **Dimensionality reduction:** Keep only the first `k` PCs (k << n), discarding the rest. This yields an approximation of **D** that captures the maximum possible variance for the chosen `k`.

### 5.2 Relation to Information Theory
- PCA can be viewed as lossy data compression — low-variance dimensions contain little information.
- If the data is Gaussian, PCA is **information-theoretically optimal**: the top PCs carry maximal information content.

### 5.3 Visual Example (Genetics)
- PCA applied to ancient human genomes (Nature 2014) — modern and ancient DNA samples projected onto the first two PCs reveal population structure. Different ancestries (Palaeolithic, Mesolithic, Neolithic from Iran, Israel, Caucasus, Romania, Scandinavia, Central Europe) cluster in distinct regions of the PC plot.

### 5.4 Limitations of PCA
- **Linearity only:** Each PC is a linear combination of original features; PCA cannot capture non-linear relationships.
- **Choosing k:** Commonly selected to retain a target percentage of total variance (e.g., 95%). Knee/elbow point detection on the explained-variance curve also helps. For visualisation, k=2 or k=3 is typical.
- **Sensitivity to scaling:** Since PCA is variance-based, features with larger ranges dominate. Normalisation is essential.
- **Outlier sensitivity:** Extreme points can skew the covariance estimate and thus the PCs.
- **Information loss:** For k < n, some variance (information) is inevitably discarded — the method is optimal only with respect to variance preservation.

### 5.5 Reliability Concerns
- A 2022 study (Elhaik, Sci Rep) demonstrated that PCA-based findings in population genetics can be **biased, non-robust, and non-replicable** depending on the choice of reference populations. The same ancient DNA may appear in different clusters when combined with different modern samples, highlighting the need for caution.

---

## 6. Singular Value Decomposition (SVD)

### 6.1 Motivation
- PCA operates on the (square) covariance matrix. SVD generalises to **non-square** matrices, making it more flexible.

### 6.2 Key Intuition
- SVD assumes a small number of latent "concepts" connect the rows and columns of the matrix.
- These concepts are ranked from most to least important (by singular values).
- Truncating (removing) the least important concepts yields a reduced matrix that closely approximates the original.

### 6.3 Formal Definition
- Let **M** be an (n x m) matrix with **rank r** (the maximum number of linearly independent rows or columns). Then:
  - **U** (n x r): column-orthonormal matrix.
  - **V** (m x r): column-orthonormal matrix.
  - **Sigma** (r x r): diagonal matrix (singular values on the diagonal, sorted descending).
- **Decomposition:** **M = U Sigma V^T**

### 6.4 Thin SVD & Applications
- Choose `k < r` — this is called **Thin SVD** (or truncated SVD) and introduces information loss.
- **Latent Semantic Indexing (LSI):** Classic application on text data (very high-dimensional term-document matrices). After SVD, distance functions operate in a low-dimensional embedding space, enabling efficient information retrieval (search).

---

## 7. t-Distributed Stochastic Neighbour Embedding (t-SNE)

### 7.1 Overview
- **Non-linear** dimensionality reduction technique, primarily used for **visualisation**.
- Preserves the **probabilistic neighbourhood structure** from high-dimensional space in the low-dimensional embedding.

### 7.2 Algorithm (Step-by-Step)
1. **High-dimensional probabilities (p_ij):** Compute pairwise probability that point `j` is a neighbour of point `i`, assuming a Gaussian distribution centred at each point. The standard deviation is estimated locally based on `k` neighbours (controlled by the **perplexity** hyperparameter).
2. **Initialisation:** Randomly initialise points in the low-dimensional (2D/3D) space.
3. **Low-dimensional probabilities (q_ij):** Compute pairwise probabilities in the low-dimensional space using a **Student t-distribution** (heavy-tailed, which alleviates the "crowding problem" by allowing greater distances).
4. **Optimisation:** Minimise the **Kullback-Leibler (KL) divergence** between the high-dimensional distribution **P** and the low-dimensional distribution **Q** using gradient-based methods. Iterate (or converge, then optionally restart from step 2).

### 7.3 Strengths & Weaknesses
- **Strengths:** Excellent at preserving local structure — clusters and manifolds are often clearly visible.
- **Weaknesses:** Global structure is not reliably preserved. The vanilla version uses Euclidean distance in high-dimensional space and thus suffers from the curse of dimensionality. The cost function is non-convex, meaning different runs can produce different results.
- **UMAP (Uniform Manifold Approximation and Projection):** An extension that better preserves global structure and is considered superior as a preprocessing step before clustering.

---

## 8. Autoencoders (Deep Learning Approach)

### 8.1 Concept
- A neural network trained to **reproduce its input** (self-supervised / technically unsupervised).
- The network is constrained by a **bottleneck layer** in the middle with far fewer neurons than the input/output layers.
- The bottleneck forces the network to learn a compressed latent representation (the embedding), discarding noise and redundancy.

### 8.2 Architecture (Example)
- 5 input neurons → a small number of hidden layers → 2 latent neurons (the bottleneck) → symmetrical decoder layers → 5 output neurons.
- The latent layer values are the low-dimensional embedding of each input sample.

### 8.3 Key Advantages
- Can learn **non-linear** transformations (unlike PCA/SVD).
- The distance function can be **learned** implicitly during training.
- Flexible architecture: convolutional autoencoders for images, variational autoencoders (VAEs) for generative modelling.

---

## 9. Conclusions

- The core goal across all methods is to transform a **high-dimensional feature space** into a **low-dimensional embedding space** that retains meaningful structure.
- **Most methods require numeric features** and a meaningful distance function.
- **Preprocessing (normalisation, centering)** is essential before applying dimensionality reduction.
- When `k < n`, information loss is inevitable — the choice of method and hyperparameters determines what type of information is preserved (global variance for PCA, local neighbourhoods for t-SNE, latent concepts for SVD, learned representations for autoencoders).

---

## 10. Quick Reference Table

| Method | Type | Linear? | Preserves | Typical Use |
|--------|------|---------|-----------|-------------|
| PCA | Feature transform | Yes | Global variance | Preprocessing, compression |
| SVD | Matrix factorisation | Yes | Latent concepts | Text (LSI), recommendation |
| t-SNE | Neighbourhood embedding | No | Local structure | Visualisation (2D/3D) |
| UMAP | Neighbourhood embedding | No | Local + global | Visualisation, preprocessing |
| Autoencoder | Neural net | Can learn non-linear | Learnt manifold | Compression, feature learning |
| Regularisation | Cost penalty | Depends on model | Model simplicity | Implicit feature selection |
| Kernel trick | Implicit transform | Linear in transformed space | Separability | SVMs, non-linear problems |
