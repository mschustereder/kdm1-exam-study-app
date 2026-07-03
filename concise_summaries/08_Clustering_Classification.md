# 08 — Clustering & Classification — Summary

## Core Definitions & Concepts

- **Clustering**: Unsupervised method that identifies groups (clusters) of similar data points. Objective: (1) maximise intra-group similarity (cohesion) and (2) minimise inter-group similarity (separation). No training labels required.
- **Classification**: Supervised learning method that builds a prediction model from labelled instances. The goal is to approximate a discrete target function f(x) that maps inputs to class labels.
- **Distance / Similarity**: Clustering relies on a distance function d(x,y) — higher distance means lower similarity. All pairwise distances form an n x n distance matrix (O(n^2) in time and memory).
- **Types of Clustering**: Exclusive (each instance in exactly one cluster), Fuzzy / Soft (instances belong to multiple clusters with weights), Hierarchical (clusters nested in sub-clusters).
- **Supervised vs Unsupervised**: Unsupervised — no labelled training data; Supervised — labelled data with desired outputs; Semi-supervised — few labelled examples; Reinforcement — rewards from action sequences.
- **Key terminology in classification**: Training example (x, f(x)), target function f, hypothesis h (a proposed function approximating f), classifier (discrete-valued function mapping to classes), hypothesis space, inductive bias.
- **Variance-Bias Trade-off**: High-bias models (e.g., linearity assumption) apply to a narrow subset of datasets; low-bias models risk overfitting (good on training data, poor generalisation to unseen data).

## Key Algorithms & Techniques

### Clustering — Iterative
- **k-Means**: Partitioning algorithm. Input: instances V, distance function d, number of clusters k. Output: k centroids (mean of instances in each cluster). Steps: (1) initialise centroids (seed selection), (2) assign each instance to nearest centroid, (3) update centroids via minimisation criterion, (4) repeat until convergence.
  - *Variations*: Bisecting k-Means (choose k by splitting largest cluster), k-Means++ (smart seeding to cover feature space), Spherical k-Means (uses cosine similarity), k-center (minimises max distance per cluster).
  - *Complexity*: O(n). Produces Voronoi partitions.
  - *Limitations*: Only finds local minima; requires fixed k; struggles with elongated/density-based clusters, outliers, nested clusters, and clusters of very different sizes.

### Clustering — Hierarchical
- **Hierarchical Agglomerative Clustering (HAC)**: Bottom-up approach — each instance starts as its own cluster, then iteratively merge the closest pair using a distance measure dC for clusters. Output: a dendrogram (cluster hierarchy).
  - *Linkage types*:
    - **Single link** (MIN): dC = min distance between any two points in the clusters — chaining tendency, good outlier detection.
    - **Complete link** (MAX): dC = max distance — produces compact clusters, poor outlier detection.
    - **Group average link**: dC = average of all pairwise distances — conservative, medium chaining.
    - **Ward criterion**: dC based on variance increase when merging — produces spherical clusters.
    - **Centroid**: dC = distance between cluster centroids.
  - *Complexity*: O(n^2) to O(n^3) depending on linkage; high memory if storing distance matrix.
  - *Advantages*: No need to specify k in advance; works with arbitrary distance/similarity measures.

### Other Clustering Algorithm Categories
- **Density-based**: DBSCAN (point density), MajorClust (attraction-based).
- **Stochastic**: Gaussian mixtures / EM.
- **Graph-based**: Clique-based, Spectral clustering, Markov clustering.
- **Matrix-based**: Co-clustering.
- **Meta-search**: Simulated annealing, evolutionary strategies.

### Pre/Post-processing for Clustering
- Preprocessing: Canopy clustering (to seed k-Means), noise/outlier removal, normalisation, estimating k via model selection, dimensionality reduction (e.g., SVD).
- Postprocessing: Removing spurious clusters, validation, merging/splitting clusters.

### Classification — Naive Bayes
- **Bayes classifier**: Applies Bayes theorem to find the most probable class given the data.
  - c_MAP = argmax P(c_j | v_1, ..., v_n) = argmax P(v_1,...,v_n|c_j) * P(c_j).
  - P(c_j) estimated via frequency of classes (MLE).
- **Naive assumption**: Attribute values are conditionally independent given the class — P(v_1,...,v_n|c_j) = ∏ P(v_i | c_j). This rarely holds in practice, but the classifier still performs well.
- **Practical note**: Smoothing is advised to avoid zero-probability estimates from missing attribute-class combinations.

### Classification — Decision Tree
- **Decision Stump**: A one-level decision tree using a single feature and a threshold as splitting criterion for binary classification. Feature selection via information gain or Gini impurity.
- **Decision Tree**: Stacks multiple decision stumps hierarchically. Each split creates a subset of the data; the tree is grown greedily.
- **Advantages**: Little preprocessing required (no normalisation), interpretable when small, handles numeric and categorical features, handles missing data.
- **Disadvantages**: Not robust (small data changes yield different trees), overfitting tendency, greedy search, decision boundaries are perpendicular to feature axes (no feature combinations in vanilla version).

## Important Formulas & Diagrams

- **Distance matrix**: All inter-instance distances d(x,y) in an n x n matrix (O(n^2)).
- **Exclusive clustering definition**: C = {C_1, ..., C_n}, C_i ⊆ X, mutually exclusive, union = X.
- **k-Means centroid**: Mean of all instances assigned to the cluster (vector of minimum variance when sum of squared distances is the minimisation criterion).
- **HAC linkage formulas**:
  - Single link: d_C(C_i, C_j) = min_{u∈C_i, v∈C_j} d(u,v)
  - Complete link: d_C(C_i, C_j) = max_{u∈C_i, v∈C_j} d(u,v)
  - Group average: d_C(C_i, C_j) = (1/|C_i||C_j|) Σ_{u∈C_i} Σ_{v∈C_j} d(u,v)
  - Ward: d_C(C_i, C_j) = sqrt( (2|C_i||C_j|) / (|C_i|+|C_j|) ) * ||ū - v̄||
  - Centroid: d_C(C_i, C_j) = d(r_i, r_j)
- **Bayes classifier**: c_MAP = argmax_{cj∈C} P(v_1,...,v_n|c_j) * P(c_j) (derived via Bayes rule, dropping the denominator).
- **Naive Bayes factorisation**: P(v_1,...,v_n | c_j) = ∏_{i} P(v_i | c_j).
- **Diagrams referenced**: Voronoi diagram (k-Means partitions), dendrogram (HAC hierarchy), decision tree structure, decision stump threshold illustration.

## Learning Objectives

- Understand the relationship between data, distance measures, and clustering algorithms.
- Be able to select appropriate clustering algorithms based on data characteristics and algorithm assumptions.
- Understand how iterative clustering (k-Means) works, its open questions (k, initialisation, distance, minimisation criterion), and its extensions.
- Understand hierarchical clustering (HAC), the role of linkage functions, and the trade-offs between linkage types.
- Recognise the need for preprocessing (normalisation, noise removal, dimensionality reduction) and postprocessing (validation, merging/splitting) in clustering pipelines.
- Understand the distinction between unsupervised (clustering) and supervised (classification) learning.
- Understand the Bayes classifier derivation and the naive (conditional independence) assumption.
- Understand Decision Stump as a single-feature binary classifier and how Decision Trees generalise this via recursive splitting.
- Appreciate the variance-bias trade-off and its implications for model generalisation.
- Be aware of practical considerations: multicollinearity, imbalanced datasets, explainability/interpretability, and the fact that more sophisticated variants (e.g., Random Forest) are typically used in practice.
