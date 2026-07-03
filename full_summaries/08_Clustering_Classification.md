# KDDM1 — Lecture 08: Clustering & Classification

**Lecturers:** Roman Kern & Denis Helic, TU Graz (ISDS) | **Version:** 2.0.2
**Topics:** k-Means, HAC, Naive Bayes, Decision Trees

---

## 1. Motivation & Learning Objectives

Knowledge discovery finds patterns in data; for clustering these are groups of "similar" data points. The goal is understanding the relation between data, distance measures, and clustering algorithms. For classification, the goal is building prediction models from labelled data without extensive feature engineering. The California housing dataset (via TeachCenter, with a Python notebook) is the running example.

---

## 2. Clustering — Foundations

### 2.1 Definition and Core Objectives

Clustering maximises **intra-group similarity (cohesion)** and minimises **inter-group similarity (separation)**. It is **unsupervised** — no labelled training data, and no external teacher. The optimisation criterion is usually task- and domain-independent.

### 2.2 Use Cases

Grouping similar items (buyer segmentation), text categorisation (search engine results), pattern recognition, outlier detection, and vector quantisation (lossy compression).

### 2.3 Distance Functions

Similarity is expressed via a distance function `d(x, y)` — higher distance means lower similarity. Choice depends on feature type (categorical, numeric). Common choices: Euclidean distance and cosine similarity. All inter-instance distances form an `n x n` **distance matrix** computed in `O(n^2)` time.

### 2.4 Types of Clusterings

| Type | Description | Key Property |
|------|-------------|--------------|
| **Exclusive (Partitioning)** | Each instance belongs to exactly one cluster | Non-overlapping; `C = {C1..Cn}`, mutually exclusive, union = `X` |
| **Fuzzy (Soft)** | Instances can belong to multiple clusters with membership weights | Overlapping clusters |
| **Hierarchical** | Clusters contain sub-clusters in a nested structure | Overlap with parent/child only; yields a tree |

### 2.5 Taxonomy of Algorithms

**Iterative:** Exemplar-based (k-Means), exchange-based (Kernighan-Lin).  
**Hierarchical:** Agglomerative (HAC), divisive (min-cut).  
**Density-based:** DBSCAN, MajorClust.  
**Meta-search:** Simulated annealing, evolutionary strategies.  
**Stochastic:** Gaussian mixtures (EM).  
**Other:** Information theory, subspace, graph-based (clique, spectral, Markov), matrix (co-clustering).

---

## 3. Iterative Clustering — k-Means

### 3.1 Overview

Partitioning algorithm. **Input:** instances `V`, distance `d(v_i, v_j)`, minimisation criterion, number of clusters `k`. **Output:** centroids `r_1..r_k` (mean of instances in each cluster). Contrast with **medoids** — the best representative instance within a cluster.

### 3.2 Algorithm

1. **Initialise** centroids `(r_1..r_k)` — seed selection.
2. Assign each instance `v in V` to the closest centroid.
3. **Update** centroids: `r_i = minimize(e(C_i))`.
4. **Repeat** steps 2-3 until convergence.

### 3.3 Configuration Choices

| Question | Basic Answer | Advanced Variant |
|----------|-------------|------------------|
| Cluster number `k`? | Predefined | **Bisecting k-Means** — start with 2, split largest |
| Initialisation? | Random | **k-Means++** — cover feature space |
| Distance measure? | Euclidean | **Spherical k-Means** — cosine similarity |
| Minimisation criterion? | Sum of squared distances (variance) | **k-Centre** — minimise `max|v - r_i|` |

When data is from a metric space, the sum of squared distances to cluster representatives (variance criterion) is natural — yielding a vector of minimum variance.

### 3.5 Advantages

Conceptually simple, efficient `O(n)`, extendable to fuzzy/online versions, partitions space into a **Voronoi diagram**.

### 3.6 Disadvantages

Only finds **local minima** (mitigate via restarts or k-Means++), requires fixed `k`, fails on elongated/nested/density-based clusters, sensitive to noise and outliers, poor with large size differences between clusters.

---

## 4. Hierarchical Clustering

### 4.1 Two Approaches

1. **Agglomerative (HAC):** Bottom-up — each instance starts as its own cluster, then merge closest pairs.
2. **Divisive:** Top-down — start with one cluster, recursively split.

### 4.2 HAC — Algorithm

**Input:** instances `V`, cluster distance `d_C`. **Output:** dendrogram (cluster hierarchy).  
**Steps:** (1) One cluster per instance. (2) Update distance matrix. (3) Select pair with lowest `d_C(Ci, Cj)`. (4) Merge. (5) Repeat until stop criterion.

### 4.3 Linkage Types

| Linkage | Formula | Characteristic |
|---------|---------|----------------|
| **Single (MIN)** | `min_{u in Ci, v in Cj} d(u, v)` | Contractive — strong chaining, very good outlier detection |
| **Complete (MAX)** | `max_{u in Ci, v in Cj} d(u, v)` | Dilating — small clusters, low chaining, poor outlier detection |
| **Group average** | `1/(|Ci||Cj|) * sum d(u, v)` | Conservative — compact clusters, medium chaining |
| **Ward (variance)** | `sqrt( (2|Ci||Cj|)/(|Ci|+|Cj|) ) * ||u_mean - v_mean||` | Conservative — spherical clusters, low chaining |
| **Centroid** | `d(r_i, r_j)` | Distance between centroids |

**Comparison:** Single link is contractive, yields low cluster count, extended form, strong chaining, very good outlier detection. Complete link is dilating, high cluster count, small form, low chaining, poor outlier detection. Average and Ward are conservative.

### 4.4 HAC — Stopping Criteria

Fixed number of clusters, distance threshold, maximum cluster size, or cluster quality (e.g., **silhouette coefficient**).

### 4.5 HAC — Advantages & Disadvantages

**Advantages:** Arbitrary distance/similarity measures, no need to pre-specify cluster count.  
**Disadvantages:** `O(n^2)` to `O(n^3)` complexity, `O(n^2)` memory for distance matrix, greedy with no backtracking.

---

## 5. Preprocessing & Postprocessing for Clustering

### 5.1 Preprocessing

Remove noise/outliers (k-Means is especially sensitive), normalise data, estimate cluster count (model selection), reduce dimensionality (SVD). Canopy clustering can approximate seeds for k-Means and help partition data in Map/Reduce.

### 5.2 Postprocessing

Remove spurious clusters (empty/small), validate/reject solutions, merge close clusters, split inhomogeneous clusters.

### 5.3 Conclusions

Many approaches with varying assumptions; preprocessing and hyperparameter tuning are essential; evaluating clustering solutions is a fundamental challenge (no ground truth).

---

## 6. Classification — Foundations

### 6.1 Types of Learning

| Type | Description |
|------|-------------|
| **Unsupervised** | No desired outputs in training data (clustering) |
| **Supervised (Inductive)** | Training data includes desired outputs (labels) |
| **Semi-supervised** | Few labelled + many unlabelled instances |
| **Reinforcement** | Rewards from a sequence of actions |

### 6.2 Supervised Learning Formalism

Given examples `(x, f(x))` or `(x, y)` with `y = f(x)`, learn `f` for new `x`.  
- **Classification:** `f(x)` discrete (class labels).  
- **Regression:** `f(x)` continuous.  
- **Probability estimation:** `f(x)` is a probability distribution.

### 6.3 Terminology

| Term | Definition |
|------|------------|
| **Training example** | `(x, f(x))` |
| **Target function** | The true `f(x)` |
| **Hypothesis** | Proposed `h` approximating `f` |
| **Classifier** | Discrete-valued; `f(x) in {1..K}` are **classes** |
| **Hypothesis space** | All possible hypotheses |

### 6.4 Inductive Bias & Variance-Bias Trade-off

- **High bias** (strong assumptions, e.g., linearity): applies to few datasets but low variance.
- **Low bias** (weak assumptions, e.g., smoothness): fits complex patterns but can **overfit** (fails to generalise).

### 6.5 Model Selection

Try multiple models and evaluate, or inspect data, formalise assumptions and select appropriately.

---

## 7. Naive Bayes Classifier

### 7.1 Bayes Approach

Applies Bayes' theorem to classification: "what is the most probable classification of a new instance given the data?" rather than "what is the most probable hypothesis?"

### 7.2 MAP Classification

For instance `x = <v_1..v_n>` and classes `C`:

```
c_MAP = argmax_{c_j in C} P(c_j | v_1..v_n)
     = argmax_{c_j in C} P(v_1..v_n | c_j) * P(c_j)
```

### 7.3 Probability Estimation

- `P(c_j)`: estimated via class frequency (MLE).
- `P(v_1..v_n | c_j)`: requires extensive data for large `n`.

### 7.4 The Naive Assumption

**Conditional independence** of attributes given the class:

```
P(v_1..v_n | c_j) = product_i P(v_i | c_j)
```

This assumption often does **not** hold in practice, yet Naive Bayes performs well. `P(v_i | c_j)` is estimated via MLE (counting). Smoothing (e.g., Laplace) prevents zero-probability estimates.

---

## 8. Decision Trees

### 8.1 Decision Stump

Simplest tree: a **single feature** with a threshold for binary classification. A good split point is where class probability distributions meet. Feature selection uses **information gain** or **Gini impurity**.

### 8.2 Decision Tree

Stacks multiple stumps into a tree of decisions, recursively partitioning the feature space. Each node creates a subset; the root feeds succeeding decisions.

### 8.3 Advantages

Little preprocessing (no normalisation), interpretable (small trees), supports numeric and categorical features, handles missing data.

### 8.4 Disadvantages

Not robust (high variance — small data changes yield different trees), prone to **overfitting**.

### 8.5 Assumptions

Decision boundaries are **perpendicular to feature axes** (no feature combinations); smoothness/monotonicity within one feature; greedy optimisation (each split locally improves performance; no backtracking).

---

## 9. Summary & Conclusions

### 9.1 Cross-Cutting Considerations

- **Multicollinearity:** Redundant/correlated features bias distance functions and some algorithms (e.g., linear regression).
- **Explainability:** Understanding how a model works and how it arrived at a decision is critical.
- **Imbalanced datasets:** Skewed class distributions bias models toward the majority class.

### 9.2 Conclusions on Classification

All classification methods rest on specific hypotheses and assumptions. The presented algorithms are **basic versions** — more advanced variants (e.g., Random Forest over plain Decision Tree) are typical in practice. Choosing the right model requires means to assess prediction quality (evaluation metrics, cross-validation).

### 9.3 Closing Remarks

Clustering finds patterns without labels; classification predicts labels from examples. Both share challenges in feature engineering, parameter selection, evaluation, and interpretation. Algorithm choice depends on data characteristics (size, dimensionality, sparsity, noise) and application requirements (speed, interpretability, accuracy).
