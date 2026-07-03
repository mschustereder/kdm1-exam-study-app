# 08 -- Clustering & Classification

**Course:** KDDM1 (Knowledge Discovery and Data Mining 1)  
**Authors:** Roman Kern & Denis Helic, TU Graz  
**Version:** 2.0.2

---

## 1. Clustering (Unsupervised Learning)

**Core idea:** Group data points so that intra-cluster similarity is maximized and inter-cluster similarity is minimized. No labeled training data is needed -- the optimisation criterion is task/domain-independent.

### Distance & Similarity
- Similarity is expressed via a **distance function** `d(x, y)` (higher distance = lower similarity).
- The choice of distance depends on feature type (numeric, categorical, etc.).
- All pairwise distances form an **n x n distance matrix** (computed in `O(n^2)` time).

### Types of Clusterings
- **Exclusive (hard):** Each instance belongs to exactly one cluster (partitioning).
- **Fuzzy (soft):** Instances can belong to multiple clusters with a weight for each.
- **Hierarchical:** Clusters contain sub-clusters; instances belong to a cluster and all its parents.

### Algorithm Categories
- **Iterative:** Exemplar-based (k-Means), exchange-based (Kerninghan-Lin).
- **Hierarchical:** Agglomerative (HAC -- bottom-up), Divisive (min-cut -- top-down).
- **Density-based:** DBSCAN (point density), MajorClust (attraction).
- **Meta-search:** Simulated annealing, evolutionary strategies.
- **Stochastic:** Gaussian mixtures (EM algorithm).
- **Graph-based:** Spectral clustering, Markov clustering, clique-based.

---

### k-Means (Iterative / Partitioning)

**Input:** Instances `V`, distance function `d`, minimization criterion, number of clusters `k`.  
**Output:** `k` centroids (the mean of instances within each cluster).

**Algorithm:**
1. Initialise centroids (seed selection).
2. Assign each instance to the closest centroid.
3. Recompute centroids based on the minimization criterion.
4. Repeat steps 2-3 until convergence.

**Variants on open questions:**
| Question | Basic choice | Advanced variant |
|---|---|---|
| Cluster number `k` | Predefined | Bisecting k-Means (split largest cluster) |
| Initialisation | Random | k-Means++ (cover feature space) |
| Distance measure | Euclidean | Spherical k-Means (cosine similarity) |
| Minimisation | Arithmetic mean | k-center (max distance to centroid) |

**Advantages:** Simple, efficient O(n), produces hard clusters (extendable to fuzzy), works iteratively (online k-Means), yields a Voronoi partition.  
**Disadvantages:** Finds only local minima, needs k fixed, poor on elongated/density-based clusters, sensitive to noise and outliers.

---

### Hierarchical Agglomerative Clustering (HAC)

**Input:** Instances `V`, inter-cluster distance measure `d_C`.  
**Output:** A dendrogram (cluster hierarchy).

**Algorithm:**
1. Start with one cluster per instance.
2. Update the distance matrix.
3. Merge the closest pair of clusters (argmin of `d_C`).
4. Repeat steps 2-3 until a stop criterion is met.

**Stop criteria:** number of clusters, distance threshold, cluster size, silhouette coefficient.

**Linkage (inter-cluster distance) functions:**

| Linkage | Formula | Behaviour |
|---|---|---|
| Single link (MIN) | `min d(u,v)` | Contractive, extended clusters, strong chaining, good outlier detection |
| Complete link (MAX) | `max d(u,v)` | Dilating, small/compact clusters, low chaining, poor outlier detection |
| Group average | Mean of all pairwise distances | Conservative |
| Ward criterion | Variance-weighted distance | Conservative, spherical clusters |

**Advantages:** Works with arbitrary distance/similarity measures; no need to specify k.  
**Disadvantages:** O(n^2) to O(n^3) complexity; O(n^2) memory for the distance matrix; greedy with no backtracking.

---

### Pre- & Post-processing for Clustering

**Preprocessing:** Remove noise/outliers, normalise data, estimate k, reduce dimensionality (e.g., SVD). Optional: Canopy clustering for k-Means seed initialisation.  
**Postprocessing:** Remove spurious clusters, validate results, merge close clusters, split inhomogeneous clusters.

---

## 2. Classification (Supervised Learning)

**Core idea:** Given labeled training examples `(x, f(x))`, build a model `h` (hypothesis) that approximates the target function `f` and predicts labels for unseen data.

### Types of Learning
- **Unsupervised:** No desired outputs in training data.
- **Supervised (inductive):** Training data includes desired outputs -- **classification** (discrete `f(x)`), **regression** (continuous `f(x)`), or **probability estimation**.
- **Semi-supervised:** A few labels are available.
- **Reinforcement:** Reward from sequences of actions.

### Key Terminology
- **Target function:** The true unknown function `f`.
- **Hypothesis:** A proposed function `h` believed to approximate `f`.
- **Classifier:** A discrete-valued function; possible outputs are **classes**.
- **Hypothesis space:** The set of all possible hypotheses.
- **Inductive bias:** The assumptions a model makes about the data.

**Variance-Bias Trade-off:** Strong bias (e.g., linearity) applies to few datasets; low bias (e.g., smoothness) risks overfitting (good on training data, poor on unseen data).

---

### Naive Bayes Classifier

**Approach:** Use Bayes' theorem to find the **MAP (Maximum A Posteriori)** class for a new instance.

**Formula:**
```
c_MAP = argmax_{c_j in C} P(c_j) * P(v_1, v_2, ..., v_n | c_j)
```

**Naive Assumption:** Feature values are conditionally independent given the class:
```
P(v_1,...,v_n | c_j) = ∏_i P(v_i | c_j)
```
- This assumption rarely holds in practice, but Naive Bayes still performs well.
- **P(c_j)** is estimated via MLE (class frequency in the dataset).
- **P(v_i | c_j)** is estimated via MLE (counting), with **smoothing** to avoid zero probabilities.

---

### Decision Tree

**Decision Stump:** A single-feature binary split using a threshold. The best feature is selected via **information gain** or **Gini impurity**.

**Decision Tree:** Stacks multiple stumps into a tree. Each internal node tests a feature; each leaf assigns a class.

**Advantages:** Little preprocessing needed (no normalisation), small trees are interpretable, supports numeric and categorical features, handles missing data.  
**Disadvantages:** Not robust (small data changes can yield different trees), prone to overfitting.  
**Assumptions:** Decision boundaries are perpendicular to feature axes (no feature combinations), splitting is greedy and monotonous.

---

## 3. Summary & Key Considerations

- **Multicollinearity:** Redundant/correlating features bias distance-based algorithms (e.g., vanilla linear regression).
- **Interpretability:** Understanding *how* a model works and *why* it made a decision is a practical concern.
- **Imbalanced datasets:** Skewed class distributions can bias models toward the majority class.
- The algorithms presented are the basic versions -- more sophisticated variants exist (e.g., Random Forest instead of a single Decision Tree).
- Model selection (choosing the right hypothesis for a dataset) requires systematic evaluation, typically via a cost function or performance metric.
