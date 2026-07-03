# KDDM1 -- Evaluation: Comprehensive Summary

**Authors:** Roman Kern, Denis Helic (TU Graz, Institute of Machine Learning and Neural Computation)  
**Version:** 2.1.1  
**Goal:** Understand evaluation scenarios and their ability to estimate real-world performance of models.

---

## 1. Motivation and Core Question

After data mining, we discover patterns and build models -- but how well do these models generalise to unseen data? Evaluation answers whether the observed results are spurious or truly predictive. The key challenge is estimating performance on new, unseen data under the assumption that the underlying generative process does not change between training and deployment.

---

## 2. Classifier Evaluation

### 2.1 The Contingency Table (Confusion Matrix)

For binary classification (class 1 = positive/Yes, class 0 = negative/No), predictions are compared against the real class labels in a 2x2 contingency table:

|              | **Real 1** | **Real 0** |
|--------------|------------|------------|
| **Pred. 1**  | True Positive (tp) | False Positive (fp) |
| **Pred. 0**  | False Negative (fn) | True Negative (tn) |

**Terminology:**
- **P** = number of real positive instances = tp + fn (also called prevalence)
- **N** = number of real negative instances = tn + fp
- **n** = total instances = P + N = tp + fn + tn + fp

### 2.2 Accuracy

Accuracy (A) is the fraction of all correct predictions:
- A = (tp + tn) / (tp + fp + fn + tn) = (tp + tn) / n
- Equivalent formulations: A = (tp / P * P + tn / N * N) / n when weighted by class size.

**The Accuracy Paradox:** With skewed class distributions, a trivial classifier that always predicts the majority class achieves high accuracy but provides no useful signal. Example: cancer prevalence P(cancer) = 0.008, P(no cancer) = 0.992. Always predicting "no cancer" yields 99.2% accuracy yet misses every positive case (recall = 0).

### 2.3 Precision and Recall

**Recall (R)** -- also called hit rate, sensitivity, true positive rate:
- R = tp / P = tp / (tp + fn)
- Answers: "How many of the real positives did we correctly identify?"
- In the cancer example, always predicting majority class gives R = 0.

**Precision (Pr)** -- also called positive predictive power:
- Pr = tp / (tp + fp)
- Answers: "Of the instances we labelled positive, how many were actually positive?"
- If we always predict class 1, recall = 1 but precision drops dramatically.

### 2.4 Precision-Recall Trade-off

- To **increase recall**, predict more 1s -- but this also increases false positives, so precision drops.
- To **increase precision**, predict fewer 1s (be more conservative) -- but this increases false negatives, so recall drops.
- This trade-off is inherent and characteristic of each classifier. The **precision-recall curve** visualises this relationship: as recall varies from 0 to 1, precision varies inversely.

### 2.5 F1 Measure and F-beta Measure

**F1 Score** -- the harmonic mean of precision and recall:
- F1 = 2 * (Pr * R) / (Pr + R)
- Always lies between precision and recall, with a tendency toward the lower value.
- Also called F-score.

**F-beta Score** -- generalised form where beta controls the importance of recall:
- F_beta = (1 + beta^2) * (Pr * R) / (beta^2 * Pr + R)
- beta = 1: equal weight (F1).
- beta > 1: recall is more important (e.g., F2 for outlier detection where missing an outlier is costly).
- beta < 1: precision is more important (e.g., web search where false positives annoy users).
- In terms of counts: F_beta = (1+beta^2)*tp / ((1+beta^2)*tp + beta^2*fn + fp).

**F-score discussion:**
- Not symmetric -- swapping the positive/negative labels changes the semantics.
- Does not consider true negatives at all; only tp, fp, fn matter.

### 2.6 Matthews Correlation Coefficient (MCC)

- A measure of correlation/association between the real class and the predictions.
- Treats both variables symmetrically.
- MCC = (tp * tn - fp * fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))
- Considered by some to be the most appropriate single default metric for binary classification as it uses all four cells of the confusion matrix.

### 2.7 ROC Curve and AUC

**Receiver Operating Characteristic (ROC)** curve:
- Applicable when the classifier outputs a continuous score or probability (not just a hard label).
- Plots **true positive rate** (recall = benefit) against **false positive rate** (fp/N = cost) as the decision threshold varies.
- Originally developed for signal detection in World War 2.
- Logistic regression provides an a-posteriori probability, making it naturally suited for ROC analysis.
- The curve provides a full overview of the trade-off; one can choose a specific "working point" (threshold) depending on the scenario.

**Area Under the Curve (AUC):**
- A single scalar summarising the entire ROC curve.
- AUC = 1 corresponds to a perfect classifier; AUC = 0.5 corresponds to random guessing.
- Commonly used for model selection, e.g., in outlier detection.

### 2.8 Multi-Class Settings

For problems with n classes (y_i in {c1, ..., cn}), binary metrics are extended:

- **Micro F1:** Global average over all instance-level predictions (equivalent to accuracy). Biased toward majority classes.
- **Macro F1:** Average of per-class F1 scores. Does not consider class sizes -- requires balanced classes for fair interpretation.
- **Weighted F1:** Average of per-class F1 scores weighted by the number of true instances (support) of each class. Balances class size with class performance.

### 2.9 Reflection on Classifier Evaluation Measures

- **Accuracy:** Not preferred, especially when minority classes matter.
- **MCC:** Reflects the overall trade-off using all four counts.
- **ROC + AUC:** Visualises the trade-off; AUC enables model selection across thresholds.
- **Weighted F1 / Precision / Recall:** Good default starting point for most tasks.
- Always consider the **cost of mis-classification** in the specific domain.

### 2.10 Evaluation for Regression

For continuous target variables, common metrics include:

- **Root Mean Squared Error (RMSE):** sqrt(mean((y_hat - y)^2)). Larger deviations have disproportionate impact (sensitive to outliers). Units match the target variable.
- **Mean Absolute Error (MAE):** mean(|y_hat - y|). Outliers have less impact. Units match the target variable.
- **Coefficient of Determination (R^2):** Proportion of variance in the target variable explained by the model. Ranges from -inf to 1; higher is better.

---

## 3. Baselines

### 3.1 Purpose

Given evaluation measures, how do we know if the results are "good"? Baselines provide a reference point reflecting the hardness of the task.

### 3.2 Types of Baselines

| Baseline Type | Description |
|---------------|-------------|
| **Lower bound** | Threshold of no utility -- any model must exceed this. |
| **Trivial solution** | Performance easily achieved (e.g., always predict majority class). |
| **State of the art (SotA)** | Currently known best solution on the task/dataset. |
| **Human performance** | Average human or well-trained expert performance. |
| **Upper bound** | Theoretical limit (e.g., Bayes error rate). |

### 3.3 Common Baseline Classifiers

- **Constant classifier:** Always predicts one (usually the majority) class. Accuracy equals the proportion of the majority class in the data.
- **Random guessing:** Predicts labels randomly, optionally respecting the class distribution. For equal-sized binary classes: 50% accuracy, precision, recall, F1.

### 3.4 Baseline Reflections

- Baselines are always needed -- classifiers that do not beat trivial solutions provide little benefit.
- Results that seem too good (e.g., outperforming human baselines) should be questioned -- especially since ground truth often comes from human annotations.
- In science, multiple SotA baselines are expected; comparing only against weak baselines is considered dishonest.
- Even if performance is worse than a baseline, other factors (runtime, memory, interpretability) may be relevant.

---

## 4. Cluster Evaluation

### 4.1 The Challenge

> "The validation of clustering structures is the most difficult and frustrating part of cluster analysis." -- Jain & Dubes (1990)

Clustering is unsupervised; there is no single "correct" clustering. The open question is: which of the possible clusterings is the correct one?

### 4.2 Overview of Approaches

**By evaluator:**
- **Human evaluation:** Domain expert judges results -- subjective, hard to reproduce, labour/time intensive.
- **Algorithmic evaluation:** Automated metrics.

**By evaluation goal:**
- Runtime vs. quality of results.

### 4.3 Algorithmic Evaluation: Three Categories

1. **External validity:** Compare clustering against a reference (e.g., ground-truth classes/partition).
2. **Internal validity:** Analyse intrinsic characteristics of the clustering (cohesion vs. separation).
3. **Relative validity:** Analyse sensitivity during clustering generation (e.g., varying algorithm parameters).

### 4.4 External Validity Measures

**Purity:**
- For each cluster: purity(C_i) = (1/|C_i|) * max_j(|C_i ∩ C_j*|), where C_j* are ground-truth classes.
- Measures how "pure" a cluster is (dominant class / cluster size).
- Favours smaller clusters (trivially achieving purity = 1 with one instance per cluster).

**Rand Index:**
- Based on pairwise comparison of all instance pairs.
- For each pair: same cluster & same class = TP, same cluster & diff class = FP, diff cluster & same class = FN, diff cluster & diff class = TN.
- Rand Index = (TP + TN) / (TP + TN + FP + FN).

**V-Measure:**
- Based on conditional entropy from information theory.
- **Homogeneity (h):** Each cluster contains only members of a single class. h = 1 - H(C* | C) / H(C*). (0 if H(C*, C) = 0)
- **Completeness (c):** All members of a given class are assigned to the same cluster. c = 1 - H(C | C*) / H(C). (0 if H(C, C*) = 0)
- **V_beta = (1+beta) * h * c / (beta * h + c):** Weighted harmonic mean of homogeneity and completeness.

**Reflection:** External evaluation is the preferred way when class labels are available and represent the desired grouping. Measures differ in their biases (e.g., number of clusters influences the result).

### 4.5 Internal Validity Measures

Use only the data and the clustering itself -- no external labels. The objectives: maximise intra-group similarity (cohesion) and minimise inter-group similarity (separation).

**Silhouette Coefficient:**
- For each instance i: s(i) = (b(i) - a(i)) / max{a(i), b(i)}
  - a(i) = average distance to other instances in the same cluster (cohesion).
  - b(i) = average distance to instances in the nearest neighbouring cluster (separation).
- s(i) ranges from -1 (wrong cluster) through 0 (on boundary) to +1 (well-clustered).
- Overall silhouette score = average over all instances.
- Visualised as a silhouette plot (each cluster in one colour, each line = one instance).
- Requires a meaningful distance measure; appropriate for k-Means but not for density-based clustering.

**Edge Correlation:**
1. Compute the pairwise similarity matrix from the data.
2. Compute the occurrence matrix (1 if two instances are in the same cluster, 0 otherwise).
3. Compute the correlation between the two matrices.

### 4.6 Internal Validity Reflections

The validation measure must match the assumptions of the clustering algorithm. For instance, Silhouette Coefficient is appropriate for k-Means (spherical clusters) but not for density-based clusterings (e.g., DBSCAN).

---

## 5. Dataset Handling

### 5.1 Core Principle

Evaluation aims to estimate performance on **unseen data**. The key assumption is that the generative process underlying training and unseen data does not change.

### 5.2 The Danger of Overestimation

Using the same data for training and testing produces overly optimistic estimates because the model may have **overfit** (e.g., a classifier that simply memorises the training instances). It will achieve perfect training-set performance but fail to generalise (interpolate/extrapolate).

### 5.3 Dataset Splitting

Randomly partition the labelled dataset into:

- **Training split:** Used to fit the model.
- **Validation split (optional):** Used for hyper-parameter tuning, model selection, visual inspection. Not for final performance reporting.
- **Test split:** Used **only once** for final performance evaluation.

Common split ratio: 80/20 (train/test) or 70/15/15 (train/validation/test).

### 5.4 The "Test Data Used Once" Rule

It is forbidden to use test results to inform improvements -- doing so effectively turns the test set into training data, invalidating the evaluation. Hence the validation split exists for iterative development.

### 5.5 Dataset Splitting Reflections

- **Small datasets / small classes:** A single split may leave too few (or zero) instances of a class in a partition. **Stratification** (preserving class proportions across splits) mitigates this.
- **Single split = single estimate:** Random variation is not accounted for. Use cross-validation instead.
- **i.i.d. assumption:** The data is assumed independently and identically distributed -- same generative process, same sampling, no covariate shift.
- **Dependencies (temporal/spatial):** Any structure in the data must be respected when creating splits (e.g., time-series splits, grouped/cross-validation splits that avoid data leakage).

### 5.6 Cross-Validation

**k-fold cross-validation:**
1. Split the dataset into k equally sized folds (optionally stratified).
2. Iterate k times: use one fold for testing, the remaining k-1 folds for training.
3. Report the **average performance** across all folds and the **variance** (standard deviation).

- Typical k: 3 to 10.
- When k equals the number of data points: **Leave-One-Out (LOO) cross-validation**.

---

## 6. A/B Testing

### 6.1 Setting and Motivation

When users interact with a system (e.g., recommender system), their behaviour provides implicit feedback (e.g., purchases, clicks). A/B testing compares two settings/models/algorithms by measuring differences in observed user behaviour.

### 6.2 Setup

- The user base is randomly split into two groups (A and B).
- Each group experiences one version of the system (control vs. treatment).
- No sample bias is allowed in the split.
- Differences in observed user behaviour (e.g., conversion rate, click-through rate) serve as an estimate of relative performance.
- Directly measures cost/benefit in real-world terms.

### 6.3 Statistical Considerations

A/B testing is a statistical hypothesis test:
- **Null hypothesis (H0):** There is no difference between the two systems.
- Compute a **p-value** to assess statistical significance.
- If H0 can be rejected (p < significance threshold), the improved system can be deployed.

### 6.4 Hill Climbing for A/B Testing

Instead of a single comparison, continuously iterate:
1. Compare current system (control) vs. a new variation (e.g., new model, changed hyperparameters).
2. If the variation is statistically significantly better, adopt it as the new control.
3. Introduce a new variation and repeat.
This accounts for changes in user behaviour over time and enables continuous improvement.

---

## 7. Summary and Best Practices

### 7.1 Ablation Tests

Many models comprise multiple components or parameters. **Ablation** (or "ablation study") removes or modifies one component at a time and measures the change in performance:

- Example: Remove a preprocessing step (e.g., data cleaning, missing value imputation, feature scaling) and observe the performance difference.
- The change in performance informs about the relative importance of each component.
- Can be extended from a single comparison to a series of experiments.
- Example series: Vary the training dataset size by successively removing data points and plot the resulting performance curve to study data efficiency.

### 7.2 Error Analysis

- Manually examine cases where the model failed (e.g., misclassified instances).
- Identify root causes and patterns/groups of failures.
- Provides actionable insights for further model improvement.
- Considered good practice and is standard in scientific reporting.

### 7.3 Conclusions

- Evaluation is required to estimate expected real-world performance.
- Assumptions for evaluation are sometimes **stronger** than those for the model itself.
- Test data can only be used for comparison -- not for informing improvements.
- Ablation tests and error analysis are good scientific practice.
- Succeeding (replication/validation) experiments help confirm the findings.

---

*End of summary. Source: KDDM1 Lecture Slides -- Evaluation (71 slides), Roman Kern & Denis Helic, TU Graz.*
