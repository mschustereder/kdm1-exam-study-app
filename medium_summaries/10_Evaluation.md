# 10 — Evaluation (KDDM1)

## 1. Motivation & Goal

- **Why evaluate?** After data mining we need to know if discovered patterns generalise to unseen data — i.e., whether results are spurious or real.
- **Goal:** Understand evaluation scenarios and their ability to estimate real-world performance.

---

## 2. Classifier Evaluation — Key Metrics

### Contingency Table & Terminology

- Rows = predicted class, columns = real class. Cells: **TP** (true positives), **FP** (false positives), **FN** (false negatives), **TN** (true negatives).
- **P = TP + FN** (real positives), **N = TN + FP** (real negatives), total **n = P + N**.

### Core Metrics

- **Accuracy A = (TP+TN) / (TP+FP+FN+TN).** Overall fraction correct. Suffers from the **accuracy paradox**: with skewed classes (e.g. 0.8% cancer), a trivial majority-class predictor achieves 99.2% accuracy yet detects no cancers — entirely useless.
- **Recall (sensitivity, TPR) R = TP/P = TP/(TP+FN).** Fraction of actual positives correctly identified. High recall means few missed positives.
- **Precision Pr = TP/(TP+FP).** Fraction of predicted positives that are truly positive. High precision means few false alarms.

### Precision-Recall Trade-off

- Predicting more positives increases recall but also adds FPs (precision drops). Predicting fewer positives raises precision but misses positives (recall drops). The **PR curve** visualises this and is characteristic of each classifier.
- **F1 = 2 · Pr · R / (Pr + R)** — harmonic mean of Pr and R; a single score balancing both. Always lies between them, tending toward the lower value. Does not consider TNs.
- **Fβ = (1+β²) · (Pr·R) / (β²·Pr + R)** — β weights recall importance. β=1 → F1 (equal weight), β>1 favours recall (e.g. F2 for outlier detection), β<1 favours precision (e.g. web search).

### Additional Measures

- **MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)).** Measures correlation between real and predicted classes using all four table cells. Considered the best single balanced metric for binary classification.
- **ROC Curve:** Plots TPR (Recall) vs. FPR across all thresholds. Only applicable when the classifier outputs a continuous score/probability. **AUC** summarises the entire curve as one number, useful for model selection (e.g. in outlier detection).

### Multi-Class Extensions

- **Micro F1:** Global average counting all TP/FP/FN across classes — biased toward majority classes, equivalent to accuracy.
- **Macro F1:** Unweighted mean of per-class F1 — ignores class sizes, requires balanced classes.
- **Weighted F1:** Per-class F1 weighted by class support — balances class size and individual performance.

### Regression Metrics

- **RMSE:** Root mean squared error — sensitive to outliers; penalises large deviations heavily.
- **MAE:** Mean absolute error — less sensitive to outliers; interpretable (same units as target).
- **R²:** Coefficient of determination — proportion of target variance explained by the model.

---

## 3. Baselines

- Provide a reference frame to judge whether results are "good" relative to task hardness. Key types:
  - **Lower bound:** Threshold of no utility.
  - **Trivial solution:** Easily achieved performance (e.g. constant majority-class predictor).
  - **State of the art (SotA):** Currently best-known solution.
  - **Human performance:** Average human or expert level.
  - **Upper bound:** Theoretical limit.
- **Constant classifier:** Always predicts majority class; accuracy = prevalence of that class.
- **Random guessing:** Randomly predicts according to class distribution (e.g. 50% for balanced data).
- **Best practice:** Always include multiple SotA baselines; comparing only against weak baselines is dishonest. Non-performance factors (runtime, memory) may justify a slightly weaker model.

---

## 4. Cluster Evaluation

- "The most difficult and frustrating part of cluster analysis" (Jain & Dubes, 1990).

### Evaluation Approaches

- **Human evaluation:** Domain expert inspects clusters — subjective, hard to reproduce, costly.
- **External validity:** Compare clustering to ground-truth class labels (when available).
- **Internal validity:** Analyse intrinsic structure — cohesion (similar within cluster) vs. separation (dissimilar between clusters).
- **Relative validity:** Sensitivity analysis across clustering parameters.

### External Measures

- **Purity:** (1/|Ci|) · max_j |Ci ∩ Cj*| — fraction of the dominant class in a cluster; favours smaller clusters.
- **Rand Index:** (TP+TN)/(TP+FP+FN+TN) from pairwise comparisons of whether instances share a cluster and a class.
- **V-Measure:** Harmonic mean of **homogeneity** (how much clustering reveals about classes) and **completeness** (how much classes reveal about clustering), computed via conditional entropy.

### Internal Measures

- **Silhouette Coefficient s(i) = (b(i)−a(i)) / max{a(i), b(i)}**, where a(i) = mean distance to other points in the same cluster, b(i) = mean distance to points in the nearest neighbour cluster. Ranges [−1, 1]; higher = better. Suitable for k-Means but not density-based clustering.
- **Edge correlation:** Correlation between the pairwise similarity matrix and the cluster co-occurrence matrix.

---

## 5. Dataset Handling for Generalisation

- **Core assumption:** The generative process is identical between training and unseen data (i.i.d. assumption — same distribution, no covariate shift).
- **Overestimation:** Evaluating on training data overestimates performance because the model may have overfit.
- **Splitting:** Randomly partition into training, validation (optional — for hyperparameter tuning), and test sets. Common split: 80/20.
- **Golden rule:** Test data must be used **only once** — reusing it for model improvements turns it into training data and invalidates the evaluation.
- **Stratification:** Preserves class proportions in each split; important for small or imbalanced datasets.
- **k-fold Cross-Validation:** Split data into k folds. Iteratively train on k−1 folds, test on the remaining one. Report average ± variance across folds (k=3–10 typical). When k = n, it is **Leave-One-Out**.

---

## 6. A/B Testing

- Users interact with a production system (e.g. recommender); their behaviour serves as implicit feedback.
- Users are randomly split into groups A and B, each experiencing a different model/configuration. No sampling bias allowed.
- Differences in measured behaviour estimate relative performance. This is a statistical hypothesis test: **null hypothesis** = no difference; a significant p-value allows rejecting the null.
- **Hill climbing:** Repeatedly introduce new variations and run A/B tests to continuously improve the system and adapt to changing user behaviour.

---

## 7. Summary — Best Practices

- **Ablation tests:** Remove one model component at a time to estimate its contribution. Can also assess the effect of dataset size by progressively removing training data points.
- **Error analysis:** Manually inspect misclassified instances to find failure patterns and root causes — directly guides improvement.
- **Conclusions:** Evaluation is essential for estimating expected performance. Assumptions (often stronger than the model's own assumptions) are required. Test data is strictly for comparison, not improvement. Ablation, error analysis, and replication studies constitute good scientific practice.
