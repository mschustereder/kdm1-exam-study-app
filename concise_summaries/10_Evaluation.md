# 10_Evaluation — Concise Summary

## Core Definitions & Concepts

- **Evaluation** is required to estimate how well a model will perform on unseen/real-world data; the goal is to understand generalisation performance and avoid spurious results.
- **Key assumption:** the underlying generative process between training and unseen data does not change (i.i.d. assumption).
- **Accuracy (A):** fraction of correct predictions — `A = (tp + tn) / (tp + fp + fn + tn)`.
- **Accuracy paradox:** with skewed class distributions, a trivial majority-class classifier can achieve high accuracy while being useless (e.g., cancer detection).
- **Contingency table** (confusion matrix): rows = predicted class, columns = real class; cells = tp, fp, fn, tn.
- **Prevalence:** P = number of real positive instances (`tp + fn`); N = number of real negative instances (`tn + fp`).

## Key Metrics & Techniques

### Classifier Evaluation (Binary)

- **Precision (Pr):** `tp / (tp + fp)` — how many predicted positives are truly positive.
- **Recall (R) / Sensitivity / TPR:** `tp / (tp + fn)` — fraction of actual positives captured.
- **Precision-Recall trade-off:** predicting more 1s increases recall but decreases precision; predicting fewer 1s increases precision but decreases recall.
- **F1 Score:** harmonic mean of precision and recall — `F1 = 2 * (Pr * R) / (Pr + R)`; always lies between the two, biased toward the lower value.
- **F-beta Score:** `Fβ = (1 + β²) * (Pr * R) / (β² * Pr + R)` — β weights recall importance (F2 emphasises recall for outlier detection).
- **Matthews Correlation Coefficient (MCC):** `(tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))` — overall correlation between real and predicted classes; recommended as a default single metric.
- **ROC Curve:** plots true positive rate vs. false positive rate for continuous classifier outputs (e.g., probabilities); originally from WWII signal detection.
- **AUC (Area Under ROC Curve):** single scalar summarising the ROC curve, used for model selection.

### Multi-Class Settings

- **Micro F1:** global average (equals accuracy); biased toward majority classes.
- **Macro F1:** unweighted average of per-class F1; ignores class sizes.
- **Weighted F1:** per-class F1 weighted by class support; balances class size and performance.

### Regression Evaluation

- **RMSE (Root Mean Squared Error):** sensitive to outliers (larger deviations penalised more).
- **MAE (Mean Absolute Error):** less sensitive to outliers, same units as target.
- **R² (Coefficient of Determination):** fraction of variance in the target explained by the model.

### Baselines

- **Lower bound:** trivial/zero-utility threshold (e.g., constant classifier predicting majority class, random guessing).
- **State of the art (SotA):** currently best known solution.
- **Human performance:** average human vs. expert.
- **Upper bound:** theoretical limit.
- Baselines must reflect task hardness; comparing only against weak baselines is dishonest.

## Cluster Evaluation

- **Human evaluation:** subjective, labour-intensive, hard to reproduce.
- **Algorithmic evaluation:**
  - **External validity:** compare clustering against a reference (e.g., ground-truth classes).
    - **Purity:** `max_j(|Ci ∩ Cj*|) / |Ci|` — ratio of dominant class in a cluster.
    - **Rand Index:** `(TP + TN) / (TP + TN + FP + FN)` based on pairwise instance comparisons.
    - **V-Measure:** weighted harmonic mean of homogeneity (how much cluster tells us about class) and completeness (how much class tells us about cluster), derived from conditional entropy.
  - **Internal validity:** analyse intrinsic cluster characteristics.
    - **Silhouette Coefficient:** `s(i) = (b(i) - a(i)) / max{a(i), b(i)}` where a(i) = mean intra-cluster distance, b(i) = mean distance to nearest other cluster; values near +1 indicate well-separated clusters.
    - **Edge correlation:** correlation between the similarity matrix and the co-occurrence (same-cluster) matrix.
  - **Relative validity:** sensitivity analysis during clustering generation.

## Dataset Handling & Validation

- **Overestimation risk:** training and testing on the same data yields overly optimistic results (the model may have memorised).
- **Dataset splitting:** split randomly into training, optional validation (for hyper-parameter tuning), and test sets (used only once). Typical split: 80/20.
- **Stratification:** preserve class proportions across splits — crucial for small or imbalanced datasets.
- **k-Fold Cross-Validation:** split into k equally-sized folds; iterate using one fold for testing and rest for training; report mean and variance. k is typically 3–10.
- **Leave-One-Out (LOO):** k = number of data points.
- **Test data may only be used once** — reusing it for improvements effectively turns it into training data.

## A/B Testing

- **Setup:** split users into groups A and B, each exposed to a different model/setting; record differences in user behaviour as implicit feedback.
- **Theoretical basis:** statistical hypothesis test with null hypothesis (no difference); p-value indicates significance.
- **Hill climbing for A/B testing:** continuously introduce new variations and repeat A/B tests to adapt to changing user behaviour.

## Diagrams Referenced

- Contingency table (binary confusion matrix)
- Precision/Recall graph (trade-off curve)
- ROC curve (TPR vs. FPR)
- Silhouette plot (visual quality of clustering)
- Learning curves from ablation tests (performance vs. training set size)

## Learning Objectives

- Understand why evaluation is critical and what assumptions underlie it.
- Master binary and multi-class classification metrics (accuracy, precision, recall, F1/Fβ, MCC, ROC/AUC).
- Select appropriate regression evaluation measures (RMSE, MAE, R²).
- Design and interpret baselines (lower bound, SotA, human, upper bound).
- Evaluate clustering solutions via external measures (purity, Rand Index, V-Measure) and internal measures (Silhouette, edge correlation).
- Apply correct dataset handling strategies (splitting, stratification, cross-validation) to avoid overestimation.
- Set up and interpret A/B tests for live-system evaluation.
- Conduct ablation tests and error analysis to understand model behaviour and guide improvements.
