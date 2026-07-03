# 06 — Feature Engineering — Full Summary

**Authors:** Roman Kern & Denis Helic (TU Graz)
**Version:** 2.1.1 | **Course:** KDDM1

---

## Motivation & Goal

The success of knowledge discovery depends on features encoding the phenomenon for ML to extract patterns. Core goal: understand interaction between **feature space**, **distance functions**, and **learning algorithms**.

---

## 1. Recap — How Did We Get Here?

**1.1 Prediction Tasks** — Classification (predict discrete labels; ordinal has natural order) and Regression (predict continuous values from independent variables).

**1.2 Feature Types** — Numeric: Continuous (interval/ratio) or Discrete (countable). Categorical: Nominal (no ordering) or Ordinal (natural order, no fixed interval).

**1.3 Types of Dependencies** — By form: linear vs. non-linear. By scope: full vs. partial. By reason: cause/effect vs. confounders.

**1.4 Structural Dependencies** — 1) Independent ↔ dependent (required for prediction; otherwise no signal). 2) Among independent variables (multicollinearity skews distances and model weights). 3) Between instances (violates i.i.d. assumption most ML relies on).

**1.5 Data Processing as Markov Chain** — Pipeline `X → Y → Z`, joint probability factors as `P(X,Y,Z) = P(Z|Y)·P(Y|X)·P(X)`. X and Z are conditionally independent given Y. **Data Processing Inequality:** `I(X;Y) ≥ I(X;Z)` — information can only be **lost**, never gained. Once discarded, no later step can recover it. Applies to neural network layers.

---

## 2. Information Theory

**2.1 Motivation** — Measure information in data: compare datasets, identify dependencies, guide feature selection.

**2.2 Kolmogorov Complexity** — Size (bits) of smallest program producing an object. Theoretical ultimate compression bound. **Not computable** in general — a theoretical construct.

**2.3 Minimum Description Length (MDL)** — Among models of dataset X, select smallest model (best compression). Practical proxy for Kolmogorov complexity. Directly relates to **Occam's Razor**: simpler explanations (models) are preferable.

**2.4 Entropy** — For discrete RV X with pmf p(x):
```
H(X) = - Σ p(x) · log₂ p(x)
```
Measured in **bits**; lower bound on average yes/no questions to guess state. High entropy → uniform, unpredictable; low → skewed, predictable. `H(X) ≤ log₂|𝒳|`, equality only for uniform distribution. **Example:** X={A,B,C,D} uniform → H=2 bits. Skewed {A:½, B:¼, C:⅛, D:⅛} → H=1.75 bits (variable-length coding: A=0, B=10, C=110, D=111).

**2.5 Joint Entropy** — 
```
H(X,Y) = - Σₓ Σᵧ p(x,y) · log₂ p(x,y)
```
If X,Y independent: `H(X,Y) = H(X) + H(Y)`. Generalizable to >2 variables.

**2.6 Conditional Entropy** — 
```
H(Y|X) = - Σₓ Σᵧ p(x,y) · log₂ p(y|x)
```
Uncertainty about Y remaining when X known. `H(Y|X)=H(X,Y)−H(X)`. **Specific Conditional Entropy:** `H(Y|X=x)` — entropy of Y given specific value x. If zero, that value of X completely determines Y. `H(Y|X) = Σ p(x)·H(Y|X=x)`.

**2.7 Information Gain (IG)** — 
```
IG(Y|X) = H(Y) − H(Y|X)
```
Uncertainty reduction about Y given X. Used in decision trees (split selection) and feature selection. Interpretation: bits saved transmitting Y if both sender/receiver know X. **Relative IG:** `RIG = IG/H(Y) ∈ [0,1]`.

**2.8 Mutual Information (MI)** — 
```
I(X;Y) = Σₓ Σᵧ p(x,y) · log₂(p(x,y) / p(x)p(y))
```
- `I=0` iff X ⟂ Y; symmetric; captures any dependency (not just linear, unlike Pearson correlation).
- `I(X;Y) = H(X)−H(X|Y) = H(Y)−H(Y|X)`.
- **Conditional MI:** `I(X;Y|Z) = H(X|Z)−H(X|Y,Z)`.
- **Connection:** `I(X;Y) = D(p(x,y)‖p(x)p(y))` — KL divergence between joint and product of marginals.

**2.9 Pointwise MI (PMI)** — 
```
pmi(x;y) = log₂(p(x,y) / p(x)p(y))
```
How much more/less likely co-occurrence is vs. independence. For binary variables in 2×2 contingency table, PMI reflects specific cell. **Normalized PMI:** `pmi/−log₂ p(x,y) ∈ [−1,+1]`.

**2.10 KL Divergence** — 
```
D(p‖q) = Σ p(x) · log₂(p(x)/q(x))
```
Measures how p diverges from q. Non-negative; zero iff p=q. **Not symmetric** (hence "divergence" not "distance"). Conventions: `0·log₂(0/q)=0`, `p·log₂(p/0)=∞`.

**2.11 Markov Chains & Data Processing Inequality** — X→Y→Z iff X ⟂ Z | Y. `p(x,y,z) = p(x)·p(y|x)·p(z|y)`. **Data Processing Inequality:** `I(X;Y) ≥ I(X;Z)` — information only decreases along any processing pipeline. Applies to all chained transformations.

**2.12 Relation to Data** — Entropy quantities defined on probability distributions. On finite random samples they must be **estimated** (plug-in estimator, shrinkage). Results may not be exact.

---

## 3. Distance Functions

**3.1 Motivation** — Many algorithms use distance to measure similarity. Assumption: features are dimensions in vector space.

**3.2 Distances by Feature Type:**
| Type | Example Functions |
|---|---|
| Numeric | Euclidean, Manhattan, Chebyshev, Minkowski, Mahalanobis, Cosine |
| Categorical | Jaccard, Hamming, Dice, Overlap, Tanimoto |
| String | Levenshtein (edit distance) |

**3.3 Numeric Distances:**
- **Euclidean (L₂):** `√(Σ (xᵢ−yᵢ)²)` — most common default.
- **Manhattan (L₁):** `Σ |xᵢ−yᵢ|`.
- **Chebyshev (L∞):** `maxᵢ |xᵢ−yᵢ|` (e.g., king moves on chessboard).
- **Minkowski:** `(Σ |xᵢ−yᵢ|ᵖ)^{1/p}`; p=1→L₁, p=2→L₂, p=∞→L∞.
- **Mahalanobis:** `√((x−y)ᵀΣ⁻¹(x−y))` — accounts for covariance structure. **Whitening** transforms space so Euclidean = Mahalanobis. Points in sparse regions are farther apart than in dense regions even if raw Euclidean distance is identical.
- **Cosine similarity:** `(x·y)/(‖x‖·‖y‖)` — angle between vectors, ignores magnitude. For normalized vectors equals dot product. Since it is a similarity, convert to distance as `1−sim`.

**3.4 Practical Considerations** — Euclidean is default but poor in high dimensions (curse: distances become less discriminative). Cosine better for sparse high-dim data (text) as it focuses on direction. Mahalanobis needs known/estimated distribution; powerful when covariance structure matters.

**3.5 Categorical Distances** — **Jaccard:** `|A∩B|/|A∪B|` (set overlap, ranges 0 to 1). **Hamming:** count differing positions in equal-length sequences (e.g., binary feature vectors).

**3.6 String Distances** — **Levenshtein (edit distance):** minimum add/remove/replace operations to transform one string to another. Replace counted as one operation.

**3.7 Key Reflection** — Some distances are proper **metrics** (non-negativity, symmetry, triangle inequality, identity of indiscernibles). Distances should reflect **semantic similarity** — semantically similar points should be close. **Problems:** different feature ranges let larger values dominate; **correlated features** share information and collectively distort distance (measuring same phenomenon via multiple features inflates its contribution).

---

## 4. Feature Engineering

**4.1 Definition** — Injecting domain knowledge by transforming raw data into better ML representations. Creative, exploratory, experimental process. No universal definition.

**4.2 Core Operations** — **Remove** unnecessary/redundant features. **Create** features: combine (ratios, products), transform (log for skewed distributions), use context (time of day, location), integrate external sources (knowledge bases, gazetteers). **Modify** types (binary↔numeric, numeric→categorical via discretization). **Modify** values (scaling, normalization, binarization).

**4.3 Goals** — 1) Best prediction accuracy (may sacrifice interpretability). 2) Explainable model (simpler, interpretable even if slightly lower accuracy).

**4.4 Terminology** — **Feature Set** (collection of all features for a task), **Feature Space** (high-dimensional space of possible value combinations), **Instance** (one data point — single assignment of feature values).

**4.5 Addressing Assumptions** — **i.i.d. violations:** flatten dependencies via aggregate/compound features (denormalization: averages, sums, counts, more sophisticated statistics). **Sequence info:** encode via sliding windows, lag features, differences — features are treated as a set, so order must be explicitly encoded.

---

## 5. Feature Value Processing (Univariate)

**5.1 Binarization** — Continuous → boolean (0/1) via threshold. Required when algorithm accepts only boolean inputs.

**5.2 Discretization** — Continuous → discrete categories. Design choices: equal-frequency (same count per bin) vs. equal-interval (same range per bin). Includes one-hot encoding of nominal categorical features.

**5.3 Scaling** — Adjust range and distribution. Large-range features dominate distance calculations. Most ML methods need normalization. Tree-based methods are **scale-invariant** (split on individual features and unaffected by scale).

**5.4 Normalization** — **Min-Max:** `x' = (x−min)/(max−min)` → [0,1] or [−1,1]. **Z-score (Standardization):** `x' = (x−μ)/σ` → centered at 0, unit variance. Result represents "number of standard deviations from mean."

**5.5 Feature Weighting** — Encode importance directly into values (domain prior about which features carry more signal). Example from text mining: term weighting differentiates grammatical words ("the", "and") from semantic words ("algorithm", "data").

---

## 6. Feature Engineering for Text Mining

**6.1 n-grams** — **Unigrams** (single words) lose word-order information. **Bigrams** (adjacent word pairs) capture local sequence context (e.g., "not good" vs. "good not"). **n-grams** extend to n-word sequences; **skip n-grams** allow gaps for non-contiguous patterns.

**6.2 Character n-grams** — n-grams at character level. Capture sub-word patterns, morphology, handle out-of-vocabulary words.

**6.3 External Sources** — Knowledge bases (e.g., WordNet: hypernyms, hyponyms, synonyms, meronyms) enrich representation beyond surface forms, improving generalization.

---

## 7. Feature Selection

**7.1 Motivation** — Remove unnecessary features (no target dependency, contribute noise). Remove redundant features (multicollinearity skews distances, dilutes signal). **Curse of dimensionality:** distances lose expressiveness, sample density decreases, overfitting risk grows.

**7.2 Benefits** — Better interpretability (simpler models are more explainable). Better generalization (lower overfitting risk). More efficient (faster training and prediction).

**7.3 Strategies** — 1) **Feature selection** — choose subset of original features. 2) **Regularization** — modify learning objective to penalize model complexity (L1/Lasso zeroes weights, L2/Ridge shrinks weights).

**7.4 The Challenge** — Optimal subset selection is NP-hard (set-of-all-subsets problem). Practical approaches rely on approximations and heuristics: **unsupervised** (data statistics, no labels) or **supervised** (leveraging labeled training data).

**7.5 Black Lists & White Lists** — **Black list:** features to exclude (e.g., stop words: "the", "you", "can"). **White list:** features to include exclusively (all others discarded). Trade-off: simple and effective, but stop words can carry meaning (band names, homonyms like "can" as verb vs. noun).

**7.6 Unsupervised Selection (Ranked)** — Score features without labels; select top-k%. Example: rank words by reference corpus frequency — very frequent (e.g., "the" in ~88% of Wikipedia docs) and very rare words are often less discriminative.

**7.7 Filter Approaches** — Score each feature independently (e.g., IG, MI) without training a model; rank and select top-k. **Limitations:** misses **redundant** features (correlated features score similarly but add no independent value) and **dependent** features (important only in combination, e.g., XOR problem).

**7.8 IG as Filter Example** — Compute IG per feature, rank, select top-k. Contact lens dataset: `tear-prod-rate` (0.549) > `astigmatism` (0.377) > `spectacle-prescrip` (0.040) > `age` (0.039).

**7.9 Wrapper Approaches** — Search subsets by training/evaluating a model for each candidate. **Forward Selection:** start empty, iteratively add feature yielding best improvement. **Backward Elimination:** start with full set, iteratively remove feature causing least degradation. **Pro:** captures feature interactions. **Con:** computationally expensive (model trained for each subset).

**7.10 Further Methods** — **Statistical:** correlation-based redundancy removal. **PCA:** dimensionality reduction via linear combinations (principal components), creating new uncorrelated features. **Information-theoretic:** MI (captures any dependency); **Interaction Information** for three-way relationships (e.g., effect of female candidate on donation willingness differs depending on supporter gender).

**7.11 Regularization** — **L1 (Lasso):** penalizes absolute weights → implicit feature selection (some weights become exactly zero). **L2 (Ridge):** penalizes squared weights → shrinks weights but does not zero them.

---

## 8. Conclusions
1. Goal: **small, interpretable models** that correctly represent data.
2. **Distance functions** should reflect semantic similarity; feature dependencies complicate this.
3. **Feature engineering** changes data representation to encode importance and meet algorithm assumptions.
4. **Feature selection** reduces dimensionality, improving interpretability, generalization, and efficiency.
5. **Information theory** provides rigorous foundations for measuring information and guiding feature decisions.

---

## 9. Further Resources
- IG tutorial: http://www.autonlab.org/tutorials/infogain11.pdf
- Info theory notes: http://www.icg.isy.liu.se/courses/infotheory/lect1.pdf
- Feature eng. slides (Princeton): http://www.cs.princeton.edu/courses/archive/spring10/cos424/slides/18-feat.pdf
- Feature eng. for NLP: http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/feature_engineering.pdf
- Web mining features: http://www.ke.tu-darmstadt.de/lehre/archiv/ss06/web-mining/wm-features.pdf

---

*End of Summary — KDDM1 Feature Engineering (v2.1.1), Roman Kern & Denis Helic, TU Graz*
