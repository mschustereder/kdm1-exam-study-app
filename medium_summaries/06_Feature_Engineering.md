# 06 — Feature Engineering (Medium Summary)

## 1. Core Definitions & Motivation

- **Feature Engineering**: The act of injecting domain knowledge into a machine learning model by transforming raw data into a representation that algorithms can exploit effectively.
- **Feature Set**: The collection of all features used for a task.
- **Feature Space**: The high-dimensional space spanned by the features and their value ranges.
- **Instance**: A single assignment of feature values (one example/data point).
- **Key Motivation**: The quality of knowledge discovery depends directly on how well features encode the underlying phenomenon. A poorly engineered feature space leads to poor results, regardless of the algorithm used.

## 2. Information Theory — Measuring Information in Data

- **Kolmogorov Complexity**: The size of the smallest program that can reproduce a dataset. It is the theoretical lower bound for compression but is **not computable** in practice.
- **Minimum Description Length (MDL)**: A practical principle: among competing models, prefer the one that compresses the data best (ties to Occam's razor).
- **Entropy** `H(X) = -sum p(x) log2 p(x)`: Measures uncertainty (information content) in bits. High entropy means a uniform distribution (hard to predict); low entropy means peaks/valleys (easier to predict).
- **Joint Entropy** `H(X,Y)`: Uncertainty of a pair of variables. If X and Y are independent, `H(X,Y) = H(X) + H(Y)`.
- **Conditional Entropy** `H(Y|X)`: Remaining uncertainty about Y once X is known. `H(Y|X) = H(X,Y) - H(X)`.
- **Information Gain** `IG(Y|X) = H(Y) - H(Y|X)`: How much uncertainty of Y is reduced by knowing X. A key criterion for feature selection and decision tree splitting.
- **Mutual Information** `I(X;Y) = D(p(x,y) || p(x)p(y))`: A general correlation measure that captures any form of dependency (not just linear). Zero iff X and Y are independent.
- **Pointwise Mutual Information (PMI)**: `pmi(x;y) = log2( p(x,y) / (p(x)p(y)) )` — measures the association of a specific pair of events.
- **Kullback-Leibler Divergence** `D(p||q) = sum p(x) log2(p(x)/q(x))`: A non-symmetric measure of how one probability distribution diverges from another. Always non-negative; zero iff p = q.
- **Data Processing Inequality**: In a Markov chain `X -> Y -> Z`, we have `I(X;Y) >= I(X;Z)`. Along any processing pipeline, information can only be lost, never gained.

## 3. Distance Functions — Measuring Similarity Between Points

- **Numeric Distances**:
  - **Minkowski distance** generalises three common metrics: Manhattan (p=1), Euclidean (p=2), Chebyshev (p=infinity).
  - **Mahalanobis distance** accounts for the covariance structure of the data, effectively measuring distance in terms of density (whitening makes it equivalent to Euclidean distance).
  - **Cosine similarity** measures the angle between vectors, ignoring magnitude. Well suited for high-dimensional sparse data (e.g., text). Converted to a distance via `1 - sim`.
- **Categorical Distances**:
  - **Jaccard index** `|A intersect B| / |A union B|` — ratio of shared to total features between two sets.
  - **Hamming distance** counts positional differences between equal-length strings/lists.
- **String Distances**:
  - **Levenshtein (edit) distance** counts insertions, deletions, and (optionally) substitutions needed to transform one string into another.
- **Key Insight**: All distance functions should reflect the *semantic* similarity between data points. Feature scaling and correlation can heavily distort distance calculations.

## 4. Feature Engineering — Process and Goals

- **Two Main Tasks**: (1) Understand task properties and how they interact with model strengths/limitations; (2) Experimental work to test what actually works.
- **Process**: Remove unnecessary/redundant features, create new ones, combine existing ones, transform values, integrate external sources, modify feature types.
- **Two Competing Goals**: Maximising prediction accuracy vs. building an explainable model (often require different trade-offs).
- **Fulfilling Assumptions**: Flatten relationships to satisfy the i.i.d. assumption; create aggregate features (e.g., averages) to simplify complex dependencies; encode sequential information since features are usually treated as a set.

## 5. Feature Value Processing — Univariate Transformations

- **Binarisation**: Thresholding a numeric feature to produce a boolean (e.g., age > 18 -> 1/0).
- **Discretisation**: Converting continuous values into discrete bins — choice of equal-width vs. equal-frequency bins matters.
- **Normalisation**: Prevents features with large ranges (e.g., -1000..+1000) from dominating distance functions. Two common methods:
  - **Min-max scaling** to [0,1].
  - **Standardisation** (z-score): centre at mean, scale to unit variance.
- **Feature Weighting**: Encoding a feature's importance into its value (e.g., term weighting in text — down-weighting grammatical words, up-weighting semantic words).
- **Note**: Tree-based methods are largely unaffected by normalisation; distance-based methods (kNN, clustering, SVM) are highly sensitive.

## 6. Feature Engineering for Text Mining

- **N-grams**: Capture sequence information lost with single-word (unigram) features. Bigrams (`brown_fox`), character n-grams (`the`, `qui`, `uic`), and skip-grams (n-grams with gaps) are common extensions.
- **External Knowledge Integration**: Enrich features using resources like **WordNet** (e.g., adding hypernyms — "fox" gains the feature "canine"). This injects semantic generalisation not present in the raw text.

## 7. Feature Selection — Reducing the Feature Space

- **Why reduce?** Fewer features improve interpretability, reduce overfitting risk, and increase efficiency. The curse of dimensionality makes distances less expressive in high dimensions.
- **Heuristic Approaches**: Blacklists/whitelists (e.g., stop-word lists for text). Simple and effective but can miss context-dependent semantics.
- **Unsupervised Selection**: Rank features by corpus statistics (e.g., term frequency in a reference corpus), keep top-k percent.
- **Supervised — Filter Approaches**: Compute a score for each feature independently (e.g., Information Gain), rank, and select top-k. Fast but ignores feature interactions and redundancy.
- **Supervised — Wrapper Approaches**: Search the subset space iteratively, evaluating each candidate set by training a model:
  - **Forward selection**: start empty, add the feature that improves performance most; stop when no improvement.
  - **Backward elimination**: start with all features, remove the least useful one; repeat.
  - Advantage: captures feature combinations. Disadvantage: computationally expensive, depends on the chosen learner.

## 8. Learning Objectives (Conclusions)

- Understand that **feature engineering = injecting knowledge** into the model.
- Be able to compute and interpret **entropy, information gain, and mutual information** as measures of feature utility.
- Select appropriate **distance functions** based on feature type (numeric, categorical, string, high-dimensional).
- Apply **normalisation/standardisation** correctly and know which algorithms require it.
- Differentiate between **filter vs. wrapper** feature selection and know when each is appropriate.
- Be aware that **information can only be lost** along the processing pipeline — make critical feature decisions early.
