# 06 Feature Engineering — Concise Summary

## Core Definitions & Concepts

- **Feature Engineering**: The act of injecting domain knowledge into a machine learning model by transforming raw data into a suitable representation.
- **Feature Set**: The collection of features used for a given task.
- **Feature Space**: The high-dimensional space spanned by the features and their value ranges.
- **Instance**: A single example (assignment of feature values).
- **Machine Learning Model**: The output of the learning process — a knowledge representation.
- **Data Processing Pipeline** (Markov chain X -> Y -> Z): Joint probability factors as P(X,Y,Z) = P(Z|Y) P(Y|X) P(X); past and future are conditionally independent given the present. Information can only be lost along the pipeline (Data Processing Inequality).
- **Structural Dependencies**: Between independent/dependent variables (required for prediction), between independent variables (redundancy / multicollinearity), and between instances (violates i.i.d. assumption).
- **Goals of Feature Engineering**: (1) Best prediction accuracy, or (2) an explainable model.
- **Process**: Remove unnecessary/redundant features, create new features (combine, transform, contextual, external), modify feature types, modify feature values.

## Information Theory

- **Kolmogorov Complexity**: Size of the smallest program that can produce the data — not computable, but a theoretical lower bound for compression.
- **Minimum Description Length (MDL)**: Among competing models, prefer the one that compresses the data best (ties to Occam's razor).
- **Entropy H(X)**: Measure of uncertainty in a discrete random variable; H(X) = -sum p(x) log2 p(x). High entropy = uniform distribution (hard to predict); low entropy = peaked distribution (easier to predict). Always non-negative; upper bound is log|X|.
- **Joint Entropy H(X,Y)**: Uncertainty of a pair of variables; H(X,Y) = H(X) + H(Y) if X and Y are independent.
- **Conditional Entropy H(Y|X)**: Remaining uncertainty in Y once X is known; H(Y|X) = H(X,Y) - H(X).
- **Information Gain IG(Y|X) = H(Y) - H(Y|X)**: Reduction in uncertainty of Y given X. Relative IG = IG(Y|X) / H(Y).
- **Mutual Information I(X;Y)**: How much information one variable contains about another; I(X;Y) = 0 iff X and Y are independent. Analogous to a generalised correlation measure.
- **Pointwise Mutual Information (PMI)**: pmi(x;y) = log2[p(x,y) / (p(x)p(y))]; can be normalised.
- **Kullback-Leibler Divergence D(p||q)**: Relative entropy between two probability distributions; always non-negative, zero iff p = q. I(X;Y) = D(p(x,y) || p(x)p(y)).

## Distance Functions

- **Purpose**: Many data mining algorithms rely on distance to estimate similarity and find nearest neighbours.
- **Numeric Distances**:
  - **Euclidean** (L2): sqrt(sum (xi - yi)^2) — default, but suffers from curse of dimensionality.
  - **Manhattan** (L1): sum |xi - yi| — city-block distance.
  - **Chebyshev** (Linf): max_i |xi - yi| — chessboard distance.
  - **Minkowski**: Generalisation with parameter p (p=1: Manhattan, p=2: Euclidean, p=inf: Chebyshev).
  - **Mahalanobis**: sqrt((x-y)^T Sigma^{-1} (x-y)) — accounts for covariance structure (whitening).
  - **Cosine Similarity**: (x . y) / (||x|| ||y||) — angle-based, better for high-dimensional data; converted to distance as 1 - sim_cosine.
- **Categorical Distances**:
  - **Jaccard Index**: |A n B| / |A u B| — ratio of intersection to union.
  - **Hamming Distance**: Count of differing positions in two equal-length lists/strings.
- **String Distances**:
  - **Levenshtein Distance**: Minimum number of character insertions, deletions, and (optionally) substitutions to transform one string into another.
- **Practical Considerations**: Euclidean is default but less suited for high dimensions; Cosine similarity excels there; normalisation/standardisation impacts all distance-based methods; correlations among features can skew distances.

## Feature Value Processing (Univariate)

- **Binarisation**: Threshold numeric values to boolean.
- **Discretisation**: Convert continuous to discrete (equal-size partitions or equal-interval partitions).
- **Transformation**: Scaling values and shifting the centre.
- **Normalisation**: Min-max scaling to [0,1]; prevents features with large ranges from dominating distance functions.
- **Standardisation**: Centre at mean, scale to units of standard deviation.
- **Feature Weighting**: Encode feature importance into the value (e.g., term weighting in text).

## Feature Engineering for Text Mining

- **N-grams**: Bigrams (word pairs), trigrams, general n-grams to capture sequence information lost with unigrams.
- **Skip N-grams**: Allow gaps between words.
- **Character N-grams**: Operate at character level (e.g., character trigrams).
- **External Sources**: Enrich features with external knowledge bases (e.g., WordNet hypernyms: adding "canine", "canid" for the word "fox").

## Feature Selection

- **Motivation**: Remove unnecessary/redundant features; combat curse of dimensionality; improve interpretability, generalisation (less overfitting), and efficiency.
- **Feature selection** is NP-hard (set-of-all-subsets); practical approaches are needed.
- **Heuristic Approaches**: Black/white lists; e.g., stop-word lists for text.
- **Unsupervised Ranked Selection**: Score features (e.g., term frequency in a reference corpus) and keep top k%.
- **Supervised Approaches**:
  - **Filter**: Compute a discriminative measure (e.g., Information Gain) per feature, rank, select top-k. Fast but ignores redundancy and feature interactions.
  - **Wrapper**: Search the space of feature subsets, evaluate each with a learner (forward selection: start empty, add best; backward elimination: start full, remove worst). Computationally intensive but accounts for feature combinations.
- **Further Methods**: Correlation-based removal, PCA, Interaction Information (from information theory for non-trivial variable relationships).
- **Alternative to selection**: Regularisation (e.g., L1/L2) as a built-in complexity reduction within the model.

## Important Diagrams Referenced

- Entropy/Information diagram showing relationships among H(X), H(Y), H(X,Y), H(X|Y), H(Y|X), and I(X;Y).
- WordNet hierarchy for "fox" showing hypernyms (e.g., canine, canid, carnivore, mammal).
- Example frequency table for Wikipedia terms (top word "the" occurs in 88% of instances).

## Learning Objectives

- Understand how to measure information content (entropy, mutual information, KL divergence).
- Select and apply appropriate distance functions for numeric, categorical, and string data.
- Apply the data processing inequality to reason about information loss in pipelines.
- Transform and normalise features to meet algorithm assumptions.
- Engineer textual features via n-grams and external knowledge integration.
- Perform feature selection using heuristics, filter methods (e.g., Information Gain), and wrapper methods (forward selection / backward elimination).
- Balance model complexity against predictive power (MDL principle).
