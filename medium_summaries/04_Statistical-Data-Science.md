# 04 Statistical Data Science — Medium Summary

**Source:** KDDM1 Lecture by Roman Kern (v2.1.0, TU Graz)
**Focus:** Foundational statistical concepts for data mining — correlation, hypothesis testing, causality, and fundamental limitations of data processing.

---

## 1. Statistical Basics

### 1.1 Data Science vs. Statistics
- **Data mining view:** The dataset is treated as the complete representation of the phenomenon of interest.
- **Statistics view:** Data is a sample drawn from an underlying generative process that we truly care about. Probabilities cannot be computed from finite data — only estimated. The **Law of Large Numbers** ensures estimates converge to true values given enough data under favourable conditions.

### 1.2 The Knowledge Discovery Process
- Two pipelines are distinguished: (a) the **Data Generation Process** where Nature (patterns) produces data via collection, and (b) the **Knowledge Discovery Process** where a dataset undergoes pre-processing, feature extraction, and data mining to recover patterns.

### 1.3 Statistical Approach
- The statistical workflow assumes a **distribution** for Nature, then uses estimation from the dataset to derive patterns. Critically, insights are only valid if the assumed distribution matches reality. The choice of distribution is an assumption, not a given.

### 1.4 Data Mining Grand Checklist
A 12-point checklist for regression diagnostics: (1) linearity via scatter plots, (2) t-statistics and confidence intervals, (3) F-tests for coefficient subsets, (4) R-squared in context, (5) influential observations/outliers, (6) normality of residuals, (7) studentized residuals, (8) heteroscedasticity and Box-Cox transformations, (9) autocorrelation in time series, (10) multicollinearity and sign checks, (11) principal components, (12) missing values.

### 1.5 Types of Data Science Projects
- **Hypothesis-driven** (positivism): e.g., "is there a quality impairment if parameter X is changed?"
- **Data-driven** (constructivism): e.g., "what insights can be generated from the data?"
- **Simulation-driven:** using ML to simulate and optimise a process.

---

## 2. Correlation and Dependency

### 2.1 Basic Definitions
- **Statistical dependence:** X and Y are positively dependent if P(X,Y) > P(X)P(Y); negatively dependent if the inequality is reversed.
- **Correlation ratio:** eta^2_{Y|X} = 1 - sigma^2_{Y|X} / sigma^2_Y. Equals 0 if independent, equals rho^2 if linearly dependent.

### 2.2 Pearson's Correlation (PC)
- Formula: r_xy = sum((x_i - x_bar)(y_i - y_bar)) / sqrt(sum(x_i - x_bar)^2 * sum(y_i - y_bar)^2)
- Ranges [-1, +1]; PC = 0 does NOT imply independence (only linear independence). Independence implies PC = 0.
- Sensitive to outliers and assumes a linear relationship. Same PC value can arise from very different data shapes (Anscombe's quartet).

### 2.3 Non-Linear Correlation Measures
- **Rank correlation (Spearman's rho, Kendall's tau):** Variables are ranked; the rankings are compared. Detects monotonic relationships.
- **Maximal Correlation (mCor):** mCor(X,Y) = max_{f,g} Cor(f(X), g(Y)). Solved via the **Alternating Conditional Expectations (ACE)** algorithm. Caveats: estimation of conditional expectations is hard; overestimation can occur.
- **Distance Correlation (dCor):** Ranges [0,1]; dCor = 0 iff X and Y are independent. Computed via pairwise distance matrices followed by double centering. More robust for small samples than MIC.
- **Maximal Information Coefficient (MIC):** Based on mutual information binned over grids. Ranges [0,1]; symmetric; tends to 0 for independence and 1 for noiseless functional relations. Requires large sample sizes.

### 2.4 Key Insight: Correlation != Dependency
- A non-monotone function (e.g., f(x) = sin(lambda * pi/2 * x) for lambda > 1) can yield zero Pearson correlation while the variables are clearly dependent.

### 2.5 Related Concepts
- Partial correlation and confounders, similarity/distance measures, statistical significance tests, symbolic regression, copulas.

---

## 3. Hypothesis Testing

### 3.1 Core Framework
- **p-value:** p = P(X_S more extreme than observed x_S | H_0 true). Reject H_0 if p <= alpha (significance level).
- **Type I error:** rejecting H_0 when it is true (false discovery).
- **Type II error:** not rejecting H_0 when it is false.

### 3.2 Fisher's Exact Test
- For 2x2 contingency tables (two groups, binary feature). Directly computes a p-value via the hypergeometric distribution without asymptotic approximation. The chi-squared test is a cheaper approximation.

### 3.3 Permutation Test
- Randomly permutes the dataset and counts how often a condition is met. Conceptually straightforward but computationally expensive and formally imprecise.

### 3.4 Multiple Hypothesis Testing
- Testing many hypotheses at the same alpha level inflates false discoveries.
- **Family-Wise Error Rate (FWER):** controls the probability of at least one false discovery.
- **Bonferroni correction:** adjusts significance level to alpha/n (conservative).
- **p-Hacking:** manipulating experiments (repeating, increasing hypotheses, HARKing) to push p-values below the threshold.

### 3.5 Data Mining Without Expert Knowledge
- Different analysis teams can reach dramatically different conclusions from the same data (e.g., 69% found a significant effect, 31% did not), highlighting the need for pre-registered analyses.

---

## 4. Causality

### 4.1 Basics and Gold Standard
- Causality studies the impact of hypothetical actions or interventions.
- **Randomised control study:** randomly assign treatment/control groups, apply intervention, observe difference. Often infeasible due to ethics or cost.

### 4.2 Causal Graphs
- Nodes = variables; directed edges = causal relationships. Key structures: indirect causes (paths), forks (one cause, multiple effects), mediators (intermediate variables), colliders (multiple causes converging on one effect).

### 4.3 Confounders and Paradoxes
- **Confounders:** hidden causes that create a spurious apparent relationship between observed variables.
- **Simpson's paradox:** A trend that holds for every subpopulation can reverse when subpopulations are combined. Typically caused by an overlooked external factor.
- **Explaining away (collider bias):** Two independent variables can appear negatively correlated when both are causes of an observed variable, due to the sampling strategy.

---

## 5. Limitations

### 5.1 Data Processing Inequality
- The knowledge discovery pipeline is modelled as a Markov chain: X -> Y -> Z (e.g., preprocessing -> features -> mining).
- Information-theoretic consequence: I(X; Y) >= I(X; Z) and I(Y; Z) >= I(X; Z) — **information can only be lost**, never invented, at each processing step.
- Practical nuance: losing information can still improve algorithmic performance if the remaining signal is more accessible (e.g., parsing text removes noise but makes structure usable).

### 5.2 Central Limit Theorem (CLT)
- The sum of independent random variables with finite variance tends toward a normal distribution after proper normalisation. Example: the sample mean is approximately normally distributed even if the underlying variables are not.
- **Cauchy distribution:** Breaks the CLT because it has no finite variance. The sample mean of i.i.d. Cauchy variables is again Cauchy — the mean cannot be reliably estimated. The median, however, can be.
- **Robust statistics:** A subfield studying methods less affected by outliers or mild assumption violations. Examples: median, interquartile range, asymptotic breakdown point, winsorising (x' = min(Large, max(-Large, x))).

### 5.3 Learning Objectives
- Understand the distinction between correlation and dependency.
- Select appropriate correlation measures based on data characteristics (linearity, sample size, monotonicity).
- Interpret p-values correctly and avoid common pitfalls (multiple testing, p-hacking).
- Recognise when causal claims are justified and how confounders distort interpretation.
- Apply the Data Processing Inequality to set realistic expectations about information preservation in a pipeline.
- Identify when the CLT applies and when distributions like Cauchy require robust alternatives.
