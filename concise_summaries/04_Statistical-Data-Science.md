# 04 - Statistical Data Science

## Core Definitions and Concepts

- **Data Mining vs. Statistics**: Data mining treats the dataset as the complete representation of the world; statistics views data as coming from an underlying generative process that we truly care about.
- **Knowledge Discovery Process**: Nature (patterns) -> Data collection -> Dataset -> Pre-processing -> Features -> Data Mining -> Patterns.
- **Statistical Approach**: Make assumptions about Nature (e.g., a distribution) to rigorously derive insights; insights are only valid if the assumptions hold.
- **Types of Data Science Projects**: Hypothesis-driven (positivism), Data-driven (constructivism), and Simulation-driven.
- **Statistical Dependence**: X and Y are positively dependent if P(X,Y) > P(X)P(Y); negatively dependent if the inequality is reversed.
- **Correlation vs. Dependency**: Correlation does not equal dependency. Most correlation measures give values in [-1,+1] or [0,+1], but values cannot be directly compared across measures.
- **Causality**: Correlation does not imply causation. Preferred method is a randomised control study (treatment vs. control group), but this is not always feasible (ethical constraints).
- **Causal Graphs**: Nodes are variables, arrows point from cause to effect; concepts include indirect causes, forks, mediators, colliders, and confounders.
- **Simpson's Paradox**: A trend that holds across all subpopulations can reverse in the aggregate — often due to confounders.
- **Explaining Away (Collider Bias)**: Two independent variables can appear negatively correlated when both are causes of an observed variable, due to sampling strategy.
- **Data Processing Inequality**: Along the processing pipeline (modeled as a Markov chain X -> Y -> Z), information can only be lost: I(X;Y) >= I(X;Z) and I(Y;Z) >= I(X;Z).
- **Central Limit Theorem (CLT)**: The sum (normalised) of independent random variables with finite variance tends toward a normal distribution, even if the original variables are not normally distributed.
- **Robust Statistics**: Methods not (or less) affected by outliers or small violations of assumptions; e.g., median, interquartile range, asymptotic breakdown point.
- **p-Hacking**: Manipulating experiments or hypotheses to force p-values below the significance threshold (publication bias, HARKing).

## Key Algorithms and Techniques

- **Pearson's Correlation (PC)**: Linear correlation coefficient. `r = sum((x_i - x̄)(y_i - ȳ)) / sqrt(sum(x_i - x̄)^2 * sum(y_i - ȳ)^2)`. PC = 0 does NOT imply independence; independence implies PC = 0. Sensitive to outliers.
- **Spearman's Rank Correlation (ρ)**: Monotonic (non-linear) correlation based on ranks.
- **Kendall's Rank Correlation (τ)**: Rank-based correlation measuring ordinal association.
- **Maximal Correlation (mCor)**: `mCor(X,Y) = max Cor(f(X), g(Y))` over functions f,g. Solved via Alternating Conditional Expectations (ACE) algorithm. mCor = 0 iff X and Y are independent.
- **Distance Correlation (dCor)**: Computes pairwise distance matrices for X and Y, doubly centres them, then computes pairwise distances between the matrices. dCor = 0 implies independence. More robust with small sample sizes compared to MIC.
- **Maximal Information Coefficient (MIC)**: Bin-based mutual information approximation. `MIC = max I(X;Y) / log2(min(#bins_X, #bins_Y))`. Range [0,1], symmetric, not an estimate of mutual information itself.
- **Hypothesis Testing**: p-value = P(test statistic more extreme than observed | H0 true). Reject H0 if p <= alpha.
- **Type I Error (False Discovery)**: Reject H0 when H0 is true.
- **Type II Error**: Fail to reject H0 when H0 is false.
- **Fisher's Exact Test**: 2x2 contingency table test providing a direct p-value for group differences on a binary feature. Chi-squared test is a computationally cheaper approximation.
- **Permutation Test**: Randomly permute the dataset and count how often a condition is met; computationally expensive, formally imprecise.
- **Multiple Hypothesis Corrections**: Bonferroni correction (p* < alpha/n), Bonferroni-Holm procedure, LAMP. Use validation dataset splits.

## Important Formulas and Diagrams

- **Pearson correlation coefficient**: `ρ_{X,Y} = cov(X,Y) / (σ_X σ_Y)`; sample form: `r_{xy} = Σ(x_i - x̄)(y_i - ȳ) / sqrt(Σ(x_i - x̄)^2 Σ(y_i - ȳ)^2)`
- **Correlation ratio**: `η²_{Y|X} = 1 - σ²_{Y|X} / σ²_Y`; equals 0 if independent, equals ρ² if linearly dependent.
- **Maximal Correlation**: `mCor(X,Y) = max_{f,g} Cor(f(X), g(Y))`
- **MIC**: `MIC(X,Y) = max_{bins < B} I(X;Y) / log₂(min(#bins_X, #bins_Y))`
- **Bonferroni correction**: `p* < α/n`
- **Markov chain (data processing)**: `P(X,Y,Z) = P(Z|Y)P(Y|X)P(X)`; `I(X;Z|Y) = 0`; `I(X;Y) >= I(X;Z)`
- **Simpson's paradox**: `P(Y|X) < P(Y|¬X)` but `P(Y|X,Z=z) > P(Y|¬X,Z=z)` for all values of Z.
- **Data Generating Process diagram**: Nature -> Data collection -> Dataset.
- **Knowledge Discovery Process diagram**: Dataset -> Pre-processing -> Features -> Data Mining -> Patterns.
- **Statistical Approach diagram**: Nature -> Assumed Distribution -> Estimation (from Dataset) -> Patterns.
- **Fisher's exact test formula**: p = C(a+b, a) * C(c+d, c) / C(n, a+c)

## Learning Objectives

- Understand the distinction between data mining and statistical perspectives on data.
- Appreciate how assumptions (e.g., distributional forms) underpin statistical inference and why violating them invalidates insights.
- Learn to detect and interpret different types of correlation (linear, rank, maximal, distance, MIC) and understand that correlation does not imply causation.
- Apply hypothesis testing correctly: define null/alternative, compute p-values, understand Type I/II errors.
- Recognise and correct for multiple hypothesis testing (Bonferroni, Holm).
- Be aware of p-hacking, HARKing, and other statistical pitfalls.
- Understand causality frameworks: randomised control trials, causal graphs, confounders, collider bias, Simpson's paradox.
- Internalise the Data Processing Inequality: information can only be lost along a pipeline, not invented.
- Grasp the Central Limit Theorem, its assumptions (finite variance), and why it fails for distributions like Cauchy.
- Apply robust statistics (median, winsorising) when data violate normality assumptions.
- Use visual tools (hexabin plots, contour plots) for detecting non-linear relationships.
