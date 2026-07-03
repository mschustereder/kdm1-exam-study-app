# 04 - Statistical Data Science

**Course:** KDDM1 (Knowledge Discovery and Data Mining 1)  
**Author:** Roman Kern, Institute for Interactive Systems and Data Science, TU Graz  
**Version:** 2.1.0  
**Goal:** Understand which statistical tools and approaches are suitable for data science, and the importance of assumptions and limitations.

---

## 1. Statistical Basics: Background

### 1.1 Data Science vs. (Inferential) Statistics

Two philosophical perspectives on data:

- **Data Mining perspective:** The data is a complete representation of the world and the phenomena under study. Questions are about fractions/quantities directly observed (e.g., "What fraction of posts are about topic T?").
- **Statistics perspective:** The data is obtained from an underlying generative process; the true process is what we care about. Questions are about probabilities in the generative process (e.g., "What is the probability a post is about topic T?").

Key insight: One cannot compute a probability from finite data -- only estimate it. The **Law of Large Numbers** tells us that under favourable conditions, the estimate converges to the true value as data increases.

### 1.2 Knowledge Discovery Process

Two linked pipelines:

1. **Data Generation Process:** Nature (patterns) -> Data collection -> Dataset
2. **Knowledge Discovery Process:** Dataset -> Pre-processing -> Features -> Data Mining -> Patterns

The arrow from Nature to Distribution does NOT naturally follow; assumptions about distributions are imposed by the analyst.

### 1.3 The Statistical Approach

1. Make assumptions about nature (e.g., it follows a certain distribution).
2. Use the dataset to estimate parameters of that distribution.
3. Derive patterns/insights from the estimated distribution.

**Crucial:** Insights only hold if the assumptions are correct.

### 1.4 Data Mining Grand Checklist (12 items)

1. **Linearity** -- scatter plots, common sense, transform including interactions if useful.
2. **t-statistics** -- are coefficients significantly different from zero? Check confidence interval width.
3. **F-tests** -- for subsets and equality of coefficients.
4. **R-squared** -- is it reasonably high in context?
5. **Influential observations** -- outliers in predictor space and dependent variable space.
6. **Normality** -- histogram of residuals.
7. **Studentized residuals.**
8. **Heteroscedasticity** -- plot residuals vs. each x variable, transform if necessary (Box-Cox).
9. **Autocorrelation** -- time series plot.
10. **Multicollinearity** -- compute correlations of x variables, check sign agreement with intuition.
11. **Principal Components.**
12. **Missing Values.**

Source: MIT OCW 15.062 Data Mining.

### 1.5 Types of Data Science Projects

- **Hypothesis-driven** (positivism): e.g., "Is there quality impairment if parameter X is changed?"
- **Data-driven** (constructivism): e.g., "What insights can be generated from the data?"
- **Simulation-driven:** e.g., "Can ML simulate and optimise a process?"

---

## 2. Correlation

### 2.1 Statistical Dependence

X and Y are **positively dependent** if P(X | Y) > P(X), i.e., P(X, Y) > P(X)P(Y). Negatively dependent if the inequalities are reversed. Independence is the special case where P(X, Y) = P(X)P(Y).

### 2.2 Correlation Ratio

Correlation ratio eta^2_{Y|X} = 1 - (sigma^2_{Y|X} / sigma^2_Y), where sigma^2_{Y|X} is the conditional variance.  
- eta^2 = 0 => X and Y are independent.  
- eta^2 = rho^2 => X and Y are linearly dependent.

### 2.3 Key Properties of Correlation

- **Correlation does NOT imply causation** (except special cases like time series, but watch for *post hoc ergo propter hoc*).
- Correlation analysis is only the first step -- must be followed by tests and model building.
- In big data, many variable pairs will show correlation purely by chance (spurious correlations).
- Most correlation measures range in [-1, +1] or [0, +1], but values cannot be compared across different measures (skewed, so 0.5 does not mean "half" correlated).
- Correlation and dependency are different (but related) concepts.

### 2.4 Pearson's Correlation (PC)

Formula: r_xy = sum((x_i - x_bar)(y_i - y_bar)) / sqrt(sum((x_i - x_bar)^2) * sum((y_i - y_bar)^2))

- Intuition: Linear regression Y = beta_0 + beta_1 X + epsilon. PC is the normalised covariance.
- PC = 0 does NOT imply independence (X independence of Y => PC = 0, but the converse is false).
- Sensitive to outliers.
- Fails for non-linear relationships (e.g., same PC value of 0.816 for very different non-linear patterns).

### 2.5 Visual Tools for Non-Linear Correlation

- **Hexabin plots** -- 2D histogram using hexagonal bins.
- **Contour plots** -- density contours for visualizing distribution shape.

### 2.6 Types of Non-Linear Correlations

Three broad categories:

1. **Rank correlation** -- detects monotonic relationships.
2. **Transformation-based** -- maps non-linear relationships into linear ones.
3. **Information-theoretic** -- based on mutual information.

### 2.7 Rank Correlation Measures

- **Spearman's rho:** Computes Pearson correlation on the ranks of the data.
- **Kendall's tau:** Compares the ordering of ranks (concordant vs. discordant pairs).

Intuition: Order all values of each variable by rank, then compare whether the rankings agree. E.g., is the first entry in X also the first entry in Y?

### 2.8 Maximal Correlation (mCor)

Definition: mCor(X, Y) = max_{f,g: R->R} Cor(f(X), g(Y))

Theoretical properties:
- Value is the correlation coefficient of optimally transformed inputs.
- Ranges [0, 1].
- mCor(X, Y) = 0 iff X is independent of Y.

**Estimation via ACE (Alternating Conditional Expectations) algorithm:**
Iteratively solve min_{f,g} E[(f(X) - g(Y))^2]:
1. Center and scale f(x).
2. Set f to conditional expectation of f(x) given g(y).
3. Repeat for g(y).
4. Loop until convergence.

**Caveats:** Estimation of conditional expectation (e.g., with splines) is hard. Overestimation in certain cases.

### 2.9 Distance Correlation (dCor)

- Ranges [0, 1].
- dCor(X, Y) = 0 => X is independent of Y (unlike Pearson).
- The raw value itself cannot be directly interpreted.

**Algorithm:**
1. Compute pairwise distance matrices D_X and D_Y for both variables.
2. Apply doubly centering to both -> D_X^centre, D_Y^centre.
3. Compute pairwise distances of the two centred matrices.

**Confidence value:** Combine dCor with a permutation test to estimate whether X is independent of Y, with a confidence value. dCor is more robust in small sample sizes compared to MIC.

### 2.10 Key Insight on Correlation vs. Dependency

Non-monotone functions (e.g., f_lambda(x) = sin(lambda * pi/2 * x) for lambda > 1) produce zero correlation despite perfect deterministic dependency. This demonstrates that **correlation is not equivalent to dependency**.

### 2.11 Maximal Information Coefficient (MIC)

- **Idea:** Compute mutual information I(X; Y) via binning, then normalise.
- **Definition:** MIC(X, Y) = max_{X_bins * Y_bins < B} I(X; Y) / log2(min(X_bins, Y_bins)), where B is a hyperparameter.
- Ranges [0, 1].
- Tends to 0 for statistical independence, tends to 1 for noiseless functional relations.
- Symmetric: MIC(X, Y) = MIC(Y, X).
- **MIC is NOT an estimate of mutual information** -- it is a normalised score.

**Comparison with dCor:** MIC requires large sample sizes; dCor is more robust in small samples.

### 2.12 Related Correlation Methods

- Non-pairwise correlations (partial correlation, confounders)
- Similarity / distance measures
- Statistical significance tests for correlation
- Symbolic regression
- Copulas

### 2.13 Important Visual Patterns in Correlation

- **Appears to be correlated:** Global trend may mislead.
- **No correlation:** Data appears random.
- **Found correlation:** There may be correlation at a global scale, but also smaller correlation structures in subsets of the data (specific ranges of certain variables).

---

## 3. Hypothesis Testing

### 3.1 Core Concepts

Let x_S = f(D) be the value of the test statistic for dataset D.  
Let X_S be the random variable describing the test statistic under the null hypothesis H0.

- **p-value:** p = P(X_S is more extreme than x_S | H0 is true). "More extreme" depends on the test (e.g., X_S >= x_S, X_S <= x_S).
- **Rejection rule:** Given significance level alpha in (0, 1), reject H0 iff p <= alpha. The result S is then called **statistically significant**.

### 3.2 Types of Errors

| | H0 true | H0 false |
|---|---|---|
| Reject H0 | **Type I error** (false discovery) | Correct |
| Do not reject H0 | Correct | **Type II error** (miss) |

- **Type I error:** Reject H0 when H0 is true -- flagging a finding as significant when it is not. (False positive.)
- **Type II error:** Fail to reject H0 when H0 is false -- missing a real effect. (False negative.)

### 3.3 Fisher's Exact Test

Used for 2x2 contingency tables (e.g., 2 groups, 1 binary feature). Tests whether the two groups differ significantly with respect to the feature. Directly computes a p-value via the hypergeometric distribution:

p = (C(a+b, a) * C(c+d, c)) / C(n, a+c)

where a, b, c, d are cell counts. The chi-squared test is a computationally cheaper approximation.

### 3.4 Permutation Test

**Idea:** Simulate the null distribution by randomly permuting the data labels many times. Count how often the observed (or more extreme) result occurs under these random permutations.

- **Downsides:** Computationally expensive; formally imprecise.
- Source: Westfall & Young, "Resampling-Based Multiple Testing".

### 3.5 Testing Multiple Hypotheses

Testing many hypotheses at the same alpha level will yield spurious significant results by chance.

- **Family-Wise Error Rate (FWER):** Probability of making at least one Type I error across all tests. Corrections provide guarantees on the expected number of false discoveries.
- **Correction methods:** Bonferroni correction, Bonferroni-Holm procedure, LAMP.
- **Validation dataset approach:** Need a statistically significant result on multiple independent splits.

### 3.6 Bonferroni Correction

- **Problem:** Random rejection of H0 due to multiple tests -- accumulation of alpha error.
- **Idea:** Adapt significance level for n tests: p* < alpha / n.
- **Alternative:** Adapt the p-value of individual tests (multiply by n).
- **Caveat:** Bonferroni is conservative (reduces power, increases Type II error risk).

### 3.7 p-Hacking

Dishonest practices to force p-values below significance thresholds:

**Motivations:** Publication bias (need "good" results), commercial interests.

**Common approaches:**
1. Repeat the experiment until -- by chance -- significance is achieved.
2. Increase the number of hypotheses tested (try all variable combinations until a significant relationship is found by chance).
3. **HARKing** -- Hypothesizing After the Results are Known (formulate hypotheses to match discovered patterns).

### 3.8 Data Mining Without Expert Knowledge

If data requires interpretation, different analysts can reach dramatically different results. Example: A crowdsourced analysis of soccer referee bias had 29 different analytical teams using 21 unique covariate combinations -- 69% found a significant positive effect, 31% found a non-significant relationship.

---

## 4. Causality

### 4.1 Basics

Causality studies the impact of hypothetical actions or interventions.  
Example: "Would there be fewer strokes if we ate fewer Wienerschnitzel?"

### 4.2 Preferred Solution: Randomised Control Study

1. Build two groups (randomly assigned) -- treatment group and control group.
2. Apply the intervention to the treatment group; no treatment for controls.
3. Observe the difference in outcomes.

**Note:** Often such studies cannot be conducted (e.g., unethical).

### 4.3 Causal Graphs

Each variable is a node; directed edges indicate causal relationships (arrow from cause/parent to effect/child).

**Structural elements:**
- **Indirect causes** -- path relationships through mediators.
- **Forks** -- one cause producing multiple effects.
- **Mediators** -- variables on the causal path between cause and effect.
- **Colliders** -- multiple causes converging on a single effect.

### 4.4 Confounders

Confounders are variables that create the impression of a causal relationship between observed variables when none exists (or mask a real relationship). They are a common source of spurious correlations.

### 4.5 Simpson's Paradox

A statistical phenomenon where a trend appears in different subgroups of data but disappears or reverses when the groups are combined.

Formally: Given event Y and variables X, Z:
- P(Y | X) < P(Y | not X) at population level
- P(Y | X, Z=z) > P(Y | not X, Z=z) for all values of Z

This contradicts intuition: a trend true for all subpopulations should hold for the total population, but confounding by the grouping variable Z reverses it.

### 4.6 Explaining Away (Collider Bias)

Two independent variables can appear (negatively) correlated when both are causes of a third observed variable. This arises from the sampling strategy (conditioning on the collider). Also known as Berkson's paradox.

---

## 5. Limitations: Data Processing Inequality

### 5.1 Motivation

Given data, we seek to find patterns. Our goal is to learn how data and patterns are related. The reverse direction (how data is generated from patterns) reveals fundamental limits.

### 5.2 Can We Reconstruct Patterns?

We can only reconstruct original patterns from a dataset **if the information (of the patterns) is still available**. If sensors do not record a pattern, we cannot recover it later.

### 5.3 Markov Chain Model of Data Processing

Model the pipeline as a Markov chain: X -> Y -> Z  
(e.g., X = raw data, Y = pre-processed data, Z = extracted features / clustering result).

Joint probability factorisation: P(X, Y, Z) = P(Z | Y) * P(Y | X) * P(X).

Key property: X and Z are **conditionally independent given Y**.

### 5.4 Information-Theoretic View

- Since X and Z are conditionally independent given Y: I(X; Z | Y) = 0.
- This implies: I(X; Y) >= I(X; Z) and I(Y; Z) >= I(X; Z).

**Conclusion:** Along any processing pipeline, information can only be lost (or at best preserved), never gained. Once we lose critical information, there is no way to recover it. This also applies to multi-layer neural networks.

### 5.5 Practical Considerations ("Yes, but")

- **Data fusion:** Using multiple datasets (sources of evidence) can add information back -- this is part of feature engineering.
- **Algorithmic performance:** Even if information is lost during processing, some algorithms may perform "better" because they can more effectively exploit the remaining information (e.g., extracting/parsing text loses information, but raw text cannot be fed directly to most algorithms).

---

## 6. Limitations: Central Limit Theorem (CLT)

### 6.1 Statement

The CLT states that when independent random variables with finite variance are summed, their properly normalised sum tends toward a normal distribution (bell curve), **even if the original variables themselves are not normally distributed**.

Example: The sample mean of i.i.d. variables will be approximately normally distributed for sufficiently large sample sizes.

### 6.2 What Breaks the CLT: Cauchy Distribution

- If X and Y are normally distributed and independent, then Z = X + Y is also normally distributed (CLT holds).
- But Z = X / Y (e.g., using a ratio as a feature) follows a **Cauchy distribution**.

**Properties of the Cauchy distribution:**
- Does NOT have a finite variance (undefined moments).
- Therefore, the CLT does NOT apply.
- The sample mean of i.i.d. Cauchy variables is again Cauchy -- the mean does not converge.
- We cannot estimate the mean (there is none).
- However, we CAN estimate the median.

### 6.3 Robust Statistics

A sub-field of statistics studying methods that are not (or are less) affected by:
- Outliers
- (Small) violations of modelling assumptions

**Key examples:** Median (instead of mean), interquartile range (instead of standard deviation).

**Asymptotic breakdown point:** The proportion of outliers a statistic can tolerate before becoming arbitrarily biased. Median has breakdown point 50%.

### 6.4 Practical Considerations

- **Do not assume everything is normally distributed.**
- **Check for outliers** -- both visually and algorithmically.
- **Consider robust techniques** like winsorising (clipping extreme values): x' = min(L, max(-L, x)) or robust statistics (median, IQR).

---

## 7. References

1. Breiman, L. E. O., & Friedman, J. H. (1985). Estimating optimal transformations for multiple regression and correlation. *Journal of the American Statistical Association*, 580-598. -- Basis for the ACE algorithm / Maximal Correlation.

2. Reshef, D. N., Reshef, Y. A., Finucane, H. K., et al. (2011). Detecting novel associations in large data sets. *Science*, 334(6062), 1518-1524. -- Introduces the Maximal Information Coefficient (MIC).

3. Silberzahn, R., Uhlmann, E., Martin, D., et al. (2015). Crowdsourcing data analysis: Do soccer referees give more red cards to dark skin toned players? -- Demonstrates how different analytical choices lead to different conclusions.

---

*End of summary. Total slides covered: 72 (slides 1-72), covering Statistical Basics (Correlation, Hypothesis Testing, Causality) and Limitations (Data Processing Inequality, Central Limit Theorem).*
