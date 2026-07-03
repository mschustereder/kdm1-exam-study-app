# 03 — Visual Preprocessing

## Learning Objectives
- Understand visual exploration tools and be able to apply them to unseen datasets.
- Use human pattern recognition to analyse and explore a given dataset before applying automated methods.

## Core Definitions & Concepts
- **Visual inspection**: Essential first step before any data mining — helps gain data understanding for small enough datasets. For large datasets, sampling or automatic filtering may be needed.
- **What to look for**: distribution of the data, outliers/missing values/artefacts, dependencies (correlations), groups/clusters, and relevant features.
- **Univariate analysis**: examines a single feature. **Multivariate analysis**: examines combinations of multiple features (bivariate = exactly two).
- **Data vs. Information**: Raw data is recorded facts; information is novel, implicit, useful patterns extracted from data.
- **Features**: Individual measurable properties of a phenomenon — also called attributes (ML) or variables (statistics). They encode properties in a form suitable for a chosen algorithm.
- **Feature types (Stevens 1946)**: Numeric (continuous, interval, ratio; discrete counts) and Categorical (nominal — unordered categories; ordinal — ordered categories). Binary features are a common special case.

## Key Algorithms / Techniques

### Distribution Visualisation
- **Boxplot**: Shows range, median/mean, interquartile range (IQR = Q3–Q1, contains 50% of values), whiskers (typically 1.5 × IQR), and outliers. For a normal distribution, 1.5 IQR covers ~99.3% of density.
- **Histogram**: Frequency of values; works directly for categorical variables; continuous variables require binning. Frequency is not identical to probability but can estimate it.
- **Kernel Density Estimator (KDE)**: Takes discrete data and produces a continuous output (similar to a PDF). Requires a kernel (normal kernel is popular) and a bandwidth parameter for smoothing.
- **Violin plot**: Combines a boxplot with a PDF plot — shows distribution shape where a boxplot alone cannot.
- **QQ Plot**: Visual check for an assumed distribution. Expected distribution on x-axis, observed on y-axis; data should align on the main diagonal. Often used for normality checks.

### Dependency Visualisation
- **Scatterplot**: Bivariate visualisation of continuous variables. Each axis is a variable, each point is an observation. Useful for detecting noise, outliers, missing data (empty patches). Hard to judge density — transparency or jittering may help. **Scatter matrix**: pairwise scatter plots for a quick overview of small datasets.
- **Heatmap**: Combines scatter plot characteristics with histograms; good for showing density or a third variable when scatter plots fall short.
- **Higher-order dependencies**: Hard to visualise beyond 3 dimensions. Often addressed via **dimensionality reduction** to 2–3 dimensions, where each resulting dimension is a combination of original variables.

### Preprocessing
- **Data import concerns**: encoding, separator/escape characters, assigning feature types, consistent parsing (decimal separators, umlauts).
- **Quality issues to identify**: unnecessary data, missing values, noise, incorrect/inconsistent data, formatting issues, duplicates, disguised data ("garbage in, garbage out").
- **Preprocessing tasks**: split into coherent subsets, remove unneeded rows/columns, transform for consistency or distribution goals, add external data. Also consider privacy and fairness ("bias in, bias out").

### Feature Extraction Examples
- **Handwriting recognition (MNIST-style)**: Input = scanned digits; Preprocessing = remove noise, normalise size, centre images; Feature extraction = pixels as binary features.
- **Feature categories**: Contextual (position, browsing history), Structural (DOM elements, markup), Linguistic (POS tags, noun phrases).
- **Domain examples**: Images → colours/textures/contours; Signals → frequency/phase/spectrum; Time series → ticks/trends/seasonality; Biomed → DNA sequence; Text → words/POS tags; Qualitative → questionnaire ratings.

## Important Formulas / Diagrams Referenced
- **Boxplot + PDF overlay**: Shows relationship between IQR and normal distribution density (1.5 IQR ≈ 99.3% density; 1 sigma ≈ 66.27%).
- **QQ Plot examples**: 100,000 points from N(1,0) align on diagonal; Beta(0.9, 0.9) deviates visibly.
- **Scatterplot examples**: Bivariate normal with varying covariance, linear relationships, and cases with multiple possible y-values per x-value. Distance correlation annotated on examples.
- **Streamlit / coordinated view tools**: Advanced visualisation with parallel coordinates and linked selections (e.g., Tableau, Power BI, AI Visualiser).
