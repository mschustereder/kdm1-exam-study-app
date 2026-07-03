# 03 Visual Preprocessing — Medium Summary

## Overview & Learning Objectives
Understand and apply visual exploration tools to inspect datasets before building models. The human visual system excels at pattern recognition — far beyond current ML — making visual inspection invaluable for spotting distributions, outliers, correlations, clusters, and relevant features. The broader lesson: raw data requires careful preprocessing and thoughtful feature extraction before it is suitable for ML algorithms. Sampling may help with large datasets; automatic filtering can assist when there are many features.

## Distributions — Understanding How Data Is Spread (Univariate Analysis)
- **Boxplot** — Visual summary of range: shows the median (central tendency), the interquartile range (IQR = Q3 minus Q1, containing the middle 50 % of data), whiskers (typically extending 1.5 IQR beyond the quartiles), and individual outlier points beyond the whiskers. Useful for quick comparisons between features or groups. For a normal distribution, 1.5 IQR covers about 99.3 % of density; 1 standard deviation covers 66.27 %.
- **Histogram** — Binned frequency display of a single variable. Works directly for categorical variables. For continuous variables, the bin width choice strongly influences appearance and interpretation. Frequencies approximate probabilities but are not identical to them.
- **KDE (Kernel Density Estimator)** — Converts a finite set of discrete observations into a smooth, continuous density estimate, resembling a probability density function (PDF). Requires a kernel function (most commonly the normal/Gaussian kernel) and a bandwidth parameter that controls smoothness: wider bandwidth yields smoother estimates; narrower bandwidth risks overfitting to noise.
- **Violin Plot** — Hybrid visualisation combining a boxplot (central statistics and outliers) with a KDE (distribution shape mirrored on both sides). Exposes multimodality and skew that a boxplot alone would hide. Considered harder to read and less commonly used than basic boxplots.
- **QQ Plot (Quantile-Quantile Plot)** — Plots observed data quantiles against the quantiles of a theoretical distribution (most often the normal distribution). If the data follows the assumed distribution, points fall near the diagonal line. Systematic curvature or S-shapes indicate skew or heavy tails. With small samples (e.g., 100 points), randomness can produce misleading deviations — larger datasets make judgement more reliable.

## Dependencies — Relationships Between Variables (Multivariate Analysis)
- **Scatterplot** — Two continuous variables mapped to x/y axes; each observation is a dot. Reveals linear/non-linear correlation, noise (dots scattered uniformly), outliers (dots in isolation or low-density regions), missing data (empty patches), and cluster structures. Density is hard to judge in dense regions; remedies include transparency (alpha blending) and jittering (adding small random noise to point positions). A scatterplot matrix shows multiple pairwise relationships in one figure, with KDE or histograms on the diagonal.
- **Heatmap** — 2D grid coloured by point count, effectively combining a scatterplot with histogram-like binning. Useful when scatterplots become too dense to read. Can encode a third variable via colour intensity.
- **Higher-Order Dependencies** — Visualisation becomes impractical beyond 3 dimensions. Dimensionality reduction techniques such as PCA (Principal Component Analysis) project high-dimensional data down to 2–3 combined dimensions, each a linear combination of the original variables. This enables at least partial visual inspection of complex datasets.
- **Domain-Specific Visualisations** — Maps for spatial/geographic data; time-series plots for sequential/temporal data. Dedicated visualisations are often needed for specialised data types.
- **Advanced Tools** — Platforms like Tableau, Power BI, and AI Visualiser offer coordinated multiple views (e.g., scatterplot plus histogram plus map, all linked so selections update across panels). Parallel coordinates are another technique for seeing many dimensions at once. Custom dashboards can be built with Streamlit.

## Feature Extraction — From Raw Data to Input Features
- **Data vs. Information** — Raw recorded data is often useless for ML until transformed. Feature extraction surfaces informative, implicit, useful patterns. A feature is an individual measurable property of a phenomenon, suitable for consumption by data mining algorithms and potentially useful for prediction.
- **Feature Types (Stevens 1946)** — Numeric: continuous (interval — equal intervals, no true zero, e.g., date; ratio — intervals with a defined zero, e.g., temperature in Kelvin, age) or discrete (counts). Categorical: nominal (unordered categories, e.g., gender, colour) or ordinal (ordered categories, e.g., rankings). Likert scales (common in surveys) are a classic ordinal example. Continuous features are frequently discretised into categorical bins; binary features are particularly common.
- **Categories by Context** — Contextual features (position, browsing history), structural features (DOM elements, markups), linguistic features (POS tags, noun phrases), plus domain-specific features (e.g., from biomedicine or finance).
- **Domain Examples of Features** — Images: colours, textures, contours, gradients. Signals: frequency, phase, peaks, spectrum. Time series: trends, seasonality, self-similarities. Text: words, POS tags, grammatical dependencies. Biomed: DNA sequence, treatment response. Qualitative: questionnaire ratings.
- **Worked Example — Handwriting Recognition** — MNIST-style pipeline: raw scanned digits → preprocessing steps (noise removal, pressure-saturation normalisation, size normalisation, centring via centre-of-mass or bounding box) → feature extraction (each pixel becomes a binary feature). Centring was found to improve SVM classifier performance.

## Preprocessing — Practical Data Quality and Transformation
- **Importing Data** — Address encoding issues (UTF-8 vs. Latin-1), separator/escape characters in CSV files, decimal comma vs. decimal dot conventions, locale-specific characters (umlauts, accents). Assign correct feature types at parse time to avoid downstream errors.
- **Common Quality Issues** — Unnecessary data (irrelevant columns/rows), missing values, noise (random measurement errors), incorrect entries, inconsistent formatting, duplicate records, and disguised missing data (e.g., placeholder values like 9999). All quality impairments can negatively affect the data mining task.
- **Transformations** — Split the dataset into coherent subsets (e.g., one per user group). Remove irrelevant rows or columns. Transform input data to achieve consistency (e.g., rescaling, centring, distribution reshaping such as log transforms). Some algorithms prefer centred or normalised data. Add external data from trusted sources to enrich the feature set.
- **Ethical and Practical Principles** — _Garbage in, garbage out_: data quality directly determines model quality. _Bias in, bias out_: biased data perpetuates biased predictions. Consider privacy (handling personal data lawfully) and fairness (avoiding sensitive-attribute discrimination) throughout the preprocessing pipeline.

## Key Figures and Diagrams Referenced in the Lecture
- Boxplot anatomy — median, IQR box, whiskers, outliers — with normal-distribution overlay showing how 1.5 IQR maps to 99.3 % density coverage and 1 sigma to 66.27 %.
- Scatterplot gallery — examples of bivariate normal distributions (varying covariance), linear relationships (including deterministic cases where one variable fully determines another), and nonlinear/circular patterns, each annotated with distance correlation values.
- QQ plot series — three independent samples of 100 normal-distribution points demonstrating how random sampling variation can make normality hard to confirm.
- Scatterplot matrix with KDE on the diagonal — a compact overview of pairwise relationships in a multi-feature dataset.
- MNIST preprocessing pipeline diagram — raw digits → despeckle → deslant → size-normalise → centre → pixel grid features.

## Summary of Key Formulas and Statistics
- **IQR** = Q3 − Q1 (middle 50 % of data). Whiskers extend to min(max, Q3 + 1.5 IQR) and max(min, Q1 − 1.5 IQR).
- **Pearson correlation** assumes bivariate normality to describe association completely — always verify distribution before relying on it.
- **Distance correlation** — a more general measure (mentioned on scatterplot examples) that can detect nonlinear dependencies where Pearson would report zero.
- **KDE formula** (conceptual): f_hat(x) = (1/n) * sum K_h(x − x_i), where K_h is the scaled kernel with bandwidth h.
