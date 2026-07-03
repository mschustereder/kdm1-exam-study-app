# 03 — Visual Preprocessing

**Lecturer:** Roman Kern <rkern@tugraz.at>, Institute of Machine Learning and Neural Computation, TU Graz  
**Version:** 1.0.2  
**Lecture:** KDDM1 — Knowledge Discovery and Data Mining 1

---

## 1. Introduction & Motivation

### Why data inspection matters
- Humans possess excellent pattern recognition abilities that still exceed most machine learning approaches for exploratory analysis.
- Visual inspection is a powerful first step to analyse and explore a given dataset before applying automated methods.
- **Goal of this unit:** Understand visual exploration tools and be able to apply them to unseen datasets.

### Sampling and filtering
- For large datasets, a **sampling approach** (working with a representative subset) may be the only feasible way to visualise the data.
- When the number of features is very large, an **automatic filtering process** (e.g., feature selection or dimensionality reduction) may be needed to reduce visual complexity.

### What to look for (objectives)
1. **Distribution of the data** — e.g., skewed distributions, multi-modality.
2. **Outliers, missing values, artefacts** — anomalies that can distort analysis (in this context, outliers and anomalies are treated as synonymous).
3. **Dependencies (correlation)** — relationships among independent variables and between independent variables and the target.
4. **Groups (clusters)** — natural groupings or partitions within the data.
5. **Relevant features** — which attributes carry predictive or informative value.

---

## 2. Distributions

### Overview
- In finite, observational data, the **distribution** is an assumption about the underlying data generation process.
- Knowing the distribution tells us whether certain statistical methods are appropriate. Example: if two variables follow a bivariate normal distribution, Pearson's correlation coefficient completely describes their association.

### Types of Analysis
- **Univariate analysis**: examines a single feature in isolation.
- **Multivariate analysis**: examines combinations of multiple features simultaneously. Start with univariate analysis, then move to bivariate (two features), then higher-order.

### Boxplot
- Provides a compact visual overview of a feature's range and key statistics.
- Displays the **median** (and sometimes the mean), the **interquartile range (IQR)** — where Q1 = 0.25 quantile and Q3 = 0.75 quantile, together covering 50% of the data — and **outliers**.
- **Whiskers** extend from the box, typically to the minimum/maximum value or to **1.5 x IQR** beyond Q1/Q3. Points beyond the whiskers are plotted individually as potential outliers.
- For a normal distribution, the 1.5 x IQR rule covers approximately 99.3% of the density, while 1 standard deviation covers about 66.27%.

### Histograms
- Shows the **frequency** (not probability, though frequency can be used to estimate probability) of values across bins.
- Works directly for **categorical variables** (each category is a bar).
- For **continuous variables**, values must be grouped into bins (binning). The choice of bin width and boundaries strongly influences the visual appearance.

### Kernel Density Estimation (KDE)
- Takes **discrete data** (finite sample) and produces a **continuous output** that approximates a probability density function (PDF).
- Requires a **kernel function** that acts as a smoother placed over each data point.
- The **normal (Gaussian) kernel** is a popular choice; it requires a **bandwidth parameter** controlling the degree of smoothing — too small produces spiky estimates, too large oversmooths structure.

### Violin Plots
- Boxplots cannot reveal the shape of a distribution (e.g., bimodality).
- **Violin plots** combine a boxplot with a rotated KDE/PDF on each side, showing both summary statistics and the full density shape.
- Note: Violin plots are less common and are generally considered harder to read and interpret by non-experts.

### QQ Plot (Quantile-Quantile Plot)
- A visual check to **confirm or reject an assumed distribution** for a dataset.
- The **expected distribution's quantiles** are plotted on the x-axis; the **observed data's quantiles** are plotted on the y-axis.
- If the data follow the assumed distribution, points should align along the **main diagonal** (y = x line).
- Most often used to test for **normality** (Gaussian distribution).
- **Important caveat:** With small samples (e.g., 100 data points), even data truly drawn from a normal distribution can produce QQ plots that deviate noticeably from the diagonal — visual judgment requires experience and awareness of sampling variability.

---

## 3. Dependencies

### Introduction
- Goal: identify **systematic dependencies** between variables — correlation between independent variables, correlation with the dependent (target) variable, groups/clusters in the data, and partitions.
- **Note:** Correlation can be observed/measured in data and may be a result of dependencies, but correlation does not imply causation.

### Scatterplot
- The primary tool for visualising **bivariate dependencies between continuous variables**.
- Each variable is assigned an axis; each data point is represented by a dot (or other marker).
- Visual patterns (linear, curved, clustered, dispersed) can indicate the presence and type of dependency.
- **Density estimation is hard** from scatterplots alone — overlapping points obscure concentration. Remedies:
  - **Transparency (alpha blending)** — reveals regions of high overlap.
  - **Jittering (adding small random noise to point positions)** — helps when many points share identical coordinates.
- Example patterns shown in lecture:
  - Top row: classical bivariate normal distributions with varying covariance — in the middle the two variables appear independent.
  - 2nd row: perfect linear relationships (knowing one variable determines the other exactly).
  - 3rd & 4th rows: non-functional relationships where one value of x maps to multiple possible values of y.

### Scatter Matrix (Pair Plot)
- A grid of multiple pairwise scatter plots, providing a comprehensive overview of a (small) dataset.
- The main diagonal often displays the univariate distribution of each feature (e.g., as a KDE or histogram).
- An excellent tool for quickly spotting correlations, outliers, and patterns across all variable pairs.

### What scatterplots reveal
- **Noise** — data points scattered all over the plot with no discernible pattern.
- **Outliers** — points far from the main body or in low-density regions.
- **Missing data** — empty patches or gaps in the plot where data would otherwise be expected.

### Heatmaps
- Scatterplots work best for continuous variables but struggle to convey density or a third variable.
- **Heatmaps** combine characteristics of scatter plots with histograms by binning the 2D space and colour-coding the count (or density) in each bin.
- Effective for large datasets where overplotting in scatterplots would obscure structure.

### Higher-Order Dependencies
- Visualising dependencies works well in up to **3 dimensions** (2D/3D scatterplots).
- Higher-order dependencies (4+ dimensions) are fundamentally hard to visualise directly.
- Common strategy: **dimensionality reduction** — transform the high-dimensional dataset to 2–3 dimensions, where each resulting dimension is a combination of the original features. Examples include PCA (Principal Component Analysis), t-SNE, and UMAP.

### Specific Dataset Visualisations
- **Spatial data** — maps (e.g., geospatial heatmaps).
- **Time series** — line plots with time on the x-axis.
- Specialised visualisations exist for domains like biomed, text, networks, etc.

### Tools
- Advanced visual inspection tools (e.g., Tableau, Power BI, AI Visualiser) offer:
  - Specialised components like **parallel coordinates** for high-dimensional data.
  - **Coordinated views** — changes and selections in one component are reflected in all others.
- Custom apps can be built with frameworks like **Streamlit**.

---

## 4. Feature Extraction

### Data vs. Information
- **Raw data** is often useless for direct machine learning — it cannot be fed directly to automatic methods.
- **Data**: recorded, collected, or crawled facts (the raw material).
- **Information**: novel, informative, implicit, or useful patterns within the data that must be extracted.
- Feature extraction is the bridge from raw data to usable information.

### What are features?
- An **individual measurable property** of a phenomenon being observed.
- The items that represent knowledge suitable for data mining algorithms.
- A piece of information that is potentially **useful for prediction**.
- Also called **attributes** (in machine learning) or **variables** (in statistics).
- Features encode properties in a way suitable for the chosen algorithm.

### Examples of Features by Domain
- **Images** — colours, textures, contours, gradients.
- **Signals** — frequency, phase, samples, peaks, spectrum.
- **Time series** — ticks, trends, self-similarities, seasonality.
- **Biomedical** — DNA sequence, response to intervention.
- **Text** — words, part-of-speech (POS) tags, grammatical dependencies.
- **Qualitative** — questionnaire responses, subjective ratings.

### Types of Features (Stevens, 1946)
**Numeric (quantitative data):**
- **Continuous** — can take any value within a range (e.g., height, time).
  - *Interval* — equal intervals but no true zero (e.g., date).
  - *Ratio* — intervals with a defined, meaningful zero point (e.g., temperature in Kelvin, age).
- **Discrete** — integer counts (e.g., number of purchases).

**Categorical (qualitative data):**
- **Nominal** — two or more categories with no ordering (e.g., gender, colour).
- **Ordinal** — categories with a meaningful order/ranking (e.g., survey responses: "bad" < "neutral" < "good").

**Binary features** are common in practice. Continuous features are often transformed to categorical ones (binning/discretisation).

### Categories of Features (by Context)
- **Contextual features** — position information, browsing history.
- **Structural features** — structural markups, DOM elements in web data.
- **Linguistic features** — POS tags, noun phrases.
- Many more domain-specific categories exist.

### Worked Example: Handwriting Recognition
- **Input:** A collection of scanned handwritten digits (popularised by the MNIST dataset; see also Harrington 2012, *Machine Learning in Action*).
- **Preprocessing steps:**
  1. Remove noise (e.g., scanning artefacts).
  2. Adapt saturation changes (due to differences in pen pressure).
  3. Normalise all images to the same size.
  4. Centre the images (using centre of mass or bounding box). Depending on the centring algorithm, some algorithms (e.g., SVM) improve significantly in performance.
- **Feature extraction:** Use the pixel values directly as **binary features** (each pixel is either "ink" or "no ink").

---

## 5. Preprocessing — Practical Considerations

### Task of Importing Data
- Data quality concerns begin at import:
  - **Encoding** (e.g., UTF-8 vs. Latin-1 for handling umlauts and special characters).
  - **Separator** and **escape character** in formats like CSV.
  - Consistent parsing of numbers (comma vs. dot as decimal separator).
  - Assign correct **feature types** after parsing.

### Quality Issues to Identify
- Unnecessary data (irrelevant rows or columns).
- Missing values.
- Noise.
- Incorrect data.
- Inconsistent data.
- Formatting issues.
- Duplicate information.
- Disguised data (e.g., placeholder values like 9999 for missing).

**Principle:** *Garbage in, garbage out* — all quality impairments can negatively affect downstream data mining tasks.

### Tasks of Preprocessing
1. **Split the dataset** into coherent parts (e.g., one dataset per user group).
2. **Remove unneeded data** — drop rows (instances) or columns (features) that are irrelevant.
3. **Transform input data**:
   - Achieve **consistency** across records.
   - **Transform the distribution** of features (e.g., log-transform to reduce skewness).
4. **Add additional data** from external sources (feature enrichment).
5. **Help the algorithm** pick up relevant information in a suitable way — e.g., some algorithms prefer centred or normalised/scaled data.
6. **Further considerations:**
   - **Privacy** — handling personal data appropriately.
   - **Fairness** — avoiding bias encoded in sensitive attributes.
   - **Bias in, bias out** — biases in the data will propagate to the model.

---

*End of summary. Based on the lecture slides "KDDM1 — Visual Preprocessing" by Roman Kern, Version 1.0.2, Institute of Machine Learning and Neural Computation, TU Graz.*
