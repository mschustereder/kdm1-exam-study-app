# 02 Dataset Collection - Summary

## Core Definitions and Concepts

- **Data Processing Pipeline**: Knowledge is extracted from nature through a data generation process, then processed via pre-processing, feature extraction, and data mining. Information not collected at the source is lost forever (data processing inequality).
- **Observational data**: Passively collected without interfering with the data generation process (e.g., sensors, questionnaires, machine logs). Cannot in general support causal claims -- correlation does not imply causation.
- **Interventional data**: Collected while actively making changes (e.g., drug trials). Reflects the intervention in the data generation process and can support causal inference.
- **Biased data / Sampling bias**: A non-representative sample is collected due to some influencing factor. Problems compound when the sampling bias changes over time.
- **Domain shift**: When combining multiple datasets collected under different conditions (e.g., demographic data from different countries), the datasets cannot simply be joined.
- **Causal inference and data-fusion** (Bareinboim & Pearl, 2016): Whether causal effects can be estimated depends on the underlying causal relationships and the data collection procedure.

## Types of Data Sets

- **By structure**: Structured (schema-defined, e.g., databases), Semi-structured (mixture, e.g., JSON, spreadsheets), Unstructured (e.g., text, log files, slides). Most real-world data is unstructured.
- **Tabular data** (most common in data science): Rows = instances, Columns = features/attributes. One column is typically the target (dependent variable). Convention follows the I.I.D. assumption: instances are independent and identically distributed.
- **Time series data**: Univariate (single variable over time) or Multivariate (multiple interacting variables). Does **not** follow the I.I.D. assumption -- often autoregressive (past predicts future).
- **Other data types**: Image data (spatial dependencies between pixels), Graph data (nodes/edges with dependencies via homophily), Text data (unstructured), Spatial data.
- **Practical aspects of tabular data**: CSV is the lingua franca; feature types, encoding, and separators are typically unknown.

## Key Algorithms / Techniques

- **Web crawling**: A specialised HTTP client that starts from seed pages, downloads content (typically HTML), extracts links, and adds them to a frontier for subsequent crawling.
- **Crawler types**: Batch (snapshot up to a threshold), Incremental (revisiting URLs for freshness), Specialised (focused, topical).
- **Crawling strategies**: Breadth-first (Queue/FIFO -- shortest path) or Depth-first (Stack/LIFO -- moves away quickly from start).
- **Concurrent crawling**: Multiple machines in parallel, requiring synchronised shared data structures.
- **Deep crawling**: Automatically filling out web forms to access the hidden/deep web.
- **Topical crawler**: On-the-fly crawling starting from seed pages, finding pages similar to a reference.
- **Focused crawler**: Collects pages with specific properties (thematic, type) using lexical cues and link topology. Relies on the **cluster hypothesis** (pages close to a relevant page are likely relevant). Uses two conjectures:
  - Link-content conjecture: linked pages tend to be lexically similar.
  - Link-cluster conjecture: linked pages tend to be semantically related.
- **Wrapper generation** for web information extraction: Manual, Wrapper induction (supervised -- uses landmarks/token sequences to locate target items), and Automatic extraction (unsupervised). Active learning / co-training reduces manual labelling effort.
- **Evaluation metrics for crawlers**: Harvest rate (percentage of good pages) and Search length (pages crawled before finding a target percentage of relevant pages).
- **Semantic Web & Microformats**: Semantic Web (RDF, machine-readable, little uptake) vs. Microformats (lightweight HTML-embedded markup, widely supported by search engines, e.g., OpenStreetMap's geo microformat).
- **Dataset augmentation**: Extending a dataset by applying label-preserving transformations (e.g., rotation, scaling) to improve model robustness.
- **Surrogate modelling**: Using computationally expensive simulations (e.g., finite element method) to generate a dataset, then training a faster ML model as a surrogate.

## Important Formulas / Diagrams

- **Data processing pipeline diagram**: Two parallel processes: (1) Data Generation Process (Nature -> Data Collection -> Dataset) and (2) Knowledge Discovery Process (Dataset -> Pre-processing -> Features -> Data Mining -> Patterns). Information is only lost, never gained, between stages.
- **Tabular data convention**: Rows = instances, Columns = features, one column = target/dependent variable.
- **I.I.D. assumption**:
  - Independence: no systematic dependency between rows.
  - Identically distributed: all rows drawn from the same distribution.
- **Link-content conjecture plot**: Cosine similarity decays as a function of mean directed link distance.
- **Link-cluster conjecture plot**: Mean likelihood ratio decays as a function of mean directed link distance from Yahoo! directory topics.
- **robots.txt example** (orf.at): Standard `User-agent: *` directives, plus explicit blocking of known bad bots.

## Learning Objectives

1. Understand why data collection is the most critical step in the knowledge discovery pipeline -- lost information cannot be recovered.
2. Distinguish between observational, interventional, biased, and domain-shifted data collection, and understand their implications for causal inference.
3. Identify data by structure (structured, semi-structured, unstructured) and by type (tabular, time series, image, graph, text, spatial).
4. Apply the I.I.D. assumption correctly and recognise when it does not hold (e.g., time series, repeated subjects).
5. Know where to find existing datasets: UCI Machine Learning Repository, UEA/UCR Time Series Repository, Zenodo, Kaggle, Awesome Public Datasets, open government data portals.
6. Understand the architecture, strategies, and challenges of web crawling (volume, volatility, dynamic content, URL canonicalisation, error handling, ethics, legal considerations).
7. Differentiate between types of crawlers (batch, incremental, topical, focused, deep-web) and their evaluation metrics.
8. Grasp wrapper generation approaches (manual, wrapper induction, automatic) for extracting structured data from template-generated web pages.
9. Distinguish study types: controlled experiments (between-subjects vs. within-subjects), quasi-experiments, and survey research (cross-sectional, case-controlled, cohort).
10. Recognise when and why to create synthetic datasets (ethical/legal constraints, isolated testing, dataset augmentation, surrogate modelling).
11. Appreciate the need for dataset documentation to avoid "documentation debt" and to capture domain knowledge (physical constraints, data invariants).
