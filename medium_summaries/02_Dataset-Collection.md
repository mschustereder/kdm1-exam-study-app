# 02 — Dataset Collection

## Definitions
- **Data Collection Types:** (a) *Observational* — passively collected without interfering (sensors, logs, questionnaires); (b) *Interventional* — collected while actively making changes (drug trials); (c) *Biased* — sampling bias distorts representativeness; (d) *Domain Shift* — combining datasets collected under differing conditions violates comparability.
- **I.I.D. Assumption:** Most tabular data methods assume rows are *Independent* (no systematic dependency) and *Identically Distributed* (drawn from the same distribution). Violations (e.g., time series, repeated athletes) break standard ML assumptions.
- **Data Structure Levels:** *Structured* (schema known, e.g. databases), *Semi-structured* (mixed, e.g. JSON/spreadsheets), *Unstructured* (free text, images, logs).
- **Study Types:** *Controlled experiments* (manipulate independent variables, random assignment, measure causal effect); *Quasi-experiments* (no random assignment, causal claims weaker); *Survey research* (cross-sectional snapshot, cohort/longitudinal, case-controlled).

## Key Algorithms & Techniques
- **Web Crawling:** A crawler starts from seed URLs, downloads pages, extracts links, and repeats (BFS using a queue or DFS using a stack). *Batch crawlers* take a snapshot; *incremental crawlers* revisit for freshness; *focused/topical crawlers* use classifiers to predict relevance before fetching.
- **Wrapper Induction (Information Extraction):** Learns landmark tokens that delimit target fields in template-generated HTML pages. *Co-training* reduces manual labelling by learning forward and backward landmarks simultaneously; disagreement signals new training candidates.
- **Focused Crawler Cues:** Relevance is predicted via (1) *lexical* similarity (cosine similarity between page text) and (2) *link topology* — cluster hypothesis and link-content conjecture (linked pages are lexically/semantically related).
- **Surrogate Modelling:** A slow but precise simulation (e.g., finite element method) generates a dataset; a faster ML model is trained on it as an approximation.

## Important Formulas & Diagrams
- **Data Processing Pipeline:** `Nature (Patterns) → Data Collection → Dataset → PreProcessing → Features → Data Mining → Patterns`. Information lost at collection can never be recovered later.
- **Causal Inference Dependency:** Observational data alone cannot (generally) support causal claims — interventional data or special designs (regression discontinuity, difference-in-differences) are required.
- **Harvest Rate & Search Length:** Metrics for crawler evaluation — *harvest rate* = percentage of good/relevant pages; *search length* = number of pages to crawl before finding a given fraction of relevant pages.

## Learning Objectives
- Distinguish data collection types and understand how the collection process limits downstream inference (causal vs. correlational).
- Recognise I.I.D. violations and their consequences for model validity.
- Describe the web crawling pipeline, compare crawling strategies (BFS/DFS, batch/incremental, focused/topical), and evaluate crawlers via harvest rate / search length.
- Explain wrapper induction and the role of landmarks, co-training, and active learning for web information extraction.
- Design or critique study types (controlled experiment, quasi-experiment, survey) and apply questionnaire design guidelines (avoid leading questions, double-barrelled items, social desirability bias).
- Motivate when and how to generate synthetic datasets (augmentation, surrogate modelling) and why documenting dataset provenance matters to avoid "documentation debt."
