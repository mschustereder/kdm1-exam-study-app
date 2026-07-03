# 02 — Dataset Collection — Full Summary

**Course:** KDDM1 | **Authors:** Roman Kern & Denis Helic (TU Graz) | **Version:** 2.3.0

---
## 1. Introduction
Data collection is foundational. **Data Processing Inequality:** information can only be lost, never gained through the pipeline — source data not collected is irretrievably lost. The generation process and collection method determine causal estimability (Bareinboim & Pearl, 2016).

---
## 2. Dataset Collection Types (Data Generation)
- **Observational:** Passive, no interference. Does **not** support causal inference.
- **Interventional:** Active manipulation (A/B testing, drug trials). Enables causal inference.
- **Biased:** Sampling bias, distribution shift (bias changing over time).
- **Domain Shift:** Combining datasets from different conditions — distributions differ; naive joins invalid.

---
## 3. Types of Data Sets
### 3.1 Structure: 1) **Structured** — full schema. 2) **Semi-structured** — mixed (JSON). 3) **Unstructured** — no schema (text, images, logs).
### 3.2 Tabular Data: Rows = instances, columns = features/target. **I.I.D. Assumption** (independent, identically distributed). Violated when same entity repeats (e.g., athlete injury prediction). CSV needs careful parsing.
### 3.3 Time Series: Not I.I.D. — **autoregressive** (past predicts future). **Univariate** vs. **Multivariate**.
### 3.4 Other: **Image** (spatial autocorrelation), **Graph** (nodes/edges, homophily), **Text**, **Spatial**.

---
## 4. Existing Datasets
**Internal:** SQL/Cypher/SPARQL queries. **External:** common for "big data"; check compatibility. **Repositories:** UCI ML Repository, UEA/UCR Time Series, Zenodo, Data (MDPI). **Public:** Awesome Public Datasets (GitHub), Kaggle, Open Governmental Data (data.europa.eu — 1.5M+ datasets).

---
## 5. Web Crawling
**Crawler:** HTTP client that systematically downloads pages (googlebot; frameworks: Scrapy, Heritrix). **Schema:** URL → download → analyse → extract links → **frontier** (queue) → repeat.
**Types:** **Batch** (snapshot), **Incremental** (revisits), **Specialised** (focused/topical). **Challenges:** volume, dynamic pages, duplicate detection (URL normalisation), redirect loops, memory.
**Structured info:** Semantic Web (RDF), **Microformats** (e.g., `class="geo"` — 4.1B+ OSM nodes).
**Strategies:** **BFS** (queue/FIFO), **DFS** (stack/LIFO), **Concurrent** (distributed), **Deep** (hidden web forms), **Topical** (find similar pages), **Focused** (classifier-based). Conjectures: **link-content** (lexical similarity decays with link distance), **link-cluster** (semantic relatedness decays). **Evaluation:** **Harvest rate** (% relevant), **Search length** (pages to target relevance).
**Wrapper Generation** — extract structured data from HTML templates. Approaches: **Manual**, **Wrapper Induction** (supervised; landmarks + active learning/co-training), **Automatic** (unsupervised).
**Ethics:** avoid DoS-like requests; respect **robots.txt**; check T&C and personal data.

---
## 6. Types of Studies
**Controlled Experiment:** Hypothesis-driven; manipulate IVs → measure DVs. Designs: **Between-subjects** (random groups), **Within-subjects** (all treatments; risk: learning effect), **Quasi-experiments** (no random assignment; RDD, DiD).
**Survey Research:** Large-sample questionnaires. **Cross-sectional** (snapshot), **Case-controlled** (correlations), **Cohort** (longitudinal). 9 rules: avoid leading, social desirability, double-barrelled, long, negations, irrelevant, poor options, big/ambiguous words.

---
## 7. Synthetic Datasets
**Why:** ethical/legal unavailability, controlled testing, size impact. **Augmentation:** label-preserving variants (rotation, scaling) → improves robustness. **Surrogate Modelling:** expensive simulation → fast ML approximation (less precise, faster).
**Documentation Debt** (Bender et al., 2021): undocumented large datasets perpetuate harm. Domain knowledge (constraints) should be formally documented (knowledge graphs).

---
## Key References
1. Bareinboim & Pearl (2016) — Causal inference and data-fusion. *PNAS* 113(27).
2. Bender et al. (2021) — Stochastic parrots. *FAccT*.
3. Lovdal et al. (2021) — Injury prediction in runners. *Int. J. Sports Physiol. Perform.* 16(10).
4. Singh & Bedathur (2023) — Embeddings for tabular data. *arXiv:2302.11777*.
