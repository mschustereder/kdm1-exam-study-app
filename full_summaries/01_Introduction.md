# 01 Introduction — Full Summary

**Course:** Knowledge Discovery and Data Mining 1 (INP.31101UF / INP.31202UF)
**Lecturer:** Roman Kern <rkern@tugraz.at>, Institute for Machine Learning and Neural Computation — Version 2.3.0
**Goal:** Provide the key elements of working with data and extracting valuable information from it.

---

## 1. Motivation & Introduction

### 1.1 Course Goals
The overall goal of KDDM1 (and related courses) is to learn how to **discover patterns in data** and **model data**. Patterns we aim for must satisfy four criteria:
- **Valid** — hold for new data with high probability (generalisability).
- **Useful** — we can base further actions on them (actionable insight).
- **Unexpected** — non-obvious, novel findings.
- **Understandable** — interpretable by humans.

### 1.2 Data Science Origins
- Popularised by **Peter Naur** in *"Concise Survey of Computer Methods"* (1974).
- Core principle: *"The data representation must be chosen with due regard to the transformation to be achieved and the data processing tools available."* This stresses the interplay between data characteristics and tool selection.
- Goes beyond classical statistics; requires skills from multiple disciplines.

### 1.3 Knowledge Discovery (KDD)
- **Definition** (Fayyad, 1996): *"Knowledge Discovery in Databases is the non-trivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns in data."*
- The KDD process is a **sequence of steps** starting from raw data and culminating in knowledge.
- Early terminology used "from databases" to emphasise large-scale data (now called Big Data).
- The process is **cyclic** (iterative refinement), not strictly linear.

### 1.4 CRISP-DM (Cross-Industry Standard Process for Data Mining)
- Proposed by Wirth & Hipp (2000); has a stronger **business-orientation** than KDD.
- Stages: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.
- KDDM1 does **not** cover Business Understanding but does cover **Data Collection** (often missing in the original KDD process).

### 1.5 Comparison of Approaches (Kurgan & Musilek, 2006)
- KDD (academic focus), CRISP-DM (business focus), and other process models each emphasise different phases.
- KDDM1 aims to cover the main aspects from all methods.

### 1.6 Data Processing Pipeline & the Two-Body Problem
- **Upper row (reality):** Nature (Patterns) → Data Collection → Dataset.
- **Lower row (our process):** Dataset → Pre-Processing → Features → Data Mining → Patterns → Knowledge.
- **Key insight:** Information can only be **lost**, never gained, during transformation (Data Processing Inequality — covered in a later lecture).

### 1.7 Core Assumptions
- Data collection is **destructive** — information (e.g., causality) is lost.
- We must make assumptions about the data generation process and the data itself.
- **KDDM1 makes the IID Assumption:**
  1. **Independent** — each data point has no dependency on others.
  2. **Identically Distributed** — probabilities do not change across the dataset.

---

## 2. Historic Example — 1854 Broad Street Cholera Outbreak

### 2.1 Background
- Part of the 1846–1860 cholera pandemic; millions of deaths worldwide.
- In historical London, no proper sewer system — waste dumped into rivers. Water companies drew from the Thames at different sites with varying filtration.
- Cholera was not understood: competing theories were **Miasma** (bad air, the common theory) and **Germs** (theorised by John Snow). This illustrates a lack of "business understanding" / domain knowledge.

### 2.2 Data Collection
- John Snow investigated a severe 1854 outbreak by **collecting data** — talking to residents and recording deaths.
- He created a **dot map** marking water pump locations (blue dots) and death locations (red dots), with dot size encoding death count.

### 2.3 Exploratory Data Analysis (Visual)
- Visual patterns emerged: **clusters** of death in certain areas, and **anomalies** (areas with fewer deaths than expected).

### 2.4 Pre-Processing
- Snow mapped each death to the **closest water pump** using "walking distance" — effectively a manual **clustering** step (analogous to k-means).

### 2.5 Analysis & Intervention
- One cluster centred on the **Broad Street water pump**.
- Intervention: removal of the pump handle — the death count dropped immediately.
- An anomaly (fewer deaths near a brewery) was explained: workers predominantly drank **beer** (heat-treated, safer) instead of water.

### 2.6 Aggregated Analysis
- Further analysis found a statistical association between the water company supplying a pump and its death count.
- Other confounding factors (age, wealth, etc.) were ruled out.
- This resembles a **double-blind study** design.

### 2.7 Recap against Pattern Criteria
- **Valid** — confirmed by the drop in cases after intervention.
- **Useful** — lives were saved.
- **Unexpected** — contradicted the mainstream miasma theory.
- **Understandable** — with modern domain knowledge, the result is clear.
- **Key lesson:** Today's tools and data are far more advanced, but the fundamental logic remains the same.

---

## 3. Theoretical Part (VO) — Course Organisation

### 3.1 Lecturers & Communication
- **Roman Kern** (rkern@tugraz.at), **Denis Helic** (dhelic@tugraz.at), **Ratko Savic** (ratko.savic@tugraz.at).
- Use **[KDDM1]** in email subject lines.
- Default language is **English** (materials, homework, etc.), though German communication is also possible.

### 3.2 Lecture Schedule
- Mondays 12:00–14:00, c.t., HS i12.
- Special sessions: Introduction (02.03.), First Q&A (27.04.), Second Q&A (08.06.).

### 3.3 Course Outline (Lecture Topics)
The semester covers the following major blocks:
1. **Dataset Collection** — Web crawling, databases, surveys.
2. **Visual Data Science** — IQR, QQ-plots, etc.
3. **Statistical Data Science** — Correlation, distribution assumptions.
4. **Pre-Processing** — Feature extraction, feature engineering, outlier detection, missing values, dataset augmentation.
5. **Unsupervised Methods** — Dimensionality reduction, clustering.
6. **Supervised Methods** — Prediction (classification & regression).
7. **Interactive Systems**.
8. **Pattern Mining**.
9. **Evaluation**.
10. **Special Topics** — Class imbalance, AutoML, XAI (Explainable AI), and an invited talk.

### 3.4 Examination via Homework (Optional)
- Homework is an optional examination path. Submitting **Homework 1** signals participation (no separate exam registration needed).
- Questions posted as PDF roughly one month before each deadline; answers submitted as PDF.
- **Schedule:** Homework 1 (mid-April → mid-May); Homework 2 (end-May → end of June).

---

## 4. Practical Part (KU) — Project Organisation

### 4.1 Study Assistants & Support
- **Theresa Doppelhofer, Syeda Faria Ahmed, Raphael Rogi, Wilfried Suss**.
- Contact via TeachCenter forum or email with [KDDM1] in the subject.
- DataScience Discord channel available.

### 4.2 Project: Ice Hockey Talent Scouting
- **Goal:** Work for an Austrian ice hockey team; predict how likely a player will be chosen for the next season.
- **Dataset** includes player statistics, coach notes, and further player details — containing different variable types, outliers, missing values, etc.
- Students must evaluate their own models, simulating a real data science task.

### 4.3 Project Phases
- **Phase I** (deadline 03.05): Focus on **pre-processing and EDA**. The target variable is released **after** Phase I deadline. Deliverables: a presentation of the most interesting findings + source code.
- **Phase II** (deadline 19.06): Focus on **model training and evaluation**. Deliverables: presentation covering the training process, evaluation, and results + source code.
- Q&A sessions before each deadline (27.04 & 08.06).
- Group registration by 31.03.; groups of 4 students.

### 4.4 Submission & Interview
- Files required: `Phase_one_group_X.pdf` (presentation), `Phase_one_group_X.zip` (code); same pattern for Phase II.
- Upload to TeachCenter.
- **Submission interview** (in person): 10-minute presentation (point deduction if over time) + 20-minute Q&A on the project and applied concepts.

### 4.5 Project Philosophy
- The goal is **not only to optimise prediction performance**.
- The true aim is to **understand** the data, how it was generated, and the implications of pre-processing / feature engineering on results.

---

## 5. Key Takeaways

1. The KDD process is **cyclic** and comprises data collection, pre-processing, mining, and interpretation.
2. **Assumptions matter** — the IID assumption underpins most of this course.
3. Information loss is inherent in every data transformation step.
4. Historical examples (Snow's cholera map) pre-figured modern data science practices: data collection, visual EDA, clustering, hypothesis testing, and intervention.
5. The course balances **theory** (VO lectures covering the full pipeline) with **hands-on practice** (KU project simulating a real-world data science task).
6. Patterns must be **valid, useful, unexpected, and understandable** to constitute real knowledge.

---

*Generated from: KDDM1 Lecture 01 — Introduction (Version 2.3.0), Roman Kern, TU Graz.*
