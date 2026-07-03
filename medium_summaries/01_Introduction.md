# 01 — Introduction (KDDM1)

## Learning Objectives

- Discover patterns in data that are **valid** (hold for new data), **useful** (actionable), **unexpected** (non-obvious), and **understandable** (human-interpretable).
- Understand the full knowledge discovery pipeline from data collection to pattern extraction.
- Gain competence in working with data across multiple disciplines (statistics, machine learning, visualisation).

## Core Definitions & Frameworks

- **Data Science** (Naur, 1974): Data representation must be chosen with regard to the transformations to be achieved and the tools available. Stresses the tight coupling between representation, processing goals, and tooling.
- **Knowledge Discovery in Databases (KDD)** — Fayyad (1996): The non-trivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns in data. Originally scoped to "databases" (synonymous with "Big Data" at the time).
- **KDD Process**: A multi-step sequence (selection, pre-processing, transformation, data mining, interpretation/evaluation) with feedback cycles. Critically includes data collection as an explicit step.
- **CRISP-DM** (Wirth & Hipp, 2000): A business-oriented alternative to KDD with six phases — Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment. Adds the "Business Understanding" phase that KDD omits.
- **Comparison** (Kurgan & Musilek, 2006): KDD is more academia-focused, CRISP-DM more industry-focused. KDDM1 aligns strongly with the KDD process but also covers data collection.

## Key Concepts & Assumptions

- **Data Processing Pipeline**: A two-body problem — the data generation process (Nature -> patterns -> data collection -> dataset) and the knowledge discovery process (dataset -> pre-processing -> features -> data mining -> patterns -> knowledge). Information can only be lost between the real world and the final dataset.
- **IID Assumption**: The course assumes data points are (1) **independent** (no dependencies between points) and (2) **identically distributed** (probabilities do not change within the dataset). This is a simplifying assumption; real data often violates it.

## Historic Example — 1854 Broad Street Cholera Outbreak

- **Context**: Cholera pandemic (1846-1860), millions dead. The dominant theory was "miasma" (bad air); John Snow hypothesised a germ theory via contaminated water.
- **Data Collection**: Snow interviewed locals and plotted deaths on a dot map alongside water pump locations.
- **Exploratory Data Analysis (EDA)**: Visual inspection revealed a death cluster around the Broad Street pump and an anomaly (fewer deaths) around a brewery whose workers drank beer (heat-treated water).
- **Pre-Processing**: Deaths were mapped to the nearest water pump by walking distance — a manual clustering step analogous to k-means.
- **Intervention & Validation**: Removing the Broad Street pump handle caused deaths to drop. Further analysis ruled out age/wealth confounders. The discovered pattern met all four goals: valid, useful, unexpected, understandable.

## Course Structure

- **VO (Theoretical Part)**: Lectures Mon 12:00-14:00. Topics — dataset collection, visual & statistical data science, pre-processing (feature extraction/engineering, outliers, missing values, augmentation), unsupervised methods (dimensionality reduction, clustering), supervised methods (classification, regression), pattern mining, evaluation, class imbalance, AutoML, XAI. Optional homework route for examination.
- **KU (Practical Part)**: Group project (4 students) on *Ice Hockey Talent Scouting*. Phase I: pre-processing and EDA (presentation due 03.05). Phase II: model training and evaluation (presentation due 19.06). Target variable released only after Phase I. Submission interview with 10-min presentation + 20-min discussion.
