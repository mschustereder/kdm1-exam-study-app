# 01 — Introduction

## Core Definitions & Concepts

- **Knowledge Discovery & Data Mining (KDDM)**: The non-trivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns in data (Fayyad, 1996).
- **Desired pattern properties**: (i) Valid — hold for new data with high probability, (ii) Useful — can base actions on them, (iii) Unexpected — non-obvious, (iv) Understandable — interpretable by humans.
- **Data Science** (coined by Peter Naur, 1974): Data representation must be chosen with regard to the transformation needed and available processing tools.
- **KDD Process** (Fayyad, 1996): A sequence of steps from raw data to knowledge; includes cycles; this course is aligned with it.
- **CRISP-DM** (2000): Industry-standard process model adding explicit Business Understanding and Data Understanding phases; more business-oriented than the academic KDD process.
- **Data Processing Pipeline (Two-Body Problem)**: A data generation process (Nature → Data) and a knowledge discovery process (Dataset → Pre-processing → Features → Data Mining → Patterns → Knowledge). Information is only ever lost (never gained) along this chain — relates to the data processing inequality.

## Key Algorithms / Techniques Mentioned

- **IID Assumption**: The course assumes data is *Independent and Identically Distributed* — no point-to-point dependencies and the underlying probability distribution does not change within the dataset.
- **Exploratory Data Analysis (EDA)** — visual tools to uncover patterns in data (e.g., John Snow's cholera dot map plotting deaths and water pump locations).
- **Clustering / Voronoi-like assignment**: Mapping deaths to the nearest water pump via walking distance — precursor to k-means clustering.
- **Intervention / Hypothesis testing**: Removing the Broad Street pump handle confirmed the hypothesized cause; comparison of death counts across water companies ruled out confounders (age, wealth).
- **Data collection, pre-processing, feature extraction, feature engineering, outlier detection, missing value imputation, dataset augmentation** — all part of the practical pipeline.

## Important Concepts / Diagrams Referenced

- **John Snow's 1854 Broad Street cholera map** — early, canonical example of data-driven discovery: dot map overlaying death locations and water pumps, leading to the pump handle removal and a drop in cases.
- **Comparison of process models** (Kurgan & Musilek, 2006) — different approaches (KDD, CRISP-DM, etc.) each place emphasis on different stages.
- **Data Processing Inequality**: Information is irreversibly lost during the destructive data collection process (e.g., causality cannot always be recovered).

## Learning Objectives

- Understand the goals and core concepts of knowledge discovery and data science.
- Learn the key process models (KDD, CRISP-DM) and their differences.
- Familiarity with the full data pipeline: collection → pre-processing → mining → evaluation → knowledge.
- Appreciate the importance of assumptions (especially IID) and the destructive nature of data collection.
- Apply the pattern criteria (valid, useful, unexpected, understandable) to real-world findings.
- Practical component: Group project on Ice Hockey Talent Scouting — Phase I (pre-processing & EDA), Phase II (model training & evaluation), submission with presentations and source code, plus an oral interview.
