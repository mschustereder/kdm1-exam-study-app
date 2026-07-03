# Old Exam — KDM1 (2024-06-25, Group B)

**Course:** KDDM1 (INP.31101UF)

---

## Question 1 — Dataset Collection

The way data has been collected has a profound impact on succeeding processing steps. Here we assume tabular data (e.g., features as columns, instances as rows) and we want to use a dataset for classification, i.e., one of the features is the target feature which we want to learn to predict.

**a)** Why is it important to test for dependencies between the features?

**b)** What are the implications if there many features are highly correlated?

**c)** What are the implications of the number of features? How does the number influence the ability to predict the target variable?

---

## Question 2 — Correlation

There are multiple different correlation measures.

**a)** Please list a number of common correlation measures? (3–5 measures)

**b)** Pick two correlation measures and list their main assumptions about the data and the dependencies in the data.

**c)** Which correlation measures allow to detect causal relationships in the data?

---

## Question 3 — Missing Values

Missing data may have profound impact on the results of the data mining steps.

**a)** What types of missing values exist and what are the implications?

**b)** Why is it important to understand why some data is missing?

**c)** What dependencies should be checked in order to inform the appropriate way to deal with missing values?

**d)** In a dataset, a feature has missing values, but there is another feature with a visible dependency. What are the implications?

---

## Question 4 — Anomalies

Data that does not conform to the notion of normality may be present in a dataset.

**a)** What role does domain knowledge play in anomaly detection?

**b)** What approach do you choose, if anomalies are characterised by having the highest Euclidean distance from the origin in feature space?

**c)** Please draw a 2D scatter plot of a dataset with anomalies/outliers (please indicate normal and anomaly behaviour), where you expect Local Outlier Factor (LOF) to work well.

---

## Question 5 — Entropy

Entropy is a measure of uncertainty in a random variable.

**a)** What is the mathematical definition of entropy?

**b)** Let X = (A, B, C, D, E), with probabilities P(A) = 1/6, P(B) = 1/6, P(C) = 1/6, P(D) = 1/6, P(E) = 1/3. Compute the entropy of X.

**c)** Give an example for probabilities that would result in the maximal entropy of X.

---

## Question 6 — Dimensionality Reduction

If there are many features we typically perform dimensionality reduction.

**a)** Why is it a good idea to reduce the number of features? Give at least three reasons.

**b)** What are typical applications where we use dimensionality reduction? Give at least two applications.

**c)** Explain the basic principle of Principal Component Analysis in a few sentences.

---

## Question 7 — Clustering

Clustering is an unsupervised method for identifying groups of similar instances.

**a)** Explain the basic objectives of clustering in a few sentences.

**b)** Give at least three types of clustering.

**c)** Explain hierarchical clustering. What hyperparameters do we need to set in this approach? What are advantages and disadvantages of hierarchical clustering?

---

## Question 8 — Classification Evaluation

Evaluation is an integral part of any data science project. We apply an e-mail spam classifier on a test dataset with 800 e-mails. The classification results are summarised in the following contingency table (with 1 representing spam class):

```
             Prediction
               1     0
Real class 1 | 70 | 40
           0 | 30 | 660
```

**a)** Calculate the accuracy (A), recall (R), precision (P) and F1 measure for the classifier. Give results in fractions.

**b)** Why do we need the F1 measure?

**c)** What other classification evaluation measures exist?
