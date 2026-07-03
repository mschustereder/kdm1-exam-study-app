# KDDM1 - Pattern Mining (Lecture 09)

**Authors:** Roman Kern & Denis Helic, Institute for Interactive Systems and Data Science, TU Graz  
**Version:** 2.2.0  
**Pages:** 42  

---

## 1. Introduction — What & Why

### Motivation and KDD Context
Knowledge Discovery in Databases (KDD) is defined as the *nontrivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns in data*. Previous lectures covered preprocessing, classification, clustering, and regression, but did not directly investigate the mining of the patterns themselves. Pattern mining fills this gap by directly searching for recurring structures and associations within data.

### The Supermarket / Market-Basket Example
The canonical example is the analysis of supermarket transaction data. Many grocery chains use loyalty cards, which record every purchase at the individual transaction level. This data enables:
- **Discovery of co-purchased items:** Which items tend to appear together in a basket.
- **Hidden conditional relationships:** The probability that a customer buys item B given that they have already bought item A.
- **Temporal patterns:** Trends over time (e.g., day-of-week effects).

A famous (though debated) real-world finding: a US Midwest grocery chain observed that on Thursdays, men who bought diapers were also likely to buy beer. This led to the association rule {diapers -> beer}. The actionable insight: place diapers and beer near each other and avoid running promotions on both on Thursdays.

### Transaction Data Model
- **Items:** I = {I1, I2, ..., In} — the set of all available items (the full inventory).
- **Transactions:** D = {t1, t2, ..., tm} — each transaction is a set of items (tx subset of I).
- This is often called **horizontal data** format. Alternatively, each item can be seen as a binary feature, making each transaction a feature vector with 1s for items present and 0s for absent items.

### Itemsets and Association Rules — Formal Definitions
- **Itemset:** X = {I1, I2, ..., Ii} — any subset of I. A **frequent itemset** is one found in a sufficiently large fraction of transactions.
- **Association Rule:** X -> Y, where X and Y are disjoint itemsets.
  - X is the **premise**, **antecedent**, or **left-hand side (LHS)**.
  - Y is the **conclusion**, **consequent**, or **right-hand side (RHS)**.
  - A **frequent association rule** is one that holds for many transactions.

### Support — Measuring Frequency

**Support of an itemset** is the probability of finding the itemset in the data:
Support(X) = P(X) = |{t in D | X subset of t}| / |D|
This is a relative frequency (though some algorithms use absolute counts). The underlying probability is estimated via maximum likelihood from the data.

**Support of an association rule** is the joint probability of the full combined itemset:
Support(X -> Y) = P(X union Y) = |{t in D | X union Y subset of t}| / |D|
This is identical to the support of the itemset X union Y.

### Confidence — Measuring Relevance
Confidence measures how often the conclusion Y appears when the premise X is present:
Conf(X -> Y) = P(Y | X) = Support(X union Y) / Support(X)
It reflects the conditional probability of finding Y in transactions that already contain X.

### The Goal of Association Rule Mining
Find all frequent, relevant association rules — i.e., all rules that exceed a user-specified **minimum support threshold** and **minimum confidence threshold**. Applications include:
- Market basket analysis
- Query completion / search suggestions
- Clickstream analysis
- Genome analysis
- Preprocessing for other data mining tasks (e.g., selecting the most frequent features from a dataset)

### The Brute-Force Approach and Its Problem
For n items, there are 2^n possible itemset combinations. Testing all of them is computationally infeasible. Two major improvements are presented:
1. **Apriori Algorithm** — uses a monotonicity property to prune the search space.
2. **FP-Growth Algorithm** — uses a compact tree data structure to avoid candidate generation entirely.

---

## 2. Apriori Algorithm

### The Apriori Principle
The key insight (the Apriori principle) is a monotonicity property:
- **If an itemset is frequent, then all of its subsets are also frequent.**
- **If an itemset is infrequent, then all of its supersets are also infrequent.**

The contrapositive is the one used for pruning: if a 2-itemset is infrequent, any larger itemset containing it will also be infrequent, so it can be discarded without evaluation.

### Apriori Algorithm for Frequent Itemsets
The algorithm works level-by-level (breadth-first), generating itemsets of increasing size:

1. **Step 1 (k=1):** Scan the database. Count the support of all single items. Retain those with support >= minsup as F1 (frequent 1-itemsets).
2. **Step 2 (k = k+1):** Increment the itemset size.
3. **Candidate generation:** Generate candidate itemsets Ck of size k by joining pairs of frequent itemsets from F(k-1). Two itemsets of size k-1 are joined if they share their first k-2 items.
4. **Prune:** Remove any candidate in Ck that has a (k-1)-subset not present in F(k-1) (the Apriori principle).
5. **Support counting:** Scan the database once more. For each candidate in Ck, count its support. Move candidates with support >= minsup to Fk; discard the rest.
6. **Repeat** from step 2 until no new frequent itemsets are found (Fk is empty).

This reduces the candidate space dramatically: infrequent small itemsets never get expanded into larger ones.

### Apriori Algorithm for Frequent Association Rules
Once all frequent itemsets are known, association rules are generated from them (not from the raw transactions, further reducing the search space):

1. **Start** with frequent itemsets of size k >= 2.
2. **Initial conclusions (m=1):** Generate candidate rules where the conclusion set H1 contains exactly one item, taken from the frequent itemset.
3. **Increment** m = m + 1.
4. **Generate new candidate conclusions** of size m by merging two rules from the previous step:
   - New premise = intersection of the two old premises.
   - New conclusion = union of the two old conclusions.
   - This effectively moves one item from premise to conclusion.
5. **Prune** using the Apriori principle: discard any rule whose conclusion has an infrequent subset.
6. **Confidence checking:** For each candidate, compute confidence. Retain those with confidence >= minconf.
7. **Repeat** from step 3 until no new candidate conclusions are generated.

A frequent itemset of size k yields (2^k - 2) possible association rules (all non-empty, non-full subsets).

### Monotonicity of Confidence
Confidence is monotonically non-decreasing as items are moved from the premise to the conclusion *within the same itemset*:
Conf(ABC -> D) >= Conf(AB -> CD) >= Conf(A -> BCD)
However, this monotonicity does **not** hold in general across different itemsets: Conf(ABC -> D) can be larger or smaller than Conf(AB -> D).

---

## 3. FP-Growth Algorithm

### Motivation and Overview
FP-Growth (Frequent Pattern Growth) takes a completely different approach from Apriori:
- It uses a **divide-and-conquer** strategy.
- It achieves **lower complexity by several orders of magnitude**.
- It requires **only two full database scans** (versus Apriori's k scans for k-itemsets).

### Main Idea
1. **Build** a compact data structure called the **FP-Tree** (Frequent Pattern Tree) that compresses the transaction database.
2. **Mine** the FP-Tree recursively by constructing **conditional FP-Trees** on the fly.

### Structure of the FP-Tree
- **Tree component:** Each node represents an item and carries a count of how many transactions share the path from the root to that node.
- **Header table / linked list:** For each frequent item, a header entry stores the total count and a pointer to a linked list connecting all nodes in the tree that contain that item.
- The tree preserves **all information** from the original transactions in a compressed form.

### Building the FP-Tree
1. **First scan:** Count the support of every item. Remove infrequent items (count < minsup). Sort frequent items in **descending order** of support (this maximizes the chance of prefix sharing).
2. **Second scan:** For each transaction:
   - Sort the items by the same descending support order.
   - Remove any infrequent items.
   - Insert the sorted items into the tree:
     - Start at the root.
     - For each item, check if it already exists as a child of the current node. If yes, increment that child's count. If no, create a new child node with count = 1.
   - Update the header-linked lists to connect all nodes of the same item.

### Mining the FP-Tree
The mining proceeds recursively, processing items in **increasing order of support** (least frequent first):

1. **Select** a single item (starting with the least frequent) as a suffix.
2. **Construct the conditional FP-Tree** for that item:
   - Follow the item's linked list to collect all paths that end at this item.
   - For each such path, compute the **prefix path** (nodes above the item node).
   - The count of the item node becomes the count of the prefix path in the conditional tree.
   - Remove any items from these prefix paths whose total count falls below the support threshold.
3. **Recursively mine** the conditional FP-Tree:
   - If the conditional tree is a **single path** (a chain), then **every combination** of items in that chain is a frequent itemset (in combination with the original suffix).
   - If the conditional tree has **multiple branches**, recurse further by picking the next (least frequent) item from the conditional tree, building a deeper conditional tree, and repeating.
4. **Repeat** for all items in the original header table until all frequent itemsets are discovered.

Because the FP-Tree compresses the data and the algorithm avoids explicit candidate generation, it is typically orders of magnitude faster than Apriori for dense datasets.

---

## 4. Evaluation & Tools

### The Challenge of Interestingness
Pattern mining typically produces a very large number of frequent itemsets and association rules. The critical question becomes: which of these are truly **interesting**, and which are **spurious**?

### Limitations of Support and Confidence
Both support and confidence assume a baseline of statistical independence, which can be misleading. The classic example is the **Tea-Coffee paradox**:

Contingency table for Tea and Coffee (N = 100):

|            | Coffee | not Coffee | Row Total |
|------------|--------|------------|-----------|
| **Tea**    | 15     | 5          | 20        |
| **not Tea**| 75     | 5          | 80        |
| **Col Tot**| 90     | 10         | 100       |

Consider the rule **Tea -> Coffee**:
Conf(Tea -> Coffee) = 15/20 = 0.75

At 75% confidence, this seems like a strong rule. However, the overall probability of Coffee (regardless of Tea) is:
P(Coffee) = 90/100 = 0.90

Thus, drinking tea actually **reduces** the probability of drinking coffee from 90% to 75%. The rule is misleading because confidence (75%) is lower than the baseline probability of Coffee (90%).

### General Rule for Avoiding Misleading Rules
A rule X -> Y should only be considered interesting if:
Conf(X -> Y) > Support(Y)
Otherwise, having X in the transaction actually makes Y **less** likely than its unconditional occurrence.

### Lift and Interest Factor
**Lift** addresses this by comparing the observed confidence against the baseline:
Lift(X -> Y) = P(Y | X) / P(Y) = Conf(X -> Y) / Support(Y)

**Interest** is the equivalent measure for itemsets:
Interest(X, Y) = P(X, Y) / (P(X) * P(Y))

Also called the **Interest Factor (IF)** or just **Interest (I)**.

**Interpretation:**
- **Lift = 1:** X and Y are statistically independent — the rule is not interesting.
- **Lift > 1:** Positive relationship — X and Y appear together more often than expected by chance.
- **Lift < 1:** Negative relationship — X and Y appear together less often than expected by chance (as in the Tea-Coffee example, where Lift = 0.75 / 0.9 = 0.83).

### Contingency Table Representation
For any pair of itemsets X and Y, the relationship can be summarized in a 2x2 contingency table:

|        | Y       | not Y   | Row Sum |
|--------|---------|---------|---------|
| **X**  | f11     | f10     | f1+     |
| **not X**| f01   | f00     | f0+     |
| **Col** | f+1    | f+0     | N       |

Where:
- f11 = Support count of X and Y together.
- f10 = Support count of X and not Y.
- f01 = Support count of not X and Y.
- f00 = Support count of neither X nor Y.
- Row/column sums are marginal counts; N is the total number of transactions.

All evaluation measures (support, confidence, lift) can be computed directly from these four cells.

### Recommended Literature and Tools

**Primary textbook:**  
P-N. Tan, M. Steinbach, A. Karpatne, V. Kumar — *Introduction to Data Mining*, 2nd Edition, Pearson 2018, Chapter 5: "Association Analysis: Basic Concepts and Algorithms".

**Recommended slides:**  
Christian Borgelt's FPM slides (516 slides) at http://www.borgelt.net/teach/fpm/slides.html

**Software tools:**

| Language   | Tool / Library   | URL / Notes |
|------------|------------------|-------------|
| **Java**   | SPMF             | http://www.philippe-fournier-viger.com/spmf/ |
| **Python** | PyFIM            | http://www.borgelt.net/pyfim.html |
| **Python** | PrefixSpan       | https://pypi.org/project/prefixspan/ |
| **R**      | arules           | https://cran.r-project.org/web/packages/arules/ |
| **R**      | arulesSequences  | https://cran.r-project.org/web/packages/arulesSequences/ |
| **Spark**  | Spark ML FPM     | https://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html |

---

## Summary of Key Concepts

| Concept              | Definition |
|----------------------|------------|
| **Itemset**          | A set of items X subset of I. |
| **Support**          | Relative frequency of an itemset/rule in the data. |
| **Confidence**       | Conditional probability P(Y | X) for a rule X -> Y. |
| **Apriori Principle**| Subsets of frequent itemsets are frequent; supersets of infrequent itemsets are infrequent. |
| **Apriori Algorithm**| Level-wise candidate generation and pruning using the Apriori principle. |
| **FP-Tree**          | Compact tree structure storing all transaction info with shared prefixes, plus a header table with linked lists per item. |
| **FP-Growth**        | Builds FP-Tree in two scans, mines recursively via conditional FP-Trees. |
| **Lift / Interest**  | Measures deviation from independence; Lift = 1 means independence. |
| **Contingency Table**| 2x2 matrix of co-occurrence counts used to compute all evaluation measures. |
