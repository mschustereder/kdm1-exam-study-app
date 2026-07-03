# 09 — Pattern Mining

## Core Definitions & Concepts

- **KDD definition**: Pattern mining is the nontrivial process of identifying valid, novel, potentially useful, and ultimately understandable patterns in data.
- **Items & Transactions**: `I = {I1, I2, ..., In}` is the set of all items; `D = {t1, t2, ..., tm}` is the set of transactions, each being a subset of `I`.
- **Itemset**: `X = {I1, I2, ..., Ii}` — a subset of items. A *frequent itemset* appears often in transactions.
- **Association Rule**: `X → Y` — X (antecedent/premise/LHS) implies Y (consequent/conclusion/RHS). A *frequent association rule* holds for many transactions.
- **Support** (itemset): `Support(X) = P(X) = |{t in D : X subset of t}| / |D|` — relative frequency of transactions containing all items in X.
- **Support** (rule): `Support(X → Y) = P(X ∪ Y)` — same as the joint itemset's support.
- **Confidence**: `Conf(X → Y) = Support(X ∪ Y) / Support(X) = P(Y | X)` — how often Y appears in transactions that contain X.
- **Goal**: Find all frequent, relevant association rules with sufficient support and sufficient confidence.
- **Applications**: Market basket analysis, query completion, click stream analysis, genome analysis, preprocessing for other data mining tasks.

## Key Algorithms & Techniques

### Apriori Algorithm
- Based on the **Apriori principle**: if an itemset is frequent, all its subsets are frequent; if an itemset is infrequent, all its supersets are infrequent.
- **Frequent itemset mining**:
  1. Find all frequent 1-itemsets (`F1`).
  2. Increment `k` and generate candidate `Ck` from frequent `Fk-1`.
  3. Prune candidates containing infrequent sub-itemsets.
  4. Scan database to compute support; keep those above threshold.
  5. Repeat until no new frequent itemsets found.
- **Association rule mining**: Operates on frequent itemsets (not raw transactions). For a `k`-itemset, `2^k - 2` rule candidates exist. Uses confidence monotonicity (on same itemset) and the apriori principle to prune.
- Limitation: still generates many candidates; `2^n` combinations in worst case.

### FP-Growth Algorithm
- Entirely different, divide-and-conquer approach; orders of magnitude faster than Apriori.
- Requires only **two scans** of the transaction database.
- **FP-Tree** data structure: a tree where each node represents an item with a count, plus a linked list connecting nodes of the same item (header table). Encodes all transaction information compactly.
- **Build the FP-Tree**:
  1. Scan 1: count item support, discard infrequent items.
  2. Scan 2: for each transaction, sort items by descending support, insert into tree (increment count on match, branch on mismatch).
- **Mine the FP-Tree**: Starting with the least frequent item, build a **conditional FP-Tree** (project transactions containing that item). Recursively mine conditional trees. Single-branch trees yield all combinations directly.
- Avoids explicit candidate generation.

### Evaluation & Tools

- **Confidence limitations**: Confidence can be misleading — e.g., `Conf(Tea → Coffee) = 0.75` but `P(Coffee) = 0.9`, meaning tea *reduces* coffee probability.
- **Rule quality check**: `Conf(X → Y) > Support(Y)` — otherwise X reduces the chance of Y.
- **Lift** (for rules): `Lift = P(Y|X) / P(Y)`. Lift = 1 implies independence; < 1 implies negative correlation; > 1 implies positive correlation.
- **Interest** (for itemsets): `Interest = P(X,Y) / (P(X)P(Y))` — same ratio under independence baseline.
- **Contingency table**: 2x2 table (`f11`, `f10`, `f01`, `f00`) for binary items X and Y, used to compute all above measures.
- **Tools**: SPMF (Java), PyFIM & PrefixSpan (Python), arules & arulesSequences (R), Spark ML (Scala/Java/Python/R).

## Important Formulas & Diagram Types

- **Support formula**: `Support(X) = |{t in D : X ⊆ t}| / |D|`
- **Confidence formula**: `Conf(X → Y) = Support(X ∪ Y) / Support(X)`
- **Lift formula**: `Lift(X → Y) = P(Y|X) / P(Y) = Conf(X → Y) / Support(Y)`
- **Interest formula**: `Interest(X,Y) = P(X,Y) / (P(X)P(Y))`
- **Diagram types**: FP-Tree construction (tree + linked-list header table), conditional FP-Trees, contingency tables for evaluation.

## Learning Objectives

- Understand the formal definitions of itemsets, association rules, support, and confidence.
- Understand and be able to apply the Apriori principle to reduce candidate search space.
- Know the Apriori algorithm steps for both frequent itemset and association rule mining.
- Know the FP-Growth algorithm: FP-Tree construction and recursive mining of conditional FP-Trees.
- Understand limitations of confidence and how lift/interest provide a better evaluation metric.
- Be aware of available toolkits in Java, Python, R, and Spark.
