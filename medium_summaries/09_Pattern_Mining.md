# 09 Pattern Mining — Medium Summary

## Definitions & Foundations

- **Frequent Itemset**: A set of items (e.g., {beer, diapers}) that appears together in many transactions, measured by **support**.
- **Association Rule** `X -> Y`: A rule stating that if itemset X (antecedent/LHS) occurs, itemset Y (consequent/RHS) is likely to also occur. Used for market basket analysis, query completion, click streams, genome analysis.
- **Support of an itemset**: `Support(X) = P(X) = |{t in D: X subset of t}| / |D|` — relative frequency of transactions containing all items in X.
- **Support of a rule**: `Support(X -> Y) = P(X union Y)` — same as support of the joint itemset.
- **Confidence of a rule**: `Conf(X -> Y) = Support(X union Y) / Support(X) = P(Y | X)` — probability of finding Y in transactions that contain X.
- **Goal**: Find all frequent, relevant association rules with sufficient support and sufficient confidence.

## Key Algorithm: Apriori

- **Apriori Principle**: If an itemset is frequent, all its subsets are frequent. Conversely, if an itemset is infrequent, all its supersets are infrequent. This drastically prunes the candidate space (from 2^n).
- **Frequent Itemset Mining (Apriori)**: (1) Find frequent 1-itemsets. (2) Increment k. (3) Generate candidates C_k from frequent F_(k-1). (4) Prune candidates with infrequent subsets. (5) Scan DB to compute support; keep those above threshold. Repeat until no new frequent itemsets.
- **Frequent Association Rule Mining**: Operates on the frequent itemsets (not raw transactions). For each frequent itemset of size k>=2, generate candidate conclusions of size m=1, compute confidence, prune, increment m, merge rules (one item moves from premise to conclusion), and recurse.
- **Candidate count**: A k-itemset yields 2^k - 2 possible rules (all non-empty proper subsets). Confidence is monotonically decreasing across rules derived from the same itemset (e.g., `Conf(ABC->D) >= Conf(AB->CD) >= Conf(A->BCD)`), but not across different itemsets.

## Key Algorithm: FP-Growth

- **Main idea**: A divide-and-conquer approach that transforms transactions into a compact **FP-Tree**, reducing database scans to just two (vs. repeated scans in Apriori), and is orders of magnitude faster.
- **Building the FP-Tree**: (1) Scan DB, count item frequencies, discard infrequent items. (2) Sort each transaction's items by descending frequency. (3) Insert each transaction into a tree: match existing paths, increment counts, or branch out for new items. A header table links all occurrences of each frequent item.
- **Mining the FP-Tree**: Start with the least frequent item. Build a **conditional FP-Tree** (projected sub-tree containing only transactions with that item). If the conditional tree is a single path, all item combinations are frequent. If it branches, recurse by selecting the next least frequent item. Repeat until no items remain.

## Evaluation & Metrics

- **Contingency table**: A 2x2 matrix (X vs. not-X, Y vs. not-Y) summarising co-occurrence counts (`f11`, `f10`, `f01`, `f00`) — the foundation for computing association measures.
- **Confidence pitfalls**: Confidence can be misleading. Example: `Conf(Tea -> Coffee) = 0.75`, but `P(Coffee) = 0.9` — tea actually reduces coffee likelihood. Always compare confidence against the baseline probability of the conclusion.
- **Lift** (for rules) and **Interest** (for itemsets): `Lift(X->Y) = P(Y|X) / P(Y)`. Lift = 1 means independence. Lift < 1 means a negative relationship (X reduces Y); Lift > 1 means positive association.
- **Rule quality heuristic**: A good rule satisfies `Conf(X->Y) > Support(Y)`, ensuring the premise genuinely raises the probability of the conclusion.

## Learning Objectives

- Understand the formal definition of itemsets, association rules, support, and confidence.
- Explain and apply the Apriori principle to prune the candidate search space.
- Describe the Apriori algorithm for both frequent itemset and association rule mining, including candidate generation and pruning steps.
- Understand the FP-Growth algorithm: building the FP-Tree and recursively mining conditional FP-Trees.
- Recognise the limitations of support and confidence, and know how Lift/Interest corrects for baseline probabilities.
- Be able to interpret contingency tables and identify spurious rules.
