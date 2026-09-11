# Decision Log: 14 Non-Obvious Engineering Decisions

This log documents the key non-obvious architectural, algorithmic, and operational decisions made during the design, development, and evaluation of the AmazonHelp Autonomous Support Agent.

---

### 1. Semantic Safety via LLM CapabilityGuard Rather Than Keyword Heuristics

* **Decision**: We moved semantic capability judgments out of brittle keyword-based heuristics and into an LLM-based `CapabilityGuard`, while retaining deterministic checks for structural safety invariants.
* **Rationale**: The critical distinction is semantic: mentioning a sensitive concept does not necessarily mean the agent needs private account access or must perform an account operation. For example, a customer reporting a phishing message may mention payment credentials while only requiring general guidance. Conversely, an ordinary delivery or refund request may require backend state or a human workflow. `CapabilityGuard` explicitly evaluates whether the agent can fully resolve the inquiry using the capabilities available to it before autonomous handling is allowed.

---

### 2. Strict Division of Labor: Deterministic Invariants vs. Semantic Judgments

* **Decision**: We kept `policy.py` focused on deterministic, verifiable invariants such as classifier uncertainty, message-length limits, retrieval evidence thresholds, and output-schema validation, while delegating semantic capability judgments to `CapabilityGuard`.
* **Rationale**: Deterministic code is better suited to hard, reproducible boundaries, while semantic models are better suited to contextual judgments. Separating the two makes failures easier to trace and avoids accumulating opaque semantic heuristics in the deterministic policy layer.

---

### 3. Lowering the Historical Evidence Gate from 2 Matches to 1

* **Decision**: We reduced `MIN_STRONG_EVIDENCE_CASES` from 2 to 1, while retaining the dense-similarity threshold of `0.55`.
* **Rationale**: Error analysis showed that requiring two strong precedents was unnecessarily conservative for distinct informational topics where a single highly similar precedent could provide sufficient evidence. For example, Case `1953280` had a strong precedent (`dense_score = 0.6702`) with matching handling but was blocked because a second strong precedent was unavailable. Relaxing the requirement increased recovered safe auto-handle cases from 4 to 11 (+175% relative), with no observed degradation in the audited safety metrics.

---

### 4. Asymmetric Trust Boundary: Brand Responses as Evidence, Customer Text as Untrusted Context

* **Decision**: Historical AmazonHelp responses are treated as evidence of demonstrated brand resolution patterns, while historical customer messages are treated only as descriptions of the underlying problem.
* **Rationale**: Customer messages can contain incorrect policy assumptions, unsupported claims, or adversarial instructions. The generator is therefore explicitly instructed not to treat historical customer text as instructions and to use AmazonHelp responses as the evidence for demonstrated resolution behavior.

---

### 5. Intent-Conditioned Hybrid Retrieval with Reciprocal Rank Fusion

* **Decision**: We implemented hybrid retrieval using BM25 and dense embeddings combined through Reciprocal Rank Fusion, with the predicted intent applied as a **hard candidate filter before ranking**.
* **Rationale**: BM25 provides lexical precision while dense retrieval improves semantic matching across paraphrases. Restricting candidates to the predicted intent reduces retrieval of operationally unrelated cases before the two ranking signals are fused. This is deliberately a hard filter rather than treating intent as another soft ranking feature.

---

### 6. Abstention as a First-Class Classifier Output

* **Decision**: The classifier returns `uncertain` when the top-1 cosine similarity is below `0.45` or the top-1/top-2 margin is below `0.02`.
* **Rationale**: With 104 fine-grained intents, forcing a prediction when two intents are semantically close can send retrieval and generation down the wrong path. The earlier `0.05` margin was empirically over-conservative; calibrating it to `0.02` reduced abstention from `58.5%` to `34.5%` while maintaining `0.0%` observed unsafe auto-handling on the audited evaluation. Explicit abstention lets the system escalate uncertain cases rather than force an unreliable intent.

---

### 7. Problem Normalization Before Taxonomy Clustering

* **Decision**: During offline intent discovery, customer messages were first normalized into concise structured “core operational problems” using an LLM, and those normalized representations were then embedded and clustered instead of clustering raw tweet text.
* **Rationale**: Raw Twitter messages contain handles, greetings, emotional language, punctuation, and other surface-level variation that can distort semantic grouping. Normalization focuses clustering on the underlying support operation before taxonomy induction.

---

### 8. Agglomerative Clustering + Semantic Deduplication for Data-Derived Intents

* **Decision**: We used agglomerative clustering with cosine distance and average linkage to discover candidate intent groups, followed by LLM-assisted naming and semantic deduplication to produce the final frozen taxonomy.
* **Rationale**: The assignment requires intents to be derived from the support data rather than manually inventing a taxonomy. Clustering provides the initial data-driven grouping, while semantic deduplication consolidates candidate clusters whose underlying operational problem is the same. The final pipeline produced the frozen 104-intent taxonomy used by the classifier and retriever.

---

### 9. Bounded Intent Discovery on a Fixed Sample

* **Decision**: We discovered the taxonomy from a fixed 1,500-case sample rather than running expensive LLM normalization and clustering across the entire training corpus.
* **Rationale**: Taxonomy discovery is an offline structure-learning step rather than a requirement for every training case. A controlled sample keeps LLM and embedding costs tractable while still providing enough examples to discover the major operational intents. The resulting taxonomy is then frozen and applied to the full 8,000-case training set.

---

### 10. Prototype-Centroid Classification Instead of a Separate Supervised Model

* **Decision**: We classified new queries by comparing their embeddings against prototype centroids computed from the frozen intent assignments rather than training a separate supervised neural classifier.
* **Rationale**: The taxonomy itself is induced dynamically from historical support data, so prototype centroids provide a simple and reproducible classifier that directly reflects the current taxonomy. It also makes similarity scores and top-1/top-2 margins available for explicit abstention. The classifier validates complete training coverage and computes one normalized centroid per intent.

---

### 11. Strict Conversation-Level Partitioning for Zero Data Leakage

* **Decision**: We split the dataset at the `conversation_id` boundary, using 200 golden evaluation cases and up to 8,000 training cases, and explicitly checked for conversation overlap.
* **Rationale**: Twitter support conversations are multi-turn. Splitting individual tweets can place different messages from the same conversation on both sides of the evaluation boundary, allowing the retrieval corpus to contain direct evidence from an evaluation conversation. Splitting entire conversations prevents this form of leakage.

---

### 12. Conditional Response-Quality Benchmark with the Runtime Confidence Gate Ablated

* **Decision**: Response-generation quality is evaluated separately from end-to-end automation coverage, using the exact 47 cases currently audited as suitable for auto-handling while ablating the deployed confidence gate.
* **Rationale**: Evaluating response quality only on the 13 cases that pass the deployed runtime gate would conflate generation quality with conservative coverage. The conditional benchmark instead evaluates whether retrieval + generation can produce strong responses when a case is considered suitable for automation. The current benchmark produced `4.72/5` overall versus `2.38/5` for direct BM25 retrieval.

---

### 13. Multi-Metric Human–Judge Agreement Harness

* **Decision**: We built a formal LLM-judge agreement harness using Spearman correlation, Quadratic Weighted Cohen's Kappa, and Mean Absolute Error across six 1–5 rubric dimensions.
* **Rationale**: A single percentage-agreement figure does not capture ordinal disagreement well. Spearman measures rank consistency, quadratic weighted kappa accounts for chance agreement while penalizing larger ordinal discrepancies more heavily, and MAE captures absolute scoring error. The current harness computes all three on the available overlapping human-scored cases.

---

### 14. Treating Intake Handoffs as Escalations, Not Resolutions

* **Decision**: When historical AmazonHelp behavior consists of directing the customer to a form, DM, tracking link, or support channel rather than actually resolving the issue, the system treats that case as an escalation instead of an autonomous resolution.
* **Rationale**: Providing an intake mechanism is not equivalent to resolving the customer's problem. Counting such handoffs as successful auto-resolutions would artificially inflate coverage and blur the boundary between autonomous resolution and human workflow initiation. `CapabilityGuard` explicitly treats forms, tracking links, data collection, and support deflection as non-autonomous resolution.
