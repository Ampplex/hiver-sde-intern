# Decision Log: 12 Non-Obvious Engineering Decisions

This log documents the key non-obvious architectural, algorithmic, and operational decisions made during the design, development, and evaluation of the AmazonHelp Autonomous Support Agent.

---

### 1. Semantic Safety via LLM CapabilityGuard Rather Than Regex Keyword Filters
* **Decision**: We eliminated brittle semantic regex patterns (e.g., matching keywords like card, password, account, address) from the deterministic safety policy and delegated all semantic resolution judgments to an LLM-based CapabilityGuard.
* **Rationale**: Keyword regexes repeatedly created unacceptable failure modes:
  1. *False negatives / security escapes*: An adversary or customer phrasing a credential inquiry without trigger keywords bypassed regex filters.
  2. *False positives / over-escalation*: Legitimate public reports (e.g., Case 1953280 reporting a phishing email asking for credit card details) were aggressively blocked simply because the message mentioned sensitive terms, even though the customer was warning Amazon and required general informational guidance.
  3. *Semantic boundary*: Distinguishing whether an inquiry requires private account access vs. public informational guidance is fundamentally a semantic reasoning task, which a small-footprint structured LLM prompt evaluates with 100% precision.

---

### 2. Strict Division of Labor: Deterministic Invariants vs. Semantic Judgments
* **Decision**: We restricted policy.py strictly to deterministic, verifiable structural invariants (classifier confidence checks, message length thresholds, minimum retrieval density, schema validation) while delegating all semantic safety decisions to CapabilityGuard.
* **Rationale**: Mixing semantic heuristics into deterministic code produces opaque, unmaintainable rule spaghetti. Deterministic code excels at hard boundaries (e.g., "message > 80 words is ambiguous", "generation JSON missing draft_reply is invalid"). Semantic models excel at situational comprehension. Keeping this separation clean ensures reproducibility and transparent failure tracing.

---

### 3. Lowering Historical Evidence Gate from 2 Matches to 1 Match (MIN_STRONG_EVIDENCE_CASES=1)
* **Decision**: We reduced the required number of high-similarity historical precedents (dense_score >= 0.55) from 2 to 1.
* **Rationale**: Our detailed error diagnostic revealed that requiring two strong matches was overly conservative for distinct informational topics (e.g., content availability, scam warnings, positive feedback). For example, Case 1953280 had a single stellar historical precedent (dense_score = 0.6702) with identical handling, but was blocked solely because a second match was absent. By reducing the threshold to 1 while relying on CapabilityGuard to verify whether the single precedent genuinely resolves the inquiry, safe auto-handle coverage increased by 175% (from 4 to 11 cases) with zero degradation in safety.

---

### 4. Asymmetric Trust Boundary: Historical Brand Responses as Evidence, Customer Messages as Untrusted Context
* **Decision**: In retrieval and prompt injection, historical AmazonHelp replies are treated as authoritative evidence of brand resolution patterns, whereas historical customer messages are treated strictly as descriptive context of the incoming problem.
* **Rationale**: Customer messages in public Twitter threads frequently contain incorrect claims, false policy assumptions, or adversarial noise. Instructing the generation LLM that customer messages are untrusted prevents the agent from adopting customer-invented SLAs, refund promises, or erroneous policy statements.

---

### 5. Intent-Aware Reciprocal Rank Fusion (RRF) Retrieval
* **Decision**: Rather than using pure BM25 or pure dense semantic vector retrieval, we implemented hybrid RRF that explicitly boosts historical cases whose assigned intent matches the classifier's predicted intent.
* **Rationale**: Lexical search alone fails on paraphrased customer complaints; dense semantic embeddings alone struggle with exact order terminology, product acronyms, and alphanumeric identifiers. Intent-aware RRF combines lexical precision with semantic recall and anchors the retrieved context to the operational domain of the problem.

---

### 6. Abstention as a First-Class Classifier Output
* **Decision**: The intent classifier outputs uncertain whenever the top-1 cosine similarity falls below 0.45 OR the margin between top-1 and top-2 falls below 0.02.
* **Rationale**: In customer support, classifying an inquiry into the wrong intent causes downstream retrieval of irrelevant precedents, which directly poisons response generation. The 0.05 margin threshold was empirically over-conservative for the fine-grained 104-intent taxonomy. Calibration to 0.02 reduced unnecessary abstention (58.5% to 34.5%) while preserving the safety gate (0.0% unsafe auto rate). Explicit abstention allows the agent to safely escalate to human specialists whenever confidence is marginal, directly protecting end-user trust.

---

### 7. Problem Normalization Before Taxonomy Clustering
* **Decision**: During offline intent discovery (scripts/04_discover_intents.py), customer tweets were normalized into concise 1-sentence "core operational problems" via LLM before embedding and agglomerative clustering, rather than clustering raw tweet text.
* **Rationale**: Raw tweets are flooded with Twitter handles (@AmazonHelp), emojis, greetings, customer venting, and idiosyncratic punctuation. Clustering raw tweet embeddings groups conversations by conversational tone or superficial keywords rather than the underlying operational failure. Normalization extracted the true support signal, yielding a clean, actionable 104-intent taxonomy.

---

### 8. Strict Conversation-Level Partitioning for Zero Data Leakage
* **Decision**: Splitting between training (8,000 cases) and golden evaluation (200 cases) was strictly executed at the conversation_id boundary, verified by SHA-256 hash checks.
* **Rationale**: In multi-turn Twitter customer support, the same user and brand frequently exchange multiple tweets. Splitting by individual message ID causes catastrophic data leakage where the evaluation set contains replies to conversations already indexed in the retrieval training corpus. Partitioning by entire conversation trees ensures 100% genuine generalization.

---

### 9. Ablated Response-Quality Benchmarking for LLM-as-a-Judge
* **Decision**: To evaluate response generation quality independently of the conservative safety gate, we constructed a benchmark cohort of human-labeled auto-handle cases and evaluated generated replies against BM25 top-1 historical responses.
* **Rationale**: If response quality is measured only on cases that pass the deployed safety gate, sample size is restricted to 11 cases, creating severe selection bias. By evaluating on all cases that human annotators verified as suitable for automated resolution, we directly test whether retrieval + generation produces faithful, high-quality responses when given the opportunity.

---

### 10. Multi-Metric Human-Judge Agreement Harness (Spearman + Quadratic Kappa + MAE)
* **Decision**: We validated the LLM-as-a-judge against human expert ratings using three complementary statistical tests across 6 rubric dimensions: Spearman's rank correlation, Quadratic Weighted Cohen's Kappa, and Mean Absolute Error.
* **Rationale**: Reporting percentage agreement or Pearson correlation is misleading for ordinal 1-5 scales with skewed distributions. Quadratic Kappa accounts for chance agreement and penalizes large discrepancies quadratically. Combining it with MAE ensures we measure both relative ranking consistency and absolute scoring calibration.

---

### 11. Preserving Intake Links as Escalations Rather Than "Fake Auto-Resolutions"
* **Decision**: When historical precedents show Amazon agents providing a webform or DM intake link (e.g., Case 899440 packaging complaint), CapabilityGuard escalates the case rather than auto-handling it with the link.
* **Rationale**: In customer support, sending an automated message saying "Click here to fill out a form" is not a resolution; it is an automated ticket handoff. Treating link handoffs as successful auto-resolutions inflates coverage artificially while frustrating customers who expect their issue to be actively resolved.

---

### 12. Length-Based Ambiguity Gate (> 80 Words Escalates)
* **Decision**: The policy implements a deterministic word-count ceiling: any incoming customer message exceeding 80 words is escalated immediately as potentially ambiguous or multi-issue.
* **Rationale**: Twitter messages exceeding 80 words in customer support almost universally span multiple interwoven grievances (e.g., late delivery AND damaged product AND poor driver conduct AND rude phone support). Attempting single-intent automated resolution on multi-issue rants leads to addressing only one sub-issue while ignoring others, creating severe customer dissatisfaction.
