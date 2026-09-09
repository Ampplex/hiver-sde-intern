# Intent Discovery Experiments Log

## Experiment 1: Determining `MATCH_THRESHOLD`

**Goal:** Determine a suitable `MATCH_THRESHOLD` for intent discovery using AWS `amazon.titan-embed-text-v2:0` embeddings.

### Initial Setup

- Started with an assumed `MATCH_THRESHOLD = 0.82`.
- Result: **0 intents discovered**, with all 1,000 cases remaining as unpromoted candidates.
- Observation: `0.82` was too strict for the similarity distribution in the sampled AmazonHelp cases.

### Diagnostic Analysis

Created `05_diagnose_similarity.py` to measure cosine similarity between embeddings for a 1,000-case sample.

**Findings:**

- Maximum pairwise similarity observed: `0.8103`.
- 90th percentile of **2nd nearest-neighbor similarity**: `0.5806`.
- Manual inspection of similarity pairs showed:
  - **`> 0.60`**: generally strong intent-level matches, such as account lockouts and delayed delivery.
  - **`< 0.60`**: increasingly frequent entity-driven similarities, where cases shared a product/entity but had different underlying intents.

### Manual Examples

**Strong matches (`0.60–0.70`):**
- Delayed Prime delivery: `0.69`
- Missing Amazon Pay cashback: `0.64`
- Similar customer-service complaints: `0.62`

**Entity-driven false positives (`0.50–0.60`):**
- Same D&D book mentioned, but one case concerned packaging while the other concerned a preorder delay: `0.55`
- Similar "great/amazing service" wording, but different underlying situations: `0.52`

**Entity-driven false positive (`0.40–0.50`):**
- Same "ceiling fan" entity, but one case concerned delivery restrictions while the other concerned receiving the wrong item: `0.46`

### Decision

Selected **`MATCH_THRESHOLD = 0.62`** as the initial threshold for the next discovery experiment.

The rationale is to retain strong recurring intent similarities while reducing the entity-driven matches observed below `0.60`.

This threshold is **experimental rather than final**. The diagnostic measures pairwise case similarity, whereas the discovery pipeline compares cases against evolving intent prototypes. The resulting taxonomy will therefore be inspected before accepting the threshold.


## Experiment 2: Intent Discovery at `MATCH_THRESHOLD = 0.62`

**Goal:** Evaluate whether the empirically selected threshold produces coherent intents using the incremental intent-registry approach.

### Results

- Cases sampled: `1,000`
- Active intents promoted: `9`
- Assigned cases: `77`
- Assignment coverage: `7.7%`
- Unpromoted candidates: `902`

### Discovered Intents

1. `prime_delivery_delay`
2. `prime_two_day_shipping_issue`
3. `unwanted_prime_charge`
4. `missing_or_incorrect_parcel`
5. `poor_customer_service`
6. `next_day_delivery_issue`
7. `poor_customer_service_2`
8. `package_not_received`
9. `account_hacked`

### Quality Review

The promoted intents were generally coherent and corresponded to actionable customer-support problems.

Examples:

- `unwanted_prime_charge` — customers reporting unexpected Prime charges.
- `account_hacked` — customers reporting compromised accounts.
- `package_not_received` — packages marked delivered but not received.

However, the sequential discovery process introduced **taxonomy fragmentation**.

Examples:

- `poor_customer_service` and `poor_customer_service_2` represent highly similar complaint types.
- `prime_delivery_delay`, `prime_two_day_shipping_issue`, and `next_day_delivery_issue` are closely related delivery problems that may ultimately share a resolution path.

### Observation

The experiment demonstrates a trade-off in the incremental prototype approach:

- A relatively high threshold helps avoid merging semantically different cases.
- However, it can fragment closely related intents into separate prototypes.

This is an expected limitation of the current discovery heuristic, particularly because cases are processed sequentially and prototypes evolve from the cases assigned to them.

### Decision

Accept the resulting taxonomy as the **initial intent registry** for the next stage.

Do not perform an LLM-based merge automatically at this point. Instead, preserve the discovered intents and evaluate whether the downstream retrieval and response-generation system can use them effectively.

Taxonomy refinement can be considered later if evaluation shows that fragmentation materially affects agent performance.

---

## Decision: Separate Intent Discovery from Retrieval

**Decision:** Use the 1,000-case discovery sample to establish an initial intent taxonomy, but build the retrieval index from a separate 5,000-case sample of the broader historical support-case corpus.

**Why:** Intent discovery answers "what kinds of problems exist?", while retrieval answers "how did AmazonHelp historically handle a problem like this?". Restricting retrieval to the 77 cases assigned during discovery would unnecessarily discard useful historical resolutions.

**Implementation:** BM25 and Titan embeddings are built over the 5,000-case retrieval corpus. Each indexed case retains its `conversation_id`, `tweet_ids`, `customer_problem`, `first_amazon_response`, and `full_transcript`.

**Trade-off:** A 5,000-case subsample does not represent the entire 60k+ case corpus, but it keeps the take-home reproducible and inexpensive while providing enough historical examples to demonstrate the retrieval approach.
