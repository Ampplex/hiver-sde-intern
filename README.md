# AmazonHelp AI Support Agent

An autonomous customer support agent built on the Twitter Customer Support (TWCS) dataset, designed to safely triage and resolve support queries using an LLM-based **CapabilityGuard**, semantic intent classification, intent-aware hybrid retrieval, and a deterministic safety policy.

---

## Headline Results (200-Case Evaluation Set)

| Metric | Proposed System (Audited Labels) | Proposed System (Raw Labels) | Always-Escalate Baseline |
| :--- | :---: | :---: | :---: |
| **Unsafe Auto-Handle Rate** | **0.0% (0/200)** | 1.0% (2/200)* | 0.0% (0/200) |
| **Auto-Handle Precision** | **100.0% (13/13)** | 81.8% (9/11)* | 0.0% |
| **Escalation Recall** | **100.0% (153/153)** | 98.6% (144/146) | 100.0% (153/153) |
| **Auto-Handle Coverage** | **6.5% (13/200)** | 6.5% (13/200) | 0.0% (0/200) |
| **Auto-Handle Recall** | **27.7% (13/47)** | 16.7% (9/54) | 0.0% (0/47) |

### Response Quality (47-Case Conditional Auto-Handle Benchmark)

| Quality Dimension (1–5) | Proposed System (RAG + Mistral Large) | Simple Baseline (BM25 Top-1 Historical Reply) |
| :--- | :---: | :---: |
| **Overall Quality** | **4.72 / 5.00** | 2.38 / 5.00 |
| **Correctness** | **4.87 / 5.00** | 2.38 / 5.00 |
| **Groundedness** | **4.45 / 5.00** | 2.40 / 5.00 |
| **Resolution Appropriateness** | **4.68 / 5.00** | 2.40 / 5.00 |
| **Completeness** | **4.83 / 5.00** | 2.38 / 5.00 |
| **Communication Quality** | **4.98 / 5.00** | 3.81 / 5.00 |

*Zero data leakage: Strictly partitioned by `conversation_id` (seed=42) with 0% overlap against the 8,000-case training corpus.*
*Headline summary: On the 200-case audited evaluation set, the system auto-handled 13 cases (6.5%) with 0 observed unsafe auto-handles and 100% escalation recall (153/153). On a separate 47-case conditional response-quality benchmark, it achieved 4.72/5 overall quality versus 2.38/5 for direct BM25 retrieval.*
*Caveat: Coverage is intentionally conservative and dataset-dependent; it should not be interpreted as a production automation rate. Evaluation labels were manually audited with AI assistance.*

---

## Deliverables Index

* 📄 **[Technical Evaluation Report](REPORT.md)**: Full 6-page report covering problem framing, results vs baselines, failure modes, the mandatory *"What is misleading about my headline number?"* section, and future roadmap.
* 📋 **[Engineering Decision Log](DECISION_LOG.md)**: Plain list of 12 non-obvious engineering decisions and their architectural rationale.
* 🏷️ **[Golden Evaluation Set](data/processed/eval/golden_labels_manual.csv)**: 200 evaluation examples with golden intents, actions, and justifications, manually audited with AI assistance.
* ⚖️ **[Judge Evaluation & Human Agreement](data/processed/eval/judge_human_agreement.json)**: Automated LLM-as-a-judge scores and human concordance analysis.

---

## System Architecture

The agent strictly separates probabilistic operations (classification, retrieval, generation, semantic safety verification) from deterministic operations (structural escalation gating and schema invariants).

```mermaid
flowchart TD
    A[Customer Tweet] --> B[Amazon Titan Embeddings v2]
    B --> C[Intent Classifier<br>104 Prototype Centroids]
    
    C -->|Sim < 0.45 OR Margin < 0.02| E[ESCALATE<br>Classifier Uncertainty]
    C -->|Confident Intent| F[Hybrid Retriever<br>BM25 + Dense + Intent-RRF]
    
    F -->|< 1 Match with Dense Score >= 0.55| E
    F -->|Historical Cases| G[Response Generator<br>Mistral Large Grounding]
    
    G -->|Generator Recommends Escalation| E
    G -->|Drafted Reply| H[CapabilityGuard<br>LLM Semantic Safety Boundary]
    
    H -->|Requires Private State / Action| E
    H -->|Informational Resolution Approved| I[Deterministic Safety Policy]
    
    I -->|Length > 80 Words OR Bad Schema| E
    I -->|Passed All Invariants| J[AUTO-HANDLE]
```

### 1. Intent Discovery & Classification
* **Discovery**: An LLM-assisted intent discovery pipeline normalized noisy customer messages into underlying operational problems, clustered representations via agglomerative clustering, and consolidated candidates into a frozen 104-intent taxonomy.
* **Classifier**: Incoming tweets are classified via cosine similarity to the 104 prototype centroids. The classifier explicitly outputs `uncertain` if top similarity $< 0.45$ or top-2 margin $< 0.02$.

### 2. Intent-Aware Hybrid Retrieval
When the classifier is confident, the message and predicted intent query the 8,000-case historical support corpus:
* **BM25**: Lexical search against historical support cases.
* **Dense**: Semantic search using Amazon Titan Embeddings v2.
* **Reciprocal Rank Fusion (RRF)**: Merges rank lists with heavy boosts for cases matching the predicted intent.
* **Evidence Gate**: Requires $\ge 1$ strong precedent with `dense_score >= 0.55`.

### 3. Grounded Response Generation (Asymmetric Trust)
The top retrieved historical cases are injected into Mistral Large (2407 via Bedrock). The prompt strictly treats historical customer messages as untrusted context, enforcing that only AmazonHelp replies serve as evidence of official brand resolution patterns.

### 4. Semantic Safety via LLM CapabilityGuard
Rather than brittle keyword regexes, a dedicated Mistral Large guard evaluates whether the drafted response genuinely and safely resolves the inquiry using public information, or whether resolving the inquiry requires private customer state, account modification, or a human support workflow.

### 5. Deterministic Safety Policy
The `apply_safety_policy` module enforces hard invariants:
1. Classifier confidence / margin checks.
2. Word-count ceiling: Messages $> 80$ words escalate immediately to prevent single-intent mishandling of multi-issue complaints.
3. Retrieval evidence gating.
4. Schema and draft validation (formatting, character limits, non-empty replies).
5. CapabilityGuard decision enforcement.

---

## Reproduction

The submitted results use frozen model-building artifacts (`data/processed/intent_taxonomy.json`, `data/processed/intent_assignments.parquet`, `data/processed/intent_assignment_embeddings.npy`, etc.) to guarantee 100% deterministic reproducibility without re-running expensive offline discovery.

### Quick Reproduction Path (< 15 Minutes)
Reproduces the complete evaluation results from the submission run:

```bash
bash run_quick_reproduction.sh
```

**Execution Stages:**
1. **Artifact Verification (`scripts/verify_submission_artifacts.py`)**: Validates the 104-intent taxonomy, sha256 hash match, 8,000 training case coverage, 200 golden evaluation cases, and 0% train/eval leakage.
2. **Retriever Build (`scripts/06_build_retriever.py`)**: Builds BM25 index and dense embeddings (< 5 sec).
3. **Classifier Build (`scripts/07_build_classifier.py`)**: Recomputes 104 prototype centroids (< 5 sec).
4. **Smoke Test (`scripts/08_test_agent_components.py`)**: Tests classification, margin estimation, and hybrid retrieval (< 5 sec).
5. **System Evaluation (`scripts/11_run_evaluation.py`)**: Runs end-to-end evaluation on the 200 golden cases across baselines and proposed agent.
6. **LLM Judge Evaluation (`scripts/12_llm_judge.py`)**: Evaluates response quality on correctness, groundedness, completeness, and communication quality.
7. **Human-Judge Agreement (`scripts/13_judge_human_agreement.py`)**: Computes Spearman rank correlation, Quadratic Cohen's Kappa, and MAE against human expert scores.

---

### Full Rebuild Path (Offline Discovery Audit)
To regenerate the taxonomy, embeddings, and training assignments from scratch via Bedrock:

```bash
bash run_full_rebuild.sh
```

*(Note: Full rebuild regenerates the 104-intent taxonomy through clustering and embedding discovery, which takes ~60 to 90 minutes due to API rate limits).*
