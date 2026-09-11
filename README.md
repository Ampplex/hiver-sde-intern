# AmazonHelp AI Support Agent

An autonomous customer support agent built on the Twitter Customer Support (TWCS) dataset, designed to safely triage and resolve support queries for **@AmazonHelp** using an LLM-based **CapabilityGuard**, semantic intent classification, intent-aware hybrid retrieval, and a deterministic safety policy.

---

## Assignment Deliverables Index

This repository satisfies all 5 core deliverables specified in the **Hiver SDE Intern Take-Home Assignment**:

| Deliverable | Requirement | Location / Artifact |
| :--- | :--- | :--- |
| **1. Runnable Pipeline Repo** | Reproduce headline results in $<15$ minutes | [`run_quick_reproduction.sh`](run_quick_reproduction.sh), [`scripts/verify_submission_artifacts.py`](scripts/verify_submission_artifacts.py) |
| **2. Golden Evaluation Set** | 150–250 hand-labelled examples with sampling & labeling note | [`data/processed/eval/golden_labels_manual.csv`](data/processed/eval/golden_labels_manual.csv), [`REPORT.md` (Section 7)](REPORT.md#7-golden-set-curation--labeling-methodology) |
| **3. Evaluation Harness** | Automated metrics + LLM judge + Human–Judge concordance | [`scripts/11_run_evaluation.py`](scripts/11_run_evaluation.py), [`scripts/12_llm_judge.py`](scripts/12_llm_judge.py), [`scripts/13_judge_human_agreement.py`](scripts/13_judge_human_agreement.py) |
| **4. Technical Report** | Max 6 pages: framing, 2 baselines, top 5 failure modes, misleading headline section, roadmap | [`REPORT.md`](REPORT.md) |
| **5. Decision Log** | Plain list of 10–15 non-obvious engineering decisions and rationale | [`DECISION_LOG.md`](DECISION_LOG.md) (14 decisions) |

---

## Headline Results (200-Case Evaluation Set)

| Metric | Proposed System (Audited Labels) | Proposed System (Raw Labels) | Always-Escalate Baseline |
| :--- | :---: | :---: | :---: |
| **Unsafe Auto-Handle Rate** | **0.0% (0/200)** | 1.0% (2/200)* | 0.0% (0/200) |
| **Auto-Handle Precision** | **100.0% (13/13)** | 81.8% (9/11)* | 0.0% |
| **Escalation Recall** | **100.0% (153/153)** | 98.6% (144/146) | 100.0% (153/153) |
| **Auto-Handle Coverage** | **6.5% (13/200)** | 6.5% (13/200) | 0.0% (0/200) |
| **Auto-Handle Recall** | **27.7% (13/47)** | 16.7% (9/54) | 0.0% (0/47) |
| **Intent Classifier Abstention** | **34.5% (69/200)** | 34.5% (69/200) | — |

### Response Quality (47-Case Conditional Auto-Handle Benchmark)

| Quality Dimension (1–5) | Proposed System (RAG + Mistral Large) | Simple Baseline (BM25 Top-1 Historical Reply) | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Quality** | **4.72 / 5.00** | 2.38 / 5.00 | **+2.34** |
| **Correctness** | **4.87 / 5.00** | 2.38 / 5.00 | **+2.49** |
| **Groundedness** | **4.45 / 5.00** | 2.40 / 5.00 | **+2.05** |
| **Resolution Appropriateness** | **4.68 / 5.00** | 2.40 / 5.00 | **+2.28** |
| **Completeness** | **4.83 / 5.00** | 2.38 / 5.00 | **+2.45** |
| **Communication Quality** | **4.98 / 5.00** | 3.81 / 5.00 | **+1.17** |

*Zero data leakage: Strictly partitioned by `conversation_id` (seed=42) with 0% overlap against the 8,000-case training corpus.*  
*Headline summary: On the 200-case audited evaluation set, the system auto-handled 13 cases (6.5%) with 0 observed unsafe auto-handles and 100% escalation recall (153/153). On a separate 47-case conditional response-quality benchmark, it achieved 4.72/5 overall quality versus 2.38/5 for direct BM25 retrieval.*  
*Caveat: Coverage is intentionally conservative and dataset-dependent; it should not be interpreted as a universal production automation rate. Evaluation labels were manually audited with AI assistance.*

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

### Architectural Walkthrough

1. **Intent Discovery & Classification**:
   - **Discovery**: Customer messages in a 1,500-sample discovery cohort were normalized into concise core problems via LLM, clustered with agglomerative clustering (cosine distance + average linkage), and deduplicated into a frozen 104-intent taxonomy.
   - **Centroid Classifier**: Queries are embedded via Amazon Titan Embeddings v2 and scored via cosine similarity against the 104 normalized centroids. The classifier explicitly outputs `uncertain` if top similarity $< 0.45$ or top-2 margin $< 0.02$.

2. **Intent-Aware Hybrid Retrieval**:
   - The predicted intent serves as a **hard candidate filter before ranking** to prune operationally unrelated cases.
   - Candidates are ranked via BM25 lexical search and Titan Dense cosine similarity, fused via Reciprocal Rank Fusion (RRF).
   - **Evidence Gate**: Requires $\ge 1$ strong historical precedent with `dense_score >= 0.55`.

3. **Grounded Response Generation (Asymmetric Trust)**:
   - Retrieved historical cases are injected into Mistral Large (2407 via Bedrock).
   - The prompt enforces an asymmetric trust boundary: historical customer text is untrusted problem description; only AmazonHelp responses represent verified brand resolution patterns.

4. **Semantic Safety via LLM CapabilityGuard**:
   - Rather than brittle regexes, a dedicated Mistral Large call inspects the draft reply against the customer inquiry.
   - It evaluates whether the inquiry is fully resolvable via public informational guidance, or whether it requires private account access, backend state changes, or a human support workflow (e.g., forms, DMs).

5. **Deterministic Structural Safety Policy**:
   - Enforces hard schema invariants, non-empty replies, word ceilings ($> 80$ words escalates to prevent mishandling multi-grievance complaints), and verifies CapabilityGuard approval.

---

## Quickstart & Reproduction (< 15 Minutes)

### 1. Environment Setup

```bash
# Clone and enter the repository
git clone <repo-url>
cd hiver-sde-intern

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Configure AWS Bedrock credentials if running live LLM inference
cp .env.example .env
# Edit .env with your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION=us-west-2
```

### 2. Instant Artifact Verification (< 1 Second)

Validate the 104-intent taxonomy, sha256 hash match, 8,000 training case coverage, 200 golden evaluation cases, zero train/eval leakage, and view the headline metrics summary:

```bash
python scripts/verify_submission_artifacts.py
```

### 3. Full Headline Reproduction Pipeline (< 15 Seconds with Checkpoints / < 15 Min Live)

Reproduces all headline metrics across retrieval build, classifier centroids, component smoke tests, 200-case evaluation, 47-case LLM judge quality benchmark, and human–judge agreement:

```bash
bash run_quick_reproduction.sh
```

*(Note: `run_quick_reproduction.sh` seamlessly supports resumption from checkpoints. If AWS credentials are present, live inference can be re-run at any time using `python scripts/11_run_evaluation.py --recompute` and `python scripts/12_llm_judge.py --force`).*

### 4. Interactive Single-Query Evaluation

Test any custom customer tweet interactively through the complete 5-stage agent pipeline:

```bash
python scripts/09_run_agent.py --query "Where is my delayed package? Tracking has not updated in 3 days."
```

---

## Project Structure

```text
hiver-sde-intern/
├── README.md                                # System overview, architecture, quick reproduction guide
├── REPORT.md                                # 6-page comprehensive technical evaluation report
├── DECISION_LOG.md                          # 14 non-obvious engineering decisions & architectural rationale
├── requirements.txt                         # Pinned Python package dependencies
├── .env.example                             # Template environment configuration (Bedrock credentials)
├── run_quick_reproduction.sh                # End-to-end headline reproduction script (<15 min)
├── run_full_rebuild.sh                      # Full from-scratch rebuild script (offline discovery)
├── run_pipeline.sh                          # Pipeline reproduction script
├── data/
│   └── processed/
│       ├── classifier/                      # 104 prototype centroids & intent taxonomy metadata
│       ├── eval/                            # Golden set (200 cases), predictions, judge scores, metrics
│       ├── intent_assignments.parquet       # 8,000 assigned training cases
│       └── support_cases_train.parquet      # 8,000 clean AmazonHelp conversations (zero leakage)
├── scripts/
│   ├── verify_submission_artifacts.py       # Submission integrity validator
│   ├── 06_build_retriever.py                # Hybrid retriever indexer
│   ├── 07_build_classifier.py               # Prototype centroid builder
│   ├── 08_test_agent_components.py          # Component smoke tester
│   ├── 09_run_agent.py                      # Interactive CLI agent
│   ├── 11_run_evaluation.py                 # End-to-end 200-case golden evaluation harness
│   ├── 12_llm_judge.py                      # LLM-as-a-judge 6-dimension rubric benchmark
│   └── 13_judge_human_agreement.py          # Spearman / Quadratic Kappa / MAE agreement
└── src/
    ├── agent/                               # CapabilityGuard, ResponseGenerator, Deterministic Policy
    ├── classification/                      # Centroid-based IntentClassifier
    └── retrieval/                           # HybridRetriever (BM25 + Dense + Intent-RRF)
```
