# Confidence Scoring & Evidence Sufficiency

## Overview

The **AI Knowledge System** uses a deterministic, mathematically grounded confidence calculation to ensure every generated answer is traceable and verifiable. It rejects arbitrary or heuristic confidence values in favor of a weighted composite score evaluated directly against the retrieved knowledge chunks and query intent.

If evidence is insufficient, the system short-circuits to an explicit, machine-readable refusal (`INSUFFICIENT_EVIDENCE`) and sets confidence to `"none"`.

---

## Confidence Calculation Formula

Given:
- A normalized user query $Q$ with extracted topic keywords $K_Q = \{k_1, k_2, \dots, k_m\}$.
- A ranked sequence of retrieved chunks $R = [c_1, c_2, \dots, c_n]$ with cosine similarity scores $s_i \in [0.0, 1.0]$.
- A minimum sufficiency threshold $\tau_{\text{suff}} = 0.45$.

The composite confidence score $C \in [0.0, 1.0]$ is computed as:

$$C = (w_{\text{top}} \cdot S_{\text{top}}) + (w_{\text{mean}} \cdot \bar{S}) + (w_{\text{lex}} \cdot C_{\text{lex}})$$

Where:
- **$S_{\text{top}}$ (Top Similarity Factor, weight $w_{\text{top}} = 0.50$)**:
  The cosine similarity score of the highest-ranked chunk ($s_1$).
- **$\bar{S}$ (Mean Evidence Strength, weight $w_{\text{mean}} = 0.30$)**:
  The arithmetic mean of similarity scores across the top $k$ retrieved chunks (where $k = \min(n, 3)$):
  $$\bar{S} = \frac{1}{k} \sum_{i=1}^{k} s_i$$
- **$C_{\text{lex}}$ (Lexical Keyword Recall, weight $w_{\text{lex}} = 0.20$)**:
  The fraction of query keywords verified to be present in the retrieved chunk evidence:
  $$C_{\text{lex}} = \frac{|\{k \in K_Q \mid k \in \text{corpus}(c_1 \dots c_k)\}|}{|K_Q|} \quad (\text{or } 1.0 \text{ if } |K_Q| = 0)$$

---

## Confidence Levels and Calibration

The composite score $C$ maps directly to categorical confidence ratings:

| Confidence Level | Composite Score Range | Description & Evidence Criteria |
|---|---|---|
| **`high`** | $C \ge 0.75$ | Strong semantic similarity ($\ge 0.75$), multi-chunk concordance, and high query keyword overlap. |
| **`medium`** | $0.55 \le C < 0.75$ | Moderate semantic similarity or single-source evidence with solid keyword alignment. |
| **`low`** | $0.40 \le C < 0.55$ | Marginal evidence near threshold; answer is plausible but evidence is sparse. |
| **`none`** | $C < 0.40$ or $s_1 < \tau_{\text{suff}}$ | Insufficient evidence. Returns controlled refusal with `reason: "INSUFFICIENT_EVIDENCE"`. |

---

## Insufficient Evidence Short-Circuiting

The system guarantees that unsupported questions do not receive fabricated or hallucinated answers through two gates:

1. **Pre-Generation Gate (Retrieval Sufficiency)**:
   - If retrieval yields $0$ chunks or the top similarity score is below $\tau_{\text{suff}}$ ($0.45$):
     - The pipeline short-circuits immediately.
     - No LLM generation call is made.
     - Response: `"I couldn't find sufficient information in the available knowledge base to answer this question reliably."`
     - Machine-readable reason: `"INSUFFICIENT_EVIDENCE"`.
     - `sources: []`, `confidence: "none"`, `confidence_score: 0.0`.

2. **Post-Generation Gate (Grounded Verification)**:
   - If the LLM generates an explicit refusal or detects absent premise details:
     - The pipeline catches the refusal.
     - Sources are zeroed out (`sources: []`) to prevent false attribution.
     - Reason is marked as `"INSUFFICIENT_EVIDENCE"`.
     - `confidence: "none"`, `confidence_score: 0.0`.

---

## Citation & Excerpt Construction

Each source citation returned by `POST /api/v1/ask` includes:
- **`document`**: Filename of originating document (e.g. `sample.docx`).
- **`version`**: Version identifier (e.g. `1.0`, `2026.1`).
- **`chunk_id`**: Unique UUID of the chunk.
- **`score`**: Exact cosine similarity score.
- **`excerpt`**: Sentence-level snippet containing the highest density of query keywords, bounded to 250 characters.
