# RAG Experiment Comparison: Config_A_TopK_3 vs. Config_B_TopK_5

Comparing **Config_A_TopK_3** (`top_k=3`) against **Config_B_TopK_5** (`top_k=5`).

## Metric Comparison & Deltas

| Metric | Configuration A | Configuration B | Delta (B - A) | % Shift |
| :--- | :---: | :---: | :---: | :---: |
| **Answer Relevancy** | 0.8455 | 0.7699 | **-0.0756** | -8.94% |
| **Context Precision** | 0.6364 | 0.9091 | **+0.2727** | +42.85% |
| **Context Recall** | 0.6364 | 0.7727 | **+0.1363** | +21.42% |
| **Faithfulness** | 1.0000 | 0.7273 | **-0.2727** | -27.27% |

## Latency Comparison

- **Config_A_TopK_3 Latency**: 4.208s
- **Config_B_TopK_5 Latency**: 1.755s
- **Net Latency Delta**: -2.453s

## Automated Engineering Findings

- Increasing retrieval depth to 5 improved Context Recall by +13.6%, capturing more multi-document evidence.
- Context Precision improved by +27.3% under Config_B_TopK_5.
- Faithfulness decreased slightly by -27.3% under larger context.
- Latency impact: Configuration B average latency changed by -2.453s per query.
