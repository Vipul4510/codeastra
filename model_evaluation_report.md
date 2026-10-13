# UrbanPulse Phase 4: Machine Learning Relationship Classification Report

**Corridor:** Mumbai Western Suburbs (Bandra to Dahisar)  
**Task:** 5-Class Multimodal Incident Relationship Classification  
**Target Classes:** `CONTRADICTORY, DUPLICATE, RELATED_BUT_DISTINCT, SAME_INCIDENT, UNRELATED`  
**Dataset Size:** 12,000 pairwise instances  
**Partitioning:** Zero-Leakage Graph Component Group Split (9,287 train / 2,713 test)  
**Evaluation Standard:** Independent, held-out, zero observation overlap  

---

## 1. Executive Summary & Model Leaderboard

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 3 Rule Engine** | **0.2296** | **0.1009** | **0.1070** | 0.2638 | 0.2152 |
| **Logistic Regression** | **0.8835** | **0.8667** | **0.8882** | 0.8567 | 0.8928 |
| **Random Forest** | **0.9292** | **0.9094** | **0.9277** | 0.9224 | 0.9009 |
| **LightGBM** | **0.9211** | **0.9005** | **0.9204** | 0.9047 | 0.8976 |
| **XGBoost** | **0.9318** | **0.9119** | **0.9299** | 0.9309 | 0.9001 |
| **Hybrid Architecture (Rules + Calibrated ML)** | **0.8743** | **0.8532** | **0.8694** | 0.8830 | 0.8451 |

### Key Findings:
1. **Tree Ensembles Achieved Superior Performance**:
   XGBoost achieved the highest test accuracy (**0.9318**) and macro F1 (**0.9119**), closely followed by Random Forest (0.9292) and LightGBM (0.9211), outperforming the linear Logistic Regression baseline (0.8835) by over 5.4% accuracy and 4.9% macro F1.
2. **Phase 3 Rule-Based Engine vs. Machine Learning**:
   The Phase 3 heuristic rule engine achieved **0.2296 accuracy** on this synthetic benchmark split. This discrepancy highlights the critical difference between domain-engineered physical rules (which require explicit semantic antonyms for CONTRADICTORY) versus the synthetic dataset generator (which assigned CONTRADICTORY using simple spatial-proximity box thresholds).
3. **Probability Calibration Verified**:
   Calibrating the tree classifier via Sigmoid scaling improved log-loss from **0.2357** to **0.2154**, providing well-calibrated confidence scores for downstream evidence graph construction.
4. **Hybrid Decision Architecture Provides Safety**:
   By coupling hard spatial/temporal cutoffs with calibrated model predictions and uncertainty thresholds, the hybrid engine prevents dangerous false merges while maintaining **87.4% accuracy**.

---

## 2. Target Label Audit & Synthetic Dataset Limitations

Before training, we audited the ground-truth labeling mechanisms of `pairwise_relationship_training.csv`:

### Label Distribution:
- `SAME_INCIDENT`: 4,075 (34.0%)
- `UNRELATED`: 2,676 (22.3%)
- `RELATED_BUT_DISTINCT`: 2,367 (19.7%)
- `DUPLICATE`: 1,886 (15.7%)
- `CONTRADICTORY`: 996 (8.3%)

### Documented Limitations of the Synthetic Prototype Labels:
> [!WARNING]
> **Synthetic Labeling Geometry Artifact:**  
> Our decision-tree audit revealed that the synthetic benchmark labels were generated using numeric box thresholds on spatio-temporal and text features rather than real-world physical ground truth. Specifically:
> - `CONTRADICTORY` was largely assigned to pairs with `geo_distance_km <= 0.72` and `text_similarity <= 0.65`, regardless of whether the texts contained factual semantic contradictions.
> - `SAME_INCIDENT` was assigned when `text_similarity > 0.65` and `geo_distance_km <= 1.03`.
> - Because these synthetic rules form piecewise-linear boundaries, tree-based models (Random Forest, LightGBM, XGBoost) fit them with high fidelity (> 93%), whereas linear models (Logistic Regression) achieve 88.3%.
> - **Recommendation for Real-World Deployment**: When deploying on actual field data from BMC / Mumbai Traffic Police, ground truth must be verified by human incident dispatch logs, ground truth camera feeds, or crowd-sourced confirmation rather than threshold formulas.

---

## 3. Detailed Per-Class Performance Comparison

### Best Model: LightGBM Classifier (Zero-Leakage Test Split)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`CONTRADICTORY`** | 0.7623 | 0.6911 | 0.7249 | 246 |
| **`DUPLICATE`** | 0.9147 | 0.9728 | 0.9429 | 441 |
| **`RELATED_BUT_DISTINCT`** | 0.9522 | 0.9274 | 0.9396 | 537 |
| **`SAME_INCIDENT`** | 0.8995 | 0.9173 | 0.9083 | 907 |
| **`UNRELATED`** | 0.9948 | 0.9794 | 0.9870 | 582 |

### Confusion Matrix (LightGBM):
```
Predicted ->
True Class             CONTRADICTORY  DUPLICATE  RELATED_DISTINCT  SAME_INCIDENT  UNRELATED
CONTRADICTORY                170          0                11             65          0
DUPLICATE                      0        429                 0             12          0
RELATED_BUT_DISTINCT          20          0               498             16          3
SAME_INCIDENT                 33         40                 2            832          0
UNRELATED                      0          0                12              0        570
```

---

## 4. Feature Importance Analysis

The top 10 most influential features driving relationship classification:

| Rank | Feature Name | LightGBM Splits | Random Forest Gini Importance | Domain Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `spatiotemporal_proximity_product` | 5,950 | 0.1562 | Key boundary discriminator |
| 2 | `text_similarity` | 5,459 | 0.1591 | Key boundary discriminator |
| 3 | `source_reliability_min` | 2,225 | 0.0055 | Key boundary discriminator |
| 4 | `reliability_product` | 2,122 | 0.0062 | Key boundary discriminator |
| 5 | `geo_distance_km` | 1,970 | 0.0709 | Key boundary discriminator |
| 6 | `time_diff_minutes` | 1,888 | 0.0561 | Key boundary discriminator |
| 7 | `phase3_heuristic_affinity_score` | 1,861 | 0.1071 | Key boundary discriminator |
| 8 | `source_reliability_max` | 1,857 | 0.0046 | Key boundary discriminator |
| 9 | `spatial_proximity_decay` | 1,342 | 0.0502 | Key boundary discriminator |
| 10 | `temporal_proximity_decay` | 1,295 | 0.0538 | Key boundary discriminator |

---

## 5. False Merges and Missed Matches Examination

1. **False Merges (`RELATED_BUT_DISTINCT` misclassified as `SAME_INCIDENT`)**:
   - Only **16** instances out of 537 `RELATED_BUT_DISTINCT` test cases (3.0%) were falsely predicted as `SAME_INCIDENT`.
   - The tree ensembles successfully preserve distinct incidents even when both events occur in the same suburban ward.
2. **Contradiction Ambiguity (`CONTRADICTORY` vs `SAME_INCIDENT`)**:
   - **65** out of 246 `CONTRADICTORY` test cases were classified as `SAME_INCIDENT`, and **33** `SAME_INCIDENT` cases were classified as `CONTRADICTORY`.
   - This occurs because both classes share identical spatio-temporal envelopes (distance ~0.36 km, time delta ~20 min) in the synthetic dataset, differing primarily around a text similarity boundary (~0.65).
3. **Missed Matches (`SAME_INCIDENT` misclassified as `RELATED_BUT_DISTINCT`)**:
   - Only **2** instances out of 907 `SAME_INCIDENT` pairs (0.2%) were predicted as `RELATED_BUT_DISTINCT`.
4. **Near-Zero Leakage Across Unrelated Events**:
   - Out of 582 true `UNRELATED` events, **570** were correctly classified (97.9% recall), with **zero** false merges into `SAME_INCIDENT`.

---

## 6. Hybrid Architecture Recommendations for Downstream Phase 5

For building the multimodal incident evidence graph (Phase 5):
1. **Front-End Hard Rules**: Apply hard cutoffs (> 5 km, > 120 min) to prune the candidate graph before model evaluation.
2. **Calibrated ML Edge Scoring**: Use the calibrated LightGBM model to predict relationship probabilities [P(SAME), P(RELATED), P(DUPLICATE), P(CONTRADICTORY), P(UNRELATED)].
3. **Uncertainty Fallback**: If max(P) < 0.45, mark edge as UNCERTAIN for operator verification.
4. **Graph Edges**: Connect SAME_INCIDENT edges with weights equal to calibrated confidence to form high-fidelity incident clusters.
