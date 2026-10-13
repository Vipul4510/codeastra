# UrbanPulse Phase 3: Rule-Based Incident Correlation Report

**Execution Date:** `2026-10-13T02:04:45+05:30`  
**Corridor:** Mumbai Western Suburbs (Bandra to Dahisar)  
**Evaluated Multimodal Slice:** 5,000 events  
**Candidate Pairs Evaluated:** 3,317 pairs  
**Clusters Formed:** 4,054 candidate incident clusters  

---

## 1. Executive Summary

UrbanPulse Phase 3 establishes the rule-based multimodal incident correlation engine for Mumbai Western Suburbs. The engine operates on standardized, quality-scored multimodal events from Phase 2 (Citizen Reports, Sensor Telemetry, Social Media Posts, and Flood Inundation Observations).

### Key Architectural Invariants Enforced:
1. **Haversine Distance**: Spatial distance calculated via great-circle math.
2. **Hard Compatibility Prior to Scoring**: Cutoffs (distance > 5 km, time > 120 min) prevent unnecessary heuristic computation.
3. **Principle 5 Safeguard**: Spatio-temporal co-occurrence alone **never** triggers an incident merge. Independent civic issues occurring at the same junction are preserved as distinct.
4. **Evidence Chain Integrity**: Social media reposts and forwarded copies are credited with **one evidence chain**, preventing artificial amplification.
5. **Contradiction Preservation**: Factual conflicts (e.g. "subway open" vs "subway flooded") are preserved with explicit flags for human dispatchers rather than being overwritten.
6. **Decoupled Severity Assessment**: Incident severity is evaluated strictly independently from correlation confidence and cluster identity.

---

## 2. Configurable Rules & Thresholds

| Parameter | Value | Domain Rationale |
| :--- | :--- | :--- |
| `candidate_max_distance_km` | **5.0 km** | Maximum spatial horizon for candidate pairing |
| `candidate_max_time_minutes` | **120.0 min** | Maximum candidate window (2 hours) |
| `same_incident_max_distance_km` | **0.85 km** | Strict localized envelope (~850m) for same physical disruption |
| `same_incident_max_time_minutes` | **45.0 min** | Strict temporal window (45 min) for same physical disruption |
| `same_incident_min_heuristic_score` | **0.65** | Heuristic affinity threshold for positive incident merge |
| `duplicate_max_distance_km` | **0.15 km** | Immediate footprint (150m) for near-instant duplicates |
| `duplicate_min_text_similarity` | **0.85** | Token Jaccard threshold (0.85) for text duplicates |
| `related_distinct_max_distance_km` | **3.5 km** | Regional corridor envelope (3.5 km) for cascading impacts |
| `related_distinct_max_time_minutes` | **180.0 min** | Regional temporal horizon (3 hours) for cascades |
| `repost_similarity_threshold` | **0.8** | Threshold for detecting viral social media copies |

---

## 3. Pairwise Relationship Classification Results

Total evaluated candidate pairs: **3,317**

| Relationship Label | Count | Percentage | Definition & Policy |
| :--- | :---: | :---: | :--- |
| **`SAME_INCIDENT`** | 1,047 | 31.6% | Corroborating evidence describing the same localized physical event |
| **`RELATED_BUT_DISTINCT`** | 1,984 | 59.8% | Cascading disruptions or co-located independent events kept separate |
| **`DUPLICATE`** | 140 | 4.2% | Redundant records or social reposts (single evidence chain count) |
| **`CONTRADICTORY`** | 0 | 0.0% | Mutually exclusive assertions at same junction (requires operator audit) |
| **`UNRELATED`** | 146 | 4.4% | Disparate observations violating spatial, temporal, or causal rules |

### Rules Triggered Distribution:
- `RULE_RELATED_DISTINCT_CORRIDOR_CASCADE`: **1,866** times
- `RULE_SAME_INCIDENT_CORROBORATION`: **1,047** times
- `RULE_UNRELATED_INDEPENDENT`: **146** times
- `RULE_EXACT_DUPLICATE_OR_REPOST_CHAIN`: **140** times
- `RULE_CO_LOCATED_INDEPENDENT_EVENTS`: **118** times

---

## 4. Decision Explanations (Sample Pairwise Decisions)

### A. Same Incident Decision:
```json
{
  "pair_id": "EVT-OBS-0004893__EVT-OBS-0003241",
  "event_id_a": "EVT-OBS-0004893",
  "event_id_b": "EVT-OBS-0003241",
  "relation_label": "SAME_INCIDENT",
  "confidence_score": 0.7051,
  "rule_id": "RULE_SAME_INCIDENT_CORROBORATION",
  "explanation": "Corroborating observations of the same localized incident in Jogeshwari: distance=0.21km, time offset=21.0min, category alignment='EXACT', heuristic score=0.71.",
  "evidence_chain_count": 2,
  "geo_distance_km": 0.21,
  "time_diff_minutes": 21.0,
  "text_similarity": 0.1,
  "category_match_type": "EXACT"
}
```

### B. Duplicate / Repost Decision (Single Evidence Chain):
```json
{
  "pair_id": "EVT-OBS-0003712__EVT-OBS-0000699",
  "event_id_a": "EVT-OBS-0003712",
  "event_id_b": "EVT-OBS-0000699",
  "relation_label": "DUPLICATE",
  "confidence_score": 0.98,
  "rule_id": "RULE_EXACT_DUPLICATE_OR_REPOST_CHAIN",
  "explanation": "Near-identical text (1.00) reported in immediate temporal burst (2.0 min).",
  "evidence_chain_count": 1,
  "geo_distance_km": 0.0857,
  "time_diff_minutes": 2.0,
  "text_similarity": 1.0,
  "category_match_type": "EXACT"
}
```

### C. Contradiction Decision (Preserved Conflict):
```json
{
  "pair_id": "EVT-MILAN-01__EVT-MILAN-02",
  "relation_label": "CONTRADICTORY",
  "rule_id": "RULE_CONTRADICTORY_ASSERTION",
  "confidence_score": 0.85,
  "explanation": "Contradictory observations: Conflict: Assertion of normal state ('clear') contradicts assertion of disruption ('flooded') at Santa Cruz within 10.0 min. Preserved for operator audit.",
  "evidence_chain_count": 2
}
```

### D. Related But Distinct Decision (Principle 5 / Cascade):
```json
{
  "pair_id": "EVT-OBS-0001706__EVT-OBS-0002879",
  "event_id_a": "EVT-OBS-0001706",
  "event_id_b": "EVT-OBS-0002879",
  "relation_label": "RELATED_BUT_DISTINCT",
  "confidence_score": 0.7327,
  "rule_id": "RULE_RELATED_DISTINCT_CORRIDOR_CASCADE",
  "explanation": "Cascading or corridor-related incident: Category 'SENSOR_ANOMALY' and 'SENSOR_ANOMALY' share causal/regional episode in Jogeshwari (dist=0.16km, delta_t=55.0min). Kept separate to prevent artificial cluster bloat.",
  "evidence_chain_count": 2,
  "geo_distance_km": 0.1591,
  "time_diff_minutes": 55.0,
  "text_similarity": 0.7143,
  "category_match_type": "EXACT"
}
```

---

## 5. Candidate Incident Clusters Output

- **Total Clusters Formed:** `4,054`
- **Multi-Observation Corroborated Clusters:** `743`
- **Independent Single-Observation Clusters:** `3,311`
- **Clusters with Active Contradictions:** `0`

### Severity Tiers (Decoupled from Identity):
- **CRITICAL**: 440
- **MAJOR**: 420
- **MODERATE**: 697
- **LOW**: 2,497

### Sample Multi-Event Cluster Details:

#### Cluster `CAND-INC-000003`
- **Primary Type**: `FIRE_REPORT` | **Locality**: `Borivali`
- **Member Count**: `2` events | **Evidence Chains**: `2`
- **Contributing Modalities**: `OBSERVATION`
- **Correlation Confidence**: `0.78` | **Severity**: `LOW`
- **Severity Rationale**: Minor civic notice without significant transit obstruction.
- **Has Contradictions**: `False`

#### Cluster `CAND-INC-000005`
- **Primary Type**: `SENSOR_ANOMALY` | **Locality**: `Goregaon`
- **Member Count**: `3` events | **Evidence Chains**: `3`
- **Contributing Modalities**: `OBSERVATION`
- **Correlation Confidence**: `0.74` | **Severity**: `LOW`
- **Severity Rationale**: Minor civic notice without significant transit obstruction.
- **Has Contradictions**: `False`

#### Cluster `CAND-INC-000008`
- **Primary Type**: `ROAD_CLOSURE` | **Locality**: `Andheri`
- **Member Count**: `2` events | **Evidence Chains**: `2`
- **Contributing Modalities**: `OBSERVATION`
- **Correlation Confidence**: `0.73` | **Severity**: `MAJOR`
- **Severity Rationale**: Significant mobility disruption: Road impassable or standing water >= 30cm (depth=0.0cm).
- **Has Contradictions**: `False`

---

## 6. Notice on Prototype Heuristics

All weighted scores in this pipeline represent domain heuristics calibrated for multimodal urban reasoning in the Mumbai Western Suburbs. They are derived from synthetic prototype data and do not claim to establish real-world physical causation.
