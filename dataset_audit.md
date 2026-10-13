# UrbanPulse: Dataset Audit and Project Understanding (Phase 1)

**Project Name**: UrbanPulse – AI-Powered Multimodal Urban Incident Intelligence Engine  
**Geographic Domain**: Mumbai Western Suburbs Corridor (Bandra to Dahisar)  
**Dataset Directory**: `UrbanPulse_Expanded_Dataset`  
**Total CSV Files Audited**: 13  
**Total Cumulative Records**: 120,940  
**Audit Status**: COMPLETED — Rigorously Verified with Automated Validation Script  

---

## 1. Executive Summary

This document presents the complete Phase 1 Dataset Audit for **UrbanPulse**. Every statistic, row count, column count, referential check, label distribution, and hazard report in this document is derived from direct programmatic execution against the actual CSV files on the local filesystem.

### Key Audit Findings at a Glance:
1. **File Completeness**: All 13 expanded dataset CSV files are present, structurally consistent, and completely readable.
2. **Primary Key Integrity**: 100% of candidate primary keys across all 13 tables are strictly unique with zero null keys.
3. **Foreign Key Integrity**: Cross-table referential integrity is exceptionally high:
   - 100% (21,000/21,000) of observation-incident links match valid observation and incident records.
   - 100% (595/595) of sensor telemetry IDs match physical assets in `infrastructure_assets.csv`.
   - 100% (461/461) of flood observation assets match physical infrastructure assets.
   - 100% (2,500/2,500) of asset dependency endpoints resolve to valid assets.
   - 100% (12,000/12,000) of pairwise training observation IDs resolve to valid observations.
   - 100% (5,000/5,000) of route impact routes and incidents match their parent catalogs.
   - 100% (36,000/36,000) of supporting evidence tokens in the knowledge graph match valid observations.
4. **Critical Data Leakage Identified**:
   - In `observations.csv`, the column `candidate_incident_id` contains the ground-truth canonical cluster ID for 21,000 out of 25,000 observations (84.0%). **This must be strictly quarantined from inference feature pipelines.**
   - In `social_post_analysis.csv`, fields such as `incident_relevance_score`, `predicted_event_type`, and `extracted_location` are synthetic model predictions, not raw ground-truth sensor telemetry.
5. **Synthetic Nature Verified**: All 13 datasets are synthetic demonstrations. Zero rows represent live, unverified sensor telemetry or real scraped citizen posts. In `social_media_posts.csv`, all 6,000 permalinks are null.
6. **Class Imbalance & Distributions Documented**: Identified severe class skews in pairwise relationship labels (33.96% `SAME_INCIDENT` vs 8.30% `CONTRADICTORY`) and incident severity (42.24% `MODERATE` vs 9.98% `CRITICAL`).

---

## 2. Comprehensive File-by-File Profile

| Dataset File | Rows | Columns | File Size (Bytes) | Exact Duplicate Rows | Unique Identifier Candidate | Primary Key Status | Null Columns Count |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| `observations.csv` | 25,000 | 18 | 6,913,549 | 0 | `observation_id` | **PASS** (100% unique, 0 null) | 2 |
| `incident_clusters.csv` | 4,200 | 11 | 576,363 | 0 | `incident_id` | **PASS** (100% unique, 0 null) | 0 |
| `observation_incident_links.csv` | 21,000 | 6 | 1,488,862 | 0 | `(observation_id, incident_id)` | **PASS** (Composite Unique, 0 dupes) | 0 |
| `infrastructure_assets.csv` | 1,200 | 12 | 152,584 | 0 | `asset_id` | **PASS** (100% unique, 0 null) | 2 |
| `asset_dependencies.csv` | 2,500 | 8 | 225,188 | 0 | `dependency_id` | **PASS** (100% unique, 0 null) | 0 |
| `sensor_readings.csv` | 15,000 | 17 | 2,273,424 | 0 | `reading_id` | **PASS** (100% unique, 0 null) | 0 |
| `social_media_posts.csv` | 6,000 | 15 | 1,174,337 | 0 | `post_id` | **PASS** (100% unique, 0 null) | 4 |
| `social_post_analysis.csv` | 6,000 | 11 | 704,091 | 0 | `post_id` | **PASS** (100% unique, 0 null) | 1 |
| `flood_incident_observations.csv` | 5,000 | 20 | 1,059,508 | 0 | `flood_observation_id` | **PASS** (100% unique, 0 null) | 2 |
| `pairwise_relationship_training.csv` | 12,000 | 12 | 1,313,394 | 0 | `pair_id` | **PASS** (100% unique, 0 null) | 0 |
| `relationship_evidence_graph_edges.csv` | 18,000 | 12 | 3,436,807 | 0 | `edge_id` | **PASS** (100% unique, 0 null) | 0 |
| `routes.csv` | 40 | 11 | 4,838 | 0 | `route_id` | **PASS** (100% unique, 0 null) | 0 |
| `route_impacts.csv` | 5,000 | 11 | 635,678 | 0 | `route_impact_id` | **PASS** (100% unique, 0 null) | 0 |
| **TOTAL** | **120,940** | **164** | **18,878,799** | **0** | — | — | — |

---

## 3. Dataset Schemas, Dtypes, and Missing Value Details

### 3.1 `observations.csv` (25,000 rows, 18 columns)
- **Columns & Dtypes**:
  `observation_id` (object), `source_id` (object), `source_type` (object), `event_time` (object/ISO 8601), `ingested_at` (object), `event_type` (object), `text_original` (object), `text_normalized` (object), `latitude` (float64), `longitude` (float64), `location_text` (object), `location_accuracy_m` (float64), `source_reliability_score` (float64), `verification_status` (object), `is_duplicate_candidate` (bool), `is_simulated` (bool), `provenance` (object), `candidate_incident_id` (object).
- **Missing Values**:
  - `location_accuracy_m`: 4,147 missing (16.59%)
  - `candidate_incident_id`: 4,000 missing (16.00%) — represents unassigned/unclustered observations.
- **Geographic Bounds**: Latitude [19.1051, 19.2614], Longitude [72.8017, 72.8713].
- **Temporal Range**: `2025-12-31T23:54:18` to `2026-10-12T23:04:04`.

### 3.2 `incident_clusters.csv` (4,200 rows, 11 columns)
- **Columns & Dtypes**:
  `incident_id` (object), `incident_type` (object), `locality` (object), `latitude` (float64), `longitude` (float64), `first_seen` (object), `last_seen` (object), `severity` (object), `cluster_status` (object), `is_synthetic` (bool), `provenance` (object).
- **Missing Values**: 0 nulls across all 11 columns.
- **Geographic Bounds**: Latitude [19.1039, 19.2619], Longitude [72.8021, 72.8718].
- **Temporal Range**: `2026-01-01T00:05:18` to `2026-10-12T21:51:52`.

### 3.3 `observation_incident_links.csv` (21,000 rows, 6 columns)
- **Columns & Dtypes**:
  `observation_id` (object), `incident_id` (object), `membership_score` (float64), `decision_type` (object), `is_ground_truth` (bool), `provenance` (object).
- **Missing Values**: 0 nulls across all 6 columns.
- **Membership Score Range**: [0.450, 0.990], Mean: 0.7195, Median: 0.7190.

### 3.4 `infrastructure_assets.csv` (1,200 rows, 12 columns)
- **Columns & Dtypes**:
  `asset_id` (object), `asset_type` (object), `asset_name` (object), `locality` (object), `latitude` (float64), `longitude` (float64), `catchment_id` (object), `design_capacity_lps` (float64), `elevation_m` (float64), `asset_status` (object), `data_source` (object), `is_synthetic` (bool).
- **Missing Values**:
  - `catchment_id`: 615 missing (51.25%) — non-drainage assets (e.g. road segments, power feeders) lack catchment basin IDs.
  - `design_capacity_lps`: 983 missing (81.92%) — applicable only to drains, pumps, and gauges.
- **Asset Type Breakdown**:
  `TRAFFIC_SENSOR` (129), `SUBWAY_UNDERPASS` (125), `RAIN_GAUGE` (125), `RAIL_STATION` (125), `WATER_LEVEL_SENSOR` (124), `ROAD_SEGMENT` (120), `DRAIN` (120), `MANHOLE` (119), `POWER_FEEDER` (116), `PUMP_STATION` (97).

### 3.5 `asset_dependencies.csv` (2,500 rows, 8 columns)
- **Columns & Dtypes**:
  `dependency_id` (object), `from_asset_id` (object), `to_asset_id` (object), `relationship_type` (object), `relationship_status` (object), `evidence_id` (object), `confidence_score` (float64), `provenance` (object).
- **Missing Values**: 0 nulls across all 8 columns.

### 3.6 `sensor_readings.csv` (15,000 rows, 17 columns)
- **Columns & Dtypes**:
  `reading_id` (object), `sensor_id` (object), `timestamp` (object), `latitude` (float64), `longitude` (float64), `rainfall_15min_mm` (float64), `rainfall_1hr_mm` (float64), `water_level_cm` (float64), `water_level_change_cm` (float64), `flow_rate_lps` (float64), `drainage_capacity_pct` (float64), `blockage_probability` (float64), `pump_status` (object), `sensor_health` (object), `data_quality` (object), `is_synthetic` (bool), `provenance` (object).
- **Missing Values**: 0 nulls across all 17 columns.
- **Unique Sensors Monitored**: 595 distinct `sensor_id`s.

### 3.7 `social_media_posts.csv` (6,000 rows, 15 columns)
- **Columns & Dtypes**:
  `post_id` (object), `platform` (object), `platform_post_id` (object), `post_time` (object), `collected_at` (object), `text_original` (object), `language` (object), `latitude` (float64), `longitude` (float64), `location_text` (object), `permalink` (float64/all null), `collection_method` (object), `collection_status` (object), `is_synthetic` (bool), `provenance` (object).
- **Missing Values**:
  - `latitude`: 2,669 missing (44.48%) — reflects realistic missing geotags in social posts.
  - `longitude`: 2,669 missing (44.48%)
  - `location_text`: 1,510 missing (25.17%)
  - `permalink`: 6,000 missing (100.0%) — intentional, as synthetic posts have no web URLs.

### 3.8 `social_post_analysis.csv` (6,000 rows, 11 columns)
- **Columns & Dtypes**:
  `post_id` (object), `incident_relevance_score` (float64), `predicted_event_type` (object), `extracted_location` (object), `event_time_estimate` (object), `duplicate_score` (float64), `spam_or_repost_flag` (bool), `evidence_quality` (object), `model_version` (object), `review_status` (object), `is_synthetic` (bool).
- **Missing Values**:
  - `extracted_location`: 900 missing (15.0%) — when NLP NER fails to extract a named location.

### 3.9 `flood_incident_observations.csv` (5,000 rows, 20 columns)
- **Columns & Dtypes**:
  `flood_observation_id` (object), `event_type` (object), `observed_time` (object), `reported_time` (object), `locality` (object), `latitude` (float64), `longitude` (float64), `catchment_id` (object), `asset_id` (object), `water_depth_cm` (float64), `affected_length_m` (float64), `road_passability` (object), `rainfall_15min_mm` (float64), `water_level_cm` (float64), `flow_rate_lps` (float64), `blockage_status` (object), `verification_status` (object), `measurement_method` (object), `is_synthetic` (bool), `provenance` (object).
- **Missing Values**:
  - `water_depth_cm`: 2,874 missing (57.48%) — when water depth was not quantitatively measured.
  - `affected_length_m`: 3,564 missing (71.28%) — when flood road segment length was unmeasured.

### 3.10 `pairwise_relationship_training.csv` (12,000 rows, 12 columns)
- **Columns & Dtypes**:
  `pair_id` (object), `observation_id_a` (object), `observation_id_b` (object), `time_diff_minutes` (float64), `geo_distance_km` (float64), `text_similarity` (float64), `same_locality` (int64), `same_source` (int64), `reliability_product` (float64), `relation_label` (object), `is_synthetic` (bool), `provenance` (object).
- **Missing Values**: 0 nulls across all 12 columns.

### 3.11 `relationship_evidence_graph_edges.csv` (18,000 rows, 12 columns)
- **Columns & Dtypes**:
  `edge_id` (object), `source_entity_type` (object), `source_entity_id` (object), `target_entity_type` (object), `target_entity_id` (object), `relationship_type` (object), `confidence_score` (float64), `rule_id` (object), `evidence_ids` (object), `verification_status` (object), `explanation` (object), `is_synthetic` (bool).
- **Missing Values**: 0 nulls across all 12 columns.

### 3.12 `routes.csv` (40 rows, 11 columns)
- **Columns & Dtypes**:
  `route_id` (object), `origin_name` (object), `origin_lat` (float64), `origin_lon` (float64), `destination_name` (object), `destination_lat` (float64), `destination_lon` (float64), `travel_mode` (object), `routing_provider` (object), `geometry_status` (object), `is_synthetic` (bool).
- **Missing Values**: 0 nulls across all 11 columns.

### 3.13 `route_impacts.csv` (5,000 rows, 11 columns)
- **Columns & Dtypes**:
  `route_impact_id` (object), `route_id` (object), `incident_id` (object), `segment_id` (object), `impact_type` (object), `distance_from_route_m` (float64), `valid_from` (object), `valid_to` (object), `impact_confidence` (float64), `verification_status` (object), `is_synthetic` (bool).
- **Missing Values**: 0 nulls across all 11 columns.

---

## 4. Referential Integrity & Entity Grounding Audit

A systematic check was executed across all cross-table entity references.

### 4.1 Observations & Incident Clusters
- `observation_incident_links.csv` -> `observations.csv`: **21,000 / 21,000 (100.0%)** valid matches. Zero orphaned links.
- `observation_incident_links.csv` -> `incident_clusters.csv`: **21,000 / 21,000 (100.0%)** valid matches.
- `observations.csv` `candidate_incident_id`:
  - 21,000 rows are populated and **100.0%** match `incident_clusters.csv`.
  - 4,000 rows are `NULL` (unassigned candidate singleton observations).
  - Incidents represented in links: 4,173 unique incidents out of 4,200 total clusters (mean 5.03 observations per incident).

### 4.2 Sensors & Infrastructure Assets
- `sensor_readings.csv` `sensor_id` -> `infrastructure_assets.csv` `asset_id`:
  - 15,000 readings reference 595 unique sensor IDs.
  - **100.0% (595 / 595)** resolve directly to valid assets in `infrastructure_assets.csv`.
  - Associated asset types: `TRAFFIC_SENSOR` (129), `RAIN_GAUGE` (125), `WATER_LEVEL_SENSOR` (124), `DRAIN` (120), `PUMP_STATION` (97).

### 4.3 Flood Observations & Infrastructure
- `flood_incident_observations.csv` `asset_id` -> `infrastructure_assets.csv` `asset_id`:
  - 5,000 flood observations reference 461 unique asset IDs.
  - **100.0% (461 / 461)** resolve directly to valid assets in `infrastructure_assets.csv`.

### 4.4 Asset Dependencies
- `asset_dependencies.csv` `from_asset_id` & `to_asset_id` -> `infrastructure_assets.csv`:
  - 2,500 dependency edges.
  - **100.0% (2,500 / 2,500)** `from_asset_id` resolved.
  - **100.0% (2,500 / 2,500)** `to_asset_id` resolved.

### 4.5 Pairwise Training Links
- `pairwise_relationship_training.csv` -> `observations.csv`:
  - `observation_id_a`: **12,000 / 12,000 (100.0%)** resolved.
  - `observation_id_b`: **12,000 / 12,000 (100.0%)** resolved.
  - Unordered unique pairs: 12,000 (zero symmetric duplicate pairs, zero self-pairs `a == b`).

### 4.6 Route Impacts Links
- `route_impacts.csv` -> `routes.csv`: **5,000 / 5,000 (100.0%)** match the 40 routes.
- `route_impacts.csv` -> `incident_clusters.csv`: **5,000 / 5,000 (100.0%)** match valid incident clusters.
- `segment_id`: 2,161 unique synthetic segment IDs (`SEG-00001` to `SEG-02161`). There is no separate segments table; these represent discretized route intervals.

### 4.7 Knowledge Graph Edge Integrity
- `relationship_evidence_graph_edges.csv` (18,000 edges):
  - **Source Entities**:
    - `OBSERVATION`: 4,520 edges (4,127 unique IDs) -> **100.0%** match `observations.csv`.
    - `INCIDENT`: 4,470 edges (2,719 unique IDs) -> **100.0%** match `incident_clusters.csv`.
    - `ASSET`: 4,470 edges (1,166 unique IDs) -> **100.0%** match `infrastructure_assets.csv`.
    - `SOCIAL_POST`: 4,540 edges (3,190 unique IDs) -> **100.0%** match `social_media_posts.csv`.
  - **Target Entities**:
    - `ROUTE`: 3,659 edges (40 unique IDs) -> **100.0%** match `routes.csv`.
    - `ASSET`: 3,634 edges (1,137 unique IDs) -> **100.0%** match `infrastructure_assets.csv`.
    - `OBSERVATION`: 3,604 edges (3,347 unique IDs) -> **100.0%** match `observations.csv`.
    - `INCIDENT`: 3,566 edges (2,421 unique IDs) -> **100.0%** match `incident_clusters.csv`.
    - `LOCATION`: 3,537 edges point to 8 synthetic regional zone anchor IDs (`LOC-01` to `LOC-08`) representing Western Suburbs sectors (Bandra to Dahisar).
  - **Supporting Evidence IDs**:
    - 36,000 observation references in `evidence_ids` -> **100.0% (36,000 / 36,000)** match valid `observation_id`s in `observations.csv`.

---

## 5. Label Distributions & Class Imbalance Analysis

### 5.1 Pairwise Relationship Labels (`pairwise_relationship_training.csv`)
| Label | Count | Percentage | Imbalance Notes |
| :--- | :---: | :---: | :--- |
| `SAME_INCIDENT` | 4,075 | 33.96% | Majority positive class for incident merging. |
| `UNRELATED` | 2,676 | 22.30% | Clear negative class (large spatio-temporal distance). |
| `RELATED_BUT_DISTINCT` | 2,367 | 19.73% | Challenging boundary class (correlated causal incidents). |
| `DUPLICATE` | 1,886 | 15.72% | Near-identical re-reports of identical incident. |
| `CONTRADICTORY` | 996 | 8.30% | **Severe Minority Class**. Conflict detection requires class weighting or focal loss. |

### 5.2 Incident Severity Distribution (`incident_clusters.csv`)
| Severity Level | Count | Percentage | Imbalance Notes |
| :--- | :---: | :---: | :--- |
| `MODERATE` | 1,774 | 42.24% | Typical urban bottleneck / minor waterlogging. |
| `MAJOR` | 1,264 | 30.10% | Significant road closures and rail delays. |
| `LOW` | 743 | 17.69% | Minor localized alerts. |
| `CRITICAL` | 419 | 9.98% | **Critical Emergency Minority Class** (~10:1 ratio vs Moderate). |

### 5.3 Route Disruption Impact Types (`route_impacts.csv`)
| Impact Category | Count | Percentage |
| :--- | :---: | :---: |
| `ON_ROUTE` | 1,041 | 20.82% |
| `POSSIBLE_DELAY` | 1,004 | 20.08% |
| `NEAR_ROUTE_CANDIDATE` | 999 | 19.98% |
| `NO_CONFIRMED_IMPACT` | 998 | 19.96% |
| `CLOSURE` | 958 | 19.16% |

*(Note: Synthetic impact distribution is intentionally balanced across all 5 classes at ~20% each).*

### 5.4 Road Passability (`flood_incident_observations.csv`)
| Passability Status | Count | Percentage |
| :--- | :---: | :---: |
| `RESTRICTED` | 1,031 | 20.62% |
| `UNKNOWN` | 1,007 | 20.14% |
| `SLOW` | 1,005 | 20.10% |
| `CLOSED` | 986 | 19.72% |
| `OPEN` | 971 | 19.42% |

### 5.5 Sensor Health Status (`sensor_readings.csv`)
| Health Status | Count | Percentage | Notes |
| :--- | :---: | :---: | :--- |
| `HEALTHY` | 9,017 | 60.11% | Majority nominal operational telemetry. |
| `DEGRADED` | 3,020 | 20.13% | Partial brownout or sensor drift. |
| `OFFLINE` | 2,963 | 19.75% | Communication dropout or power cut. |

---

## 6. Critical Data Leakage & Synthetic Artifact Analysis

### 6.1 CRITICAL HAZARD 1: Ground-Truth Cluster Label in Input Stream
- **Finding**: In `observations.csv`, the column `candidate_incident_id` contains the ground-truth canonical cluster ID for 21,000 out of 25,000 observations (84.0%).
- **Leakage Risk**: If an ML model (e.g. incident deduplication, pairwise correlation, or cluster aggregator) includes `candidate_incident_id` in its feature set, it will achieve artificially perfect 100% accuracy during testing by memorizing the target.
- **Mandatory Policy**:
  - `candidate_incident_id` must be **strictly excluded** from all inference feature transformations.
  - It must be used exclusively during offline clustering evaluation (e.g. Adjusted Rand Index, V-Measure, B-Cubed precision/recall).

### 6.2 CRITICAL HAZARD 2: Model Pseudo-Labels in `social_post_analysis.csv`
- **Finding**: Columns `incident_relevance_score`, `predicted_event_type`, `extracted_location`, and `duplicate_score` in `social_post_analysis.csv` are model-generated synthetic features.
- **Policy**: Do not treat these as objective ground-truth labels. If an NLP model is trained on social media text, it must predict from raw `social_media_posts.text_original`, not from `social_post_analysis.csv`.

### 6.3 Artificial Correlations in Pairwise Training Data
The audit revealed strong synthetic separation boundaries in `pairwise_relationship_training.csv`:
- `DUPLICATE`: Geo Distance mean = **0.09 km**, Time Diff mean = **2.9 min**, Text Similarity = **0.90**, Same Locality = **100%**.
- `SAME_INCIDENT`: Geo Distance mean = **0.36 km**, Time Diff mean = **19.9 min**, Text Similarity = **0.82**, Same Locality = **100%**.
- `RELATED_BUT_DISTINCT`: Geo Distance mean = **1.23 km**, Time Diff mean = **89.3 min**, Text Similarity = **0.55**, Same Locality = **50.0%**.
- `UNRELATED`: Geo Distance mean = **4.07 km**, Time Diff mean = **370.1 min**, Text Similarity = **0.23**, Same Locality = **32.6%**.

*ML Warning*: Because these synthetic pairs were generated with clear mathematical distance envelopes, linear and tree models will easily achieve >95% accuracy. Real-world urban multimodal pairs contain significant noise, conflicting reports, delayed citizen reports, and ambiguous landmarks. Evaluation metrics on synthetic data must be clearly documented as simulated benchmark performance.

### 6.4 Synthetic Text Duplication
In `observations.csv`, 23,590 rows share templated text formulations (e.g. repeated phrasing generated by parameterized templates). Models should not rely on exact string hashing, but rather on semantic vector embeddings and spatio-temporal distance functions.

---

## 7. Synthetic Provenance Verification

All 13 CSV files are synthetically generated for schema design, simulation, and software prototyping:
- Provenance flags: Every file contains explicit boolean tags (`is_synthetic: True`, `is_simulated: True`, `provenance: "SYNTHETIC_GENERATED"`).
- `social_media_posts.csv`: 100% of permalinks are `NULL`. These are synthetic narrative templates, not live posts scraped from X/Twitter or Instagram.
- Telemetry: Sensor readings are generated from parameterized physical equations, not live municipal SCADA systems.
- Route Geometry: Routes represent simulated corridors across the Western Suburbs, not real-time GPS probe streams.

**Strict Operational Rule**: Never describe synthetic events, social posts, sensor readings, or route examples as real-time observations in application UIs, APIs, or documentation.

---

## 8. Separation of Inference Inputs vs Evaluation Ground-Truth

To maintain absolute scientific validity and prevent label contamination, the dataset columns are partitioned into strictly separated tiers:

| Tier | Dataset | Columns | Usage Policy |
| :--- | :--- | :--- | :--- |
| **Tier 1: Inference Inputs (Permitted Features)** | `observations.csv` | `source_type`, `event_time`, `text_original`, `text_normalized`, `latitude`, `longitude`, `location_text`, `location_accuracy_m`, `source_reliability_score`, `is_duplicate_candidate` | Raw input multimodal stream features. |
| | `sensor_readings.csv` | `timestamp`, `latitude`, `longitude`, `rainfall_15min_mm`, `rainfall_1hr_mm`, `water_level_cm`, `water_level_change_cm`, `flow_rate_lps`, `drainage_capacity_pct`, `pump_status`, `sensor_health` | Environmental telemetry inputs. |
| | `infrastructure_assets.csv` | `asset_type`, `latitude`, `longitude`, `elevation_m`, `design_capacity_lps`, `catchment_id` | Static GIS and physical graph features. |
| | `asset_dependencies.csv` | `from_asset_id`, `to_asset_id`, `relationship_type`, `confidence_score` | Graph topology edges for NetworkX reasoning. |
| | `routes.csv` | `origin_lat`, `origin_lon`, `destination_lat`, `destination_lon`, `travel_mode` | Query inputs for commuter route planning. |
| **Tier 2: Intermediate Inference Candidates** | `pairwise_relationship_training.csv` | `time_diff_minutes`, `geo_distance_km`, `text_similarity`, `same_locality`, `same_source`, `reliability_product` | Pairwise candidate features for incident linking. |
| | `observation_incident_links.csv` | `membership_score`, `decision_type` | Candidate cluster associations generated by models. |
| **Tier 3: Evaluation-Only Ground Truth (STRICTLY PROHIBITED AS FEATURES)** | `observations.csv` | `candidate_incident_id` | **LEAKAGE HAZARD.** Target for cluster purity and ARI evaluation only. |
| | `pairwise_relationship_training.csv` | `relation_label` | Target label for pairwise model training/testing. |
| | `incident_clusters.csv` | `incident_id`, `incident_type`, `severity`, `cluster_status` | Benchmark clusters for grouping evaluation. |
| | `route_impacts.csv` | `impact_type` | Target label for route impact classification. |
| | `flood_incident_observations.csv` | `road_passability`, `blockage_status` | Target labels for flood passability prediction. |

---

## 9. Safe Train / Validation / Test Split Strategy

### The Cross-Incident Leakage Vulnerability
If observations or pairs are split randomly (e.g. simple `train_test_split`), observations from the **same physical incident** will end up in both the training set and the test set. Because observations from the same incident share near-identical geographic coordinates, timestamps, and wording, the model would simply memorize the incident cluster rather than learning generalized incident correlation rules.

### Recommended Disjoint Incident-Group Split
All data must be partitioned strictly by **Canonical Incident Cluster (`incident_id`)**:
1. Partition the 4,200 unique incident IDs into disjoint groups using a fixed random seed (`seed=42`):
   - **Train Incidents**: 2,940 incidents (70.0%)
   - **Validation Incidents**: 630 incidents (15.0%)
   - **Test Incidents**: 630 incidents (15.0%)
   - **Overlap**: Strictly **0** incidents shared between any splits.
2. **Observations Split**:
   - Observations linked to Train Incidents -> Train Set: **14,738 observations (58.95%)**
   - Observations linked to Val Incidents -> Val Set: **3,155 observations (12.62%)**
   - Observations linked to Test Incidents -> Test Set: **3,107 observations (12.43%)**
   - Unclustered Observations (4,000 singletons) -> Partitioned via a spatial-temporal grid hash to ensure regional/temporal separation.
3. **Pairwise Training Pairs Split**:
   - A pair $(Obs_A, Obs_B)$ is assigned to Train **if and only if** both observations belong to Train incidents.
   - Any cross-incident pair linking an incident from Train to Val/Test is **dropped from evaluation sets** to guarantee zero information leakage.

---

## 10. Summary of Detected Issues & Action Items

| Item | Problem / Finding | Impact | Required Action in Phase 2 |
| :---: | :--- | :--- | :--- |
| **1** | `candidate_incident_id` in `observations.csv` | Direct target leakage into observation stream | Quarantined; stripped from all feature extractors. |
| **2** | 4,000 unclustered observations | Missing candidate cluster IDs | Handled as true unclustered background noise / singletons. |
| **3** | Location entities `LOC-01` to `LOC-08` in graph | No dedicated `locations.csv` table | Mapped as 8 regional zone centroid nodes in NetworkX graph. |
| **4** | Missing permalinks in `social_media_posts.csv` | 100% null permalinks | Treated as synthetic demonstration data; permalink validation disabled. |
| **5** | Missing flood measurement depths | 57.5% null depth, 71.3% null length | Imputation / categorical masking required for flood regression models. |
| **6** | Severe class imbalance in pairwise labels | Only 8.3% `CONTRADICTORY` vs 34.0% `SAME_INCIDENT` | Weighted cross-entropy or focal loss in classifier training. |

---

## 11. Conclusion & Phase Gate

Phase 1 dataset audit is complete. All 13 CSV files are profiled, referential relationships verified, label skews quantified, and leakage risks isolated. 

**Status**: Ready for User Review and Sign-off before proceeding to Phase 2.
