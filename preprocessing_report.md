# UrbanPulse: Multimodal Event Preprocessing Report (Phase 2)

**Pipeline Version**: 1.0.0-phase2  
**Target Corridor**: Mumbai Western Suburbs (Bandra to Dahisar)  
**Dataset Input Directory**: `UrbanPulse_Expanded_Dataset`  
**Cleaned Output Directory**: `UrbanPulse_Expanded_Dataset/cleaned_output`  
**Execution Runtime**: 17.34 seconds  
**Total Raw Records Ingested & Standardized**: 51,000  
**Phase Status**: **PHASE 2 PREPROCESSING COMPLETE — TESTS VERIFIED**

---

## 1. Executive Summary & Pipeline Objectives

In Phase 2, we implemented the preprocessing, normalization, and deduplication engine for **UrbanPulse**. The pipeline ingests 51,000 raw multimodal records across 4 disparate data streams:
1. Unified Multimodal Observations (`observations.csv`: 25,000 records)
2. Social Media Posts (`social_media_posts.csv`: 6,000 records)
3. SCADA Sensor Telemetry (`sensor_readings.csv`: 15,000 records)
4. Field Flood Incident Observations (`flood_incident_observations.csv`: 5,000 records)

### Core Mandates Met:
- **No Silent Data Fabrication**: Records with missing timestamps or coordinates are **never deleted or silently imputed**. They are preserved in full with granular quality flags.
- **Consistent Timezone Alignment**: 100% of timestamps are parsed and normalized into Indian Standard Time (IST, `UTC+05:30`), with epoch milliseconds preserved.
- **Spatial Validation & Corridor Fencing**: Preserved raw latitude/longitude while validating against the Mumbai Western Suburbs bounding box ($18.98^\circ\text{N}$ to $19.28^\circ\text{N}$, $72.78^\circ\text{E}$ to $72.90^\circ\text{E}$).
- **Two-Tier Deduplication**:
  - Exact duplicates detected via unique source record IDs and deterministic content fingerprinting (source type, locality, 5-minute time bucket, normalized text).
  - Social media reposts / copy-pastes detected using token and character $n$-gram Jaccard similarity ($\ge 0.82$) within a 36-hour temporal window.
- **Strict Separation of Duplicate vs Same-Incident**: Ensured that duplicate detection (redundant transmissions of the same signal) is strictly decoupled from incident correlation (distinct multimodal observations of the same physical event).
- **Multi-Dimensional Temporal & Trust Preservation**: Independently preserved `observation_time`, `publication_time`, `collection_time`, and `source_reliability`.
- **Standardized Unified Event Output**: Produced a 51,000-row canonical event table with standardized schema for downstream correlation algorithms.

---

## 2. Ingestion & Preprocessing Metrics Summary

| Stream Name | Input Raw Rows | Standardized Rows | Missing Coords | Exact Dupes | Reposts / Copies | Mean Quality Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `observations.csv` | 25,000 | 25,000 | 1,501 (6.00%) | 2,955 (11.82%) | 0 | 0.9575 |
| `social_media_posts.csv` | 6,000 | 6,000 | 2,669 (44.48%) | 0 | 270 (4.50%) | 0.8443 |
| `sensor_readings.csv` | 15,000 | 15,000 | 0 (0.00%) | 0 | 0 | 0.9800 |
| `flood_incident_observations.csv` | 5,000 | 5,000 | 0 (0.00%) | 0 | 0 | 0.9800 |
| **UNIFIED TOTAL** | **51,000** | **51,000** | **4,170 (8.18%)** | **2,955 (5.79%)** | **270 (0.53%)** | **0.9479** |

---

## 3. Normalization Rules & Domain Taxonomies

### 3.1 Timestamp Normalization
- All raw timestamps (ISO strings, UTC timestamps with `Z`, space-delimited timestamps, or naive timestamps) are converted to timezone-aware ISO 8601 strings in Indian Standard Time (`YYYY-MM-DDTHH:MM:SS+05:30`).
- Both the formatted string (`observation_time_iso`) and unix epoch in milliseconds (`observation_time_epoch_ms`) are recorded.
- **Quality Status**:
  - `VALID`: 51,000 / 51,000 (100.0%) valid timestamps resolved.
  - `MISSING`: 0 records.
  - `INVALID_FORMAT`: 0 records.

### 3.2 Geographic Coordinates & Western Suburbs Bounding Box
- **Raw Coordinates Preserved**: `latitude_raw` and `longitude_raw` retain the original unmodified inputs.
- **Corridor Boundaries**:
  $$\text{Latitude}: [18.9800^\circ\text{N}, 19.2800^\circ\text{N}], \quad \text{Longitude}: [72.7800^\circ\text{E}, 72.9000^\circ\text{E}]$$
- **Spatial Quality Classifications**:
  - `VALID_IN_CORRIDOR`: 46,830 events (91.82%) — within the Western Suburbs bounding box.
  - `VALID_OUT_OF_BOUNDS`: 0 events.
  - `MISSING_COORDINATES`: 4,170 events (8.18%) — 2,669 in social media posts, 1,501 in observations. Retained with `is_geocoded = False`.
  - `INVALID_COORDINATES`: 0 events.

### 3.3 Locality & Regional Zone Normalization
- Localities are normalized to clean title-case suburban names (`Andheri West`, `Borivali East`, `Goregaon`, `Versova`, `Jogeshwari`, `Kandivali`, `Dahisar`, `Malad`).
- Mapped to regional zone anchors corresponding to the Knowledge Graph topology:
  - `LOC-01`: Bandra, Khar, Santacruz
  - `LOC-02`: Vile Parle, Juhu, Versova
  - `LOC-03`: Andheri
  - `LOC-04`: Jogeshwari, Oshiwara
  - `LOC-05`: Goregaon
  - `LOC-06`: Malad
  - `LOC-07`: Kandivali
  - `LOC-08`: Borivali, Dahisar

### 3.4 Canonical Event Taxonomy
Harmonized all raw event type variations into standardized incident classes:
- Hydrology: `ROAD_WATERLOGGING`, `WATER_LEVEL_RISE`, `SUBWAY_INUNDATION`, `MANHOLE_OVERFLOW`, `DRAIN_BLOCKAGE`, `PUMP_FAILURE`, `HEAVY_RAINFALL`
- Traffic & Transit: `TRAFFIC_SLOWDOWN`, `ROAD_CLOSURE`, `ROAD_ACCIDENT`, `TRANSIT_DELAY`
- Infrastructure: `RAIL_SIGNAL_FAILURE`, `OHE_WIRE_FAILURE`, `POWER_FAILURE`, `SENSOR_ANOMALY`
- Civil: `TREE_FALL`, `FIRE_REPORT`, `GENERAL_INCIDENT`

---

## 4. Quality Flagging (Zero Silent Fabrication)

In strict adherence to instructions, **zero records were deleted**. Incomplete records are retained with explicit quality flags:
- `has_missing_timestamp`: boolean (0 records)
- `has_missing_coordinates`: boolean (4,170 records flagged)
- `has_missing_text`: boolean (0 records flagged)
- `is_usable_for_spatial_clustering`: boolean (46,830 records eligible)
- `is_usable_for_text_matching`: boolean (51,000 records eligible)

### Composite Quality Score Formula
$$Q = 0.25 \cdot \mathbf{1}_{\text{valid\_ts}} + 0.35 \cdot \mathbf{1}_{\text{in\_corridor\_coords}} + 0.20 \cdot \mathbf{1}_{\text{has\_text}} + 0.20 \cdot \mathbf{1}_{\text{verified\_reliability}}$$
- **Quality Score Statistics**:
  - Mean: **0.9479**
  - Median: **1.0000**
  - Min: **0.4500** (occurs only when coordinates are missing and reliability was imputed)
  - Max: **1.0000**

---

## 5. Deduplication & Social Repost Analysis

### 5.1 Exact Deduplication
- **Methodology**:
  1. Unique external source record IDs (`platform_post_id` in social feeds).
  2. Deterministic content fingerprinting: $\text{MD5}(\text{source\_type} \parallel \text{locality} \parallel \text{time\_5min\_bucket} \parallel \text{normalized\_text})$.
  3. Raw stream duplicate candidate flags.
- **Results**:
  - Flagged **2,955 exact duplicate records** (5.79% of total events).
  - Linked each duplicate to its primary event via `duplicate_of_id`.

### 5.2 Social Media Repost & Copy Detection
- **Methodology**:
  - Sanitized text (stripped mentions, URLs, punctuation).
  - Computed token Jaccard and character 3-gram Jaccard similarity across a 36-hour sliding window.
  - Threshold: $\ge 0.82$ similarity.
- **Results**:
  - Flagged **270 social media posts (4.50%)** as viral reposts, retweets, or copy-pasted forwarded messages.
  - Linked each repost to the earliest known post in the viral chain via `repost_parent_id`.
  - Stored `repost_similarity` score.

### 5.3 Architectural Independence: Deduplication vs Same-Incident Correlation
The pipeline enforces the critical conceptual boundary:
- **Deduplication** operates on *signal redundancy*: Did the pipeline receive the same transmission twice (e.g. re-ingested feed packet, automated bot repost)?
- **Incident Correlation** operates on *event corroboration*: Two completely different citizens or a citizen and a physical water-level sensor can independently report the same flood. They have different wording, different sources, and different perspectives. They are **never duplicates**, but they belong to the **same incident**.
- Enforced by `IndependenceGuard.verify_separation()` in code and test suite.

---

## 6. Preservation of Temporal Dimensions & Provenance

For every standardized event, four distinct temporal and provenance vectors are maintained without collision:
1. `observation_time_iso`: Timestamp of physical ground occurrence.
2. `publication_time_iso`: Timestamp of report publication (distinct for social posts & field reports).
3. `collection_time_iso`: Timestamp of pipeline ingestion.
4. `source_reliability`: Calibrated source reliability prior [0.0 - 1.0], with `reliability_imputed` tracking whether raw data provided a score.
5. `provenance`: Preserved as `SYNTHETIC_GENERATED`, with `is_synthetic: True`.

---

## 7. Standardized Event Output Schema

Every record in [unified_multimodal_events.csv](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/cleaned_output/unified_multimodal_events.csv) adheres to the schema:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `event_id` | string | Canonical standardized event ID (e.g. `EVT-OBS-0000001`, `EVT-POST-000001`) |
| `raw_id` | string | Source table raw record identifier |
| `source_modality` | string | `OBSERVATION`, `SOCIAL_POST`, `SENSOR_TELEMETRY`, `FLOOD_REPORT` |
| `source_channel` | string | Normalized ingestion feed channel |
| `canonical_event_type`| string | Normalized taxonomy event category |
| `raw_event_type` | string | Original unnormalized event type string |
| `observation_time_iso`| string | ISO 8601 with `+05:30` IST timezone |
| `observation_time_epoch_ms`| int | Milliseconds since Unix epoch |
| `publication_time_iso`| string | Timestamp of publication (or null) |
| `collection_time_iso` | string | Ingestion pipeline timestamp |
| `latitude` | float | Validated decimal degree latitude |
| `longitude` | float | Validated decimal degree longitude |
| `latitude_raw` | string/float| Unmodified raw input latitude |
| `longitude_raw` | string/float| Unmodified raw input longitude |
| `locality_normalized` | string | Standardized title-case suburban locality |
| `locality_zone` | string | Regional zone anchor (`LOC-01` to `LOC-08`) |
| `text_raw` | string | Unmodified original narrative |
| `text_cleaned` | string | Sanitized text for NLP embedding |
| `source_reliability` | float | Source trust weight [0.0 - 1.0] |
| `reliability_imputed` | bool | True if default prior was assigned |
| `provenance` | string | Data origin (`SYNTHETIC_GENERATED`) |
| `is_synthetic` | bool | Boolean synthetic indicator |
| `is_exact_duplicate` | bool | True if duplicate source ID or exact fingerprint |
| `duplicate_of_id` | string | Parent event ID if duplicate |
| `duplicate_type` | string | `EXACT_SOURCE_ID`, `EXACT_CONTENT_FINGERPRINT`, `RAW_STREAM_DUPLICATE_CANDIDATE` |
| `is_repost_or_copy` | bool | True if social media repost / forwarded copy |
| `repost_parent_id` | string | Earlier post ID in repost chain |
| `repost_similarity` | float | Similarity score to repost parent |
| `has_missing_timestamp`| bool | Missing timestamp indicator |
| `has_missing_coordinates`| bool | Missing coordinates indicator |
| `has_missing_text` | bool | Missing text description indicator |
| `spatial_quality` | string | `VALID_IN_CORRIDOR`, `VALID_OUT_OF_BOUNDS`, `MISSING_COORDINATES` |
| `data_quality_score` | float | Composite completeness score [0.0 - 1.0] |
| `is_usable_for_spatial_clustering`| bool | Flag for spatial DBSCAN / clustering |
| `is_usable_for_text_matching`| bool | Flag for TF-IDF / vector search |
| `evaluation_candidate_incident_id`| string | **QUARANTINED GROUND TRUTH FOR EVAL ONLY** |

---

## 8. Unit & Integration Test Suite Verification

The test suite [tests/test_preprocessing.py](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/tests/test_preprocessing.py) was executed with Python's built-in `unittest` runner. All 13 tests passed:

```text
test_contradictory_records_preservation ... ok
test_exact_duplicate_content_fingerprint ... ok
test_exact_duplicate_detection_by_source_id ... ok
test_independence_of_duplicate_from_same_incident ... ok
test_invalid_and_missing_coordinates ... ok
test_invalid_timestamp_format ... ok
test_missing_timestamp_handling ... ok
test_out_of_bounds_coordinates ... ok
test_preservation_of_temporal_and_reliability_dimensions ... ok
test_social_media_repost_detection ... ok
test_taxonomy_normalization ... ok
test_timezone_normalization ... ok
test_valid_in_corridor_coordinates ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.007s
OK
```

### Test Coverage Highlights:
1. **Missing Data Handling**: Verified that missing timestamps and coordinates do not cause crashes and do not fabricate synthetic defaults.
2. **Invalid Coordinates**: Verified that coordinates $>90^\circ$ or $<-90^\circ$ are flagged as `INVALID_COORDINATES` and excluded from spatial indexing while retaining the record.
3. **Contradictory Records**: Verified that contradictory reports (e.g. Report A: "Road open", Report B: "Road closed") are both preserved with independent reliability weights.
4. **Duplicate vs Incident Separation**: Verified that independent eyewitness reports can share an incident cluster without being flagged as duplicates.
5. **Repost Detection**: Verified that viral retweets are correctly linked to original parent posts across time windows.

---

## 9. Output Deliverables & File Locations

All Phase 2 deliverables have been generated and saved:

1. **Preprocessing Module**: [preprocessing/pipeline.py](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/preprocessing/pipeline.py)
2. **Normalizers**: [preprocessing/normalizers.py](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/preprocessing/normalizers.py)
3. **Deduplication Module**: [preprocessing/deduplicator.py](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/preprocessing/deduplicator.py)
4. **Event Schema**: [preprocessing/event_schema.py](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/preprocessing/event_schema.py)
5. **Configuration & Taxonomies**: [preprocessing/config.py](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/preprocessing/config.py)
6. **Top-Level Pipeline Runner**: [run_preprocessing.py](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/run_preprocessing.py)
7. **Test Suite**: [tests/test_preprocessing.py](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/tests/test_preprocessing.py)
8. **Cleaned Datasets Directory**: [cleaned_output/](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/cleaned_output)
   - [unified_multimodal_events.csv](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/cleaned_output/unified_multimodal_events.csv) (51,000 rows)
   - [standardized_observations.csv](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/cleaned_output/standardized_observations.csv) (25,000 rows)
   - [standardized_social_posts.csv](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/cleaned_output/standardized_social_posts.csv) (6,000 rows)
   - [standardized_sensor_readings.csv](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/cleaned_output/standardized_sensor_readings.csv) (15,000 rows)
   - [standardized_flood_observations.csv](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/cleaned_output/standardized_flood_observations.csv) (5,000 rows)
   - [preprocessing_metrics.json](file:///d:/Chaitrali/SEM%207/Urban%20flow/CodeAstra/UrbanPulse_Expanded_Dataset/cleaned_output/preprocessing_metrics.json)

---

## 10. Conclusion & Approval Request

Phase 2 is complete. All 51,000 multimodal events are standardized, quality-scored, deduplicated, and tested.

**Per instructions, execution has stopped after Phase 2. Awaiting your approval before proceeding to Phase 3.**
