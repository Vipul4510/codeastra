"""
UrbanPulse Standardized Event Schema
Defines the canonical multimodal event representation for all downstream algorithms.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

@dataclass
class StandardizedEvent:
    # 1. Primary & Source Identification
    event_id: str                          # Globally unique standardized event ID (e.g. EVT-OBS-0000001)
    raw_id: str                            # Original identifier in raw source table
    source_modality: str                   # OBSERVATION, SENSOR_TELEMETRY, SOCIAL_POST, FLOOD_REPORT
    source_channel: str                    # SENSOR, SOCIAL_MEDIA, CITIZEN_REPORT, TRAFFIC_FEED, etc.
    
    # 2. Event Classification
    canonical_event_type: str              # Normalized taxonomy (ROAD_WATERLOGGING, PUMP_FAILURE, etc.)
    raw_event_type: str                    # Original unmodified event type string
    
    # 3. Explicitly Separated Temporal Dimensions
    observation_time_iso: Optional[str]    # When event occurred / was observed on the ground (IST ISO 8601)
    observation_time_epoch_ms: Optional[int]
    publication_time_iso: Optional[str]    # When report was posted / filed (if distinct)
    collection_time_iso: Optional[str]     # When ingestion pipeline collected the data
    
    # 4. Spatial Attributes & Original Preservations
    latitude: Optional[float]              # Validated coordinate in decimal degrees
    longitude: Optional[float]             # Validated coordinate in decimal degrees
    latitude_raw: Any                      # Unmodified original latitude
    longitude_raw: Any                     # Unmodified original longitude
    locality_normalized: str               # Title-cased suburban locality (e.g. Andheri West)
    locality_zone: str                     # Western Suburbs regional zone (LOC-01 to LOC-08)
    
    # 5. Multimodal Content
    text_raw: str                          # Original unmodified narrative or post text
    text_cleaned: str                      # Sanitized text for NLP and embedding extraction
    
    # 6. Source Reliability & Provenance
    source_reliability: float              # Heuristic trust score [0.0 - 1.0]
    reliability_imputed: bool              # True if default prior was assigned due to missing score
    provenance: str                        # Provenance origin tag (e.g. SYNTHETIC_GENERATED)
    is_synthetic: bool                     # Explicit flag indicating simulated data
    
    # 7. Deduplication & Repost Dimensions
    is_exact_duplicate: bool               # True if identical source ID or content fingerprint
    duplicate_of_id: Optional[str]         # ID of the primary event this duplicates
    duplicate_type: Optional[str]          # EXACT_SOURCE_ID, EXACT_CONTENT_FINGERPRINT
    is_repost_or_copy: bool                # True if social media repost / forwarded copy
    repost_parent_id: Optional[str]        # Earlier post ID that this copies
    repost_similarity: Optional[float]     # Text similarity score to repost parent
    
    # 8. Data Quality & Completeness Flags (No Silent Imputation)
    has_missing_timestamp: bool
    has_missing_coordinates: bool
    has_missing_text: bool
    spatial_quality: str                   # VALID_IN_CORRIDOR, VALID_OUT_OF_BOUNDS, MISSING_COORDINATES, INVALID_COORDINATES
    data_quality_score: float              # Composite completeness metric [0.0 - 1.0]
    is_usable_for_spatial_clustering: bool # False if coordinates missing or invalid
    is_usable_for_text_matching: bool      # False if text missing or empty
    
    # 9. EVALUATION GROUND TRUTH (STRICTLY QUARANTINED FROM INFERENCE FEATURES)
    evaluation_candidate_incident_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def calculate_quality_score(
        has_ts: bool,
        spatial_quality: str,
        has_text: bool,
        has_reliability: bool
    ) -> float:
        """
        Calculates a transparent, weighted completeness rating:
        - Timestamp valid: 25%
        - Spatial coordinates valid in corridor: 35% (15% if valid but out of bounds, 0% if missing)
        - Text present: 20%
        - Reliability verified: 20%
        """
        score = 0.0
        if has_ts:
            score += 0.25
        
        if spatial_quality == "VALID_IN_CORRIDOR":
            score += 0.35
        elif spatial_quality == "VALID_OUT_OF_BOUNDS":
            score += 0.15
        
        if has_text:
            score += 0.20
            
        if has_reliability:
            score += 0.20
            
        return round(score, 4)
