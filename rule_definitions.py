"""
UrbanPulse Incident Correlation Rulebook & Configurable Rule Definitions
========================================================================
Defines hard compatibility rules, soft heuristic affinity scores,
category causality relationships, and contradiction matrices.

NOTICE ON HEURISTICS:
All weighted scores in this module represent domain heuristics calibrated for
multimodal urban event reasoning in Mumbai Western Suburbs. They do not claim
statistical real-world physical causation.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

# -------------------------------------------------------------------------
# 1. Configurable Parameters & Thresholds
# -------------------------------------------------------------------------

@dataclass
class CorrelationRuleConfig:
    """Configurable thresholds for candidate filtering, pairwise labels, and clustering."""
    # Candidate Generation Window
    candidate_max_distance_km: float = 5.0
    candidate_max_time_minutes: float = 120.0
    
    # Same-Incident Matching Thresholds (Strict spatial & temporal envelope)
    same_incident_max_distance_km: float = 0.85
    same_incident_max_time_minutes: float = 45.0
    same_incident_min_text_similarity: float = 0.40
    same_incident_min_heuristic_score: float = 0.65
    
    # Exact Duplicate Thresholds
    duplicate_max_distance_km: float = 0.15
    duplicate_max_time_minutes: float = 20.0
    duplicate_min_text_similarity: float = 0.85
    
    # Related-But-Distinct Thresholds (Broader corridor / causal horizon)
    related_distinct_max_distance_km: float = 3.5
    related_distinct_max_time_minutes: float = 180.0
    related_distinct_min_heuristic_score: float = 0.35
    
    # Social Repost Window & Similarity
    repost_similarity_threshold: float = 0.80
    repost_time_window_hours: float = 36.0
    
    # Cluster Formation Threshold (Connected components cutoff)
    clustering_confidence_threshold: float = 0.65

# -------------------------------------------------------------------------
# 2. Category Compatibility, Causality & Contradiction Matrices
# -------------------------------------------------------------------------

# Causal & Symptomatic Event Relationships in Mumbai Monsoon Disruption
# Key: Primary Cause -> Set of secondary correlated consequences
CAUSAL_RELATIONSHIPS: Dict[str, Set[str]] = {
    "HEAVY_RAINFALL": {
        "ROAD_WATERLOGGING", "SUBWAY_INUNDATION", "WATER_LEVEL_RISE",
        "MANHOLE_OVERFLOW", "DRAIN_BLOCKAGE", "TRAFFIC_SLOWDOWN", "TREE_FALL"
    },
    "WATER_LEVEL_RISE": {
        "ROAD_WATERLOGGING", "SUBWAY_INUNDATION", "MANHOLE_OVERFLOW", "DRAIN_BLOCKAGE"
    },
    "DRAIN_BLOCKAGE": {
        "ROAD_WATERLOGGING", "MANHOLE_OVERFLOW", "WATER_LEVEL_RISE"
    },
    "PUMP_FAILURE": {
        "ROAD_WATERLOGGING", "SUBWAY_INUNDATION", "WATER_LEVEL_RISE"
    },
    "ROAD_WATERLOGGING": {
        "TRAFFIC_SLOWDOWN", "ROAD_CLOSURE", "TRANSIT_DELAY", "SUBWAY_INUNDATION"
    },
    "SUBWAY_INUNDATION": {
        "TRAFFIC_SLOWDOWN", "ROAD_CLOSURE", "TRANSIT_DELAY"
    },
    "ROAD_ACCIDENT": {
        "TRAFFIC_SLOWDOWN", "ROAD_CLOSURE"
    },
    "OHE_WIRE_FAILURE": {
        "TRANSIT_DELAY", "RAIL_SIGNAL_FAILURE", "POWER_FAILURE"
    },
    "RAIL_SIGNAL_FAILURE": {
        "TRANSIT_DELAY"
    },
    "POWER_FAILURE": {
        "RAIL_SIGNAL_FAILURE", "PUMP_FAILURE", "TRAFFIC_SLOWDOWN"
    },
    "TREE_FALL": {
        "ROAD_CLOSURE", "TRAFFIC_SLOWDOWN", "OHE_WIRE_FAILURE", "POWER_FAILURE"
    }
}

# Incompatible / Mutually Exclusive Phrases indicating direct CONTRADICTION
CONTRADICTION_KEYWORDS_PAIRS: List[Tuple[List[str], List[str]]] = [
    # Positive state (Clear/Dry/Open) vs Negative state (Flooded/Submerged/Blocked)
    (["clear", "dry", "open", "passable", "normal traffic", "no waterlogging"],
     ["waterlogged", "submerged", "flooded", "closed", "blocked", "impassable", "halted", "overflow"]),
    (["traffic moving fast", "smooth flow", "no delay"],
     ["standstill", "gridlock", "jammed", "paralyzed", "deadlock", "crawling"]),
    (["power restored", "supply normal"],
     ["blackout", "power failure", "outage", "sparking", "line snapped"]),
    (["trains running normally", "on time"],
     ["trains cancelled", "track submerged", "service halted", "signal red"]),
]

# -------------------------------------------------------------------------
# 3. Spatial Math & Haversine Distance
# -------------------------------------------------------------------------

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates using Haversine formula.
    """
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 4)

# -------------------------------------------------------------------------
# 4. Text & Semantic Similarity Utilities
# -------------------------------------------------------------------------

def compute_token_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Computes word token Jaccard similarity between two cleaned strings."""
    if not text_a or not text_b:
        return 0.0
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersect = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return round(intersect / union, 4) if union > 0 else 0.0

def detect_contradictory_text(text_a: str, text_b: str) -> Tuple[bool, str]:
    """
    Detects whether two texts contain mutually exclusive factual assertions about the same location.
    """
    if not text_a or not text_b:
        return False, ""
    
    ta = text_a.lower()
    tb = text_b.lower()
    
    for pos_list, neg_list in CONTRADICTION_KEYWORDS_PAIRS:
        has_pos_a = any(k in ta for k in pos_list)
        has_neg_a = any(k in ta for k in neg_list)
        has_pos_b = any(k in tb for k in pos_list)
        has_neg_b = any(k in tb for k in neg_list)
        
        # If A asserts positive and B asserts negative (or vice versa) without both being mixed
        if (has_pos_a and not has_neg_a and has_neg_b and not has_pos_b) or \
           (has_neg_a and not has_pos_a and has_pos_b and not has_neg_b):
            matched_pos = [k for k in pos_list if k in ta or k in tb]
            matched_neg = [k for k in neg_list if k in ta or k in tb]
            return True, f"Conflict: Assertion of normal state ('{matched_pos[0]}') contradicts assertion of disruption ('{matched_neg[0]}')"
            
    return False, ""

# -------------------------------------------------------------------------
# 5. Core Compatibility Rules & Evaluator Functions
# -------------------------------------------------------------------------

def check_hard_incompatibility(
    dist_km: Optional[float],
    time_diff_min: float,
    config: CorrelationRuleConfig
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Applies hard distance and time horizon cutoffs.
    Returns: (is_incompatible, label, explanation)
    """
    # 1. Hard Temporal Cutoff
    if time_diff_min > config.candidate_max_time_minutes:
        return (
            True,
            "UNRELATED",
            f"Exceeds maximum candidate temporal window ({time_diff_min:.1f} min > {config.candidate_max_time_minutes:.1f} min)."
        )
        
    # 2. Hard Spatial Cutoff (if coordinates exist)
    if dist_km is not None and dist_km > config.candidate_max_distance_km:
        return (
            True,
            "UNRELATED",
            f"Exceeds maximum candidate geographic distance ({dist_km:.2f} km > {config.candidate_max_distance_km:.2f} km)."
        )
        
    return False, None, None


def check_exact_or_repost_duplicate(
    event_a: Dict[str, Any],
    event_b: Dict[str, Any],
    dist_km: Optional[float],
    time_diff_min: float,
    text_sim: float,
    config: CorrelationRuleConfig
) -> Tuple[bool, Optional[str], Optional[str], int]:
    """
    Detects if event pair represents redundant duplicates or viral social reposts.
    Returns: (is_duplicate, label, explanation, evidence_chain_count)
    """
    # Check 1: Explicit duplicate flag from ingestion / pipeline
    dupe_b = event_b.get("duplicate_of_id")
    dupe_a = event_a.get("duplicate_of_id")
    eid_a = event_a.get("event_id")
    eid_b = event_b.get("event_id")
    if (dupe_b and eid_a and dupe_b == eid_a) or (dupe_a and eid_b and dupe_a == eid_b):
        return (
            True,
            "DUPLICATE",
            f"Identical source record or exact ingestion fingerprint match ({event_b.get('duplicate_type') or event_a.get('duplicate_type')}).",
            1  # Evidence chain count: 1 (single evidence source)
        )
        
    # Check 2: Social media repost / forwarded copy chain
    rep_b = event_b.get("repost_parent_id")
    rep_a = event_a.get("repost_parent_id")
    raw_a = event_a.get("raw_id")
    raw_b = event_b.get("raw_id")
    if (rep_b and raw_a and rep_b == raw_a) or (rep_a and raw_b and rep_a == raw_b):
        return (
            True,
            "DUPLICATE",
            f"Social media viral repost/copy detected (similarity={event_b.get('repost_similarity', text_sim):.2f}). Treated as one evidence chain.",
            1
        )
        
    # Check 3: Near-instant identical text duplicate at same location
    is_same_locality = (
        event_a.get("locality_normalized") and
        event_a.get("locality_normalized") == event_b.get("locality_normalized")
    )
    if text_sim >= config.duplicate_min_text_similarity and \
       time_diff_min <= config.duplicate_max_time_minutes and \
       (dist_km is None or dist_km <= config.duplicate_max_distance_km or is_same_locality):
        return (
            True,
            "DUPLICATE",
            f"Near-identical text ({text_sim:.2f}) reported in immediate temporal burst ({time_diff_min:.1f} min).",
            1
        )
        
    return False, None, None, 2


def check_contradiction(
    event_a: Dict[str, Any],
    event_b: Dict[str, Any],
    dist_km: Optional[float],
    time_diff_min: float,
    config: CorrelationRuleConfig
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Rule 3: Detect mutually exclusive factual claims at the same location and time.
    """
    # Contradictions require localized spatio-temporal co-occurrence
    is_co_located = False
    if dist_km is not None and dist_km <= 0.75:
        is_co_located = True
    elif dist_km is None and event_a.get("locality_normalized") == event_b.get("locality_normalized"):
        is_co_located = True
        
    if is_co_located and time_diff_min <= 60.0:
        is_contra, reason = detect_contradictory_text(
            event_a.get("text_raw", ""),
            event_b.get("text_raw", "")
        )
        if is_contra:
            return (
                True,
                "CONTRADICTORY",
                f"Contradictory observations: {reason} at {event_a.get('locality_normalized')} within {time_diff_min:.1f} min. Preserved for operator audit."
            )
            
    return False, None, None


def calculate_heuristic_correlation_score(
    dist_km: Optional[float],
    time_diff_min: float,
    text_sim: float,
    cat_match_type: str,
    same_source: bool,
    rel_product: float
) -> float:
    """
    Calculates a heuristic multimodal affinity score [0.0 - 1.0].
    
    Weights formulation:
    - Spatial Proximity (0.35): decay over 1.0 km
    - Temporal Proximity (0.25): decay over 60 min
    - Category Compatibility (0.20): exact match (1.0), causal pair (0.75), compatible (0.40)
    - Text Similarity (0.15): word Jaccard score
    - Source Reliability Product (0.05): prior trust product
    """
    # 1. Spatial component
    if dist_km is not None:
        spatial_score = math.exp(-dist_km / 1.0)  # half-life ~700m, decay scale 1.0 km
    else:
        spatial_score = 0.50  # unknown spatial penalty
        
    # 2. Temporal component
    temporal_score = math.exp(-time_diff_min / 60.0)  # decay scale 60 min
    
    # 3. Category match component
    cat_weights = {
        "EXACT": 1.0,
        "CAUSAL": 0.75,
        "COMPATIBLE": 0.40,
        "INCOMPATIBLE": 0.0
    }
    cat_score = cat_weights.get(cat_match_type, 0.20)
    
    # 4. Text score
    text_score = min(1.0, max(0.0, text_sim))
    
    # 5. Composite heuristic weighted sum
    composite = (
        0.35 * spatial_score +
        0.25 * temporal_score +
        0.20 * cat_score +
        0.15 * text_score +
        0.05 * rel_product
    )
    return round(min(1.0, max(0.0, composite)), 4)
