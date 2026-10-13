"""
UrbanPulse Flood Intelligence & Infrastructure Dependency Rules (Phase 6)
===========================================================================
Defines configurable thresholds, dependency taxonomies, physical propagation constraints,
and audit schemas for causal and cascading urban flood analysis.

Core Safety Invariants:
1. Asset connectivity (DRAINS_TO, UPSTREAM_OF, POWERED_BY, SERVES_ROAD) must be verified
   before propagating dependencies.
2. Temporal sequences must be physically plausible (cause precedes effect within realistic propagation delay).
3. Do NOT infer drain blockage from high water level alone.
4. Do NOT infer power failure caused pump failure unless supply dependency is verified AND supporting evidence exists.
5. Inferred relationships default to POSSIBLE_DEPENDENCY unless causation is independently proven.
6. Severity is assigned strictly separately from dependency confidence.
7. Missing evidence results in UNCERTAIN_INVESTIGATING status rather than confirmed claims.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any


class DependencyType(str, Enum):
    RAINFALL_TO_WATER_RISE = "RAINFALL_TO_WATER_RISE"
    DRAIN_BLOCKAGE_TO_ACCUMULATION = "DRAIN_BLOCKAGE_TO_ACCUMULATION"
    PUMP_FAILURE_TO_DRAINAGE_IMPAIRMENT = "PUMP_FAILURE_TO_DRAINAGE_IMPAIRMENT"
    WATER_LEVEL_TO_ROAD_WATERLOGGING = "WATER_LEVEL_TO_ROAD_WATERLOGGING"
    ROAD_WATERLOGGING_TO_TRAFFIC_SLOWDOWN = "ROAD_WATERLOGGING_TO_TRAFFIC_SLOWDOWN"
    UNDERPASS_INUNDATION_TO_TRAVEL_DISRUPTION = "UNDERPASS_INUNDATION_TO_TRAVEL_DISRUPTION"
    POWER_FAILURE_TO_PUMP_FAILURE = "POWER_FAILURE_TO_PUMP_FAILURE"
    CASCADE_MULTI_HOP = "CASCADE_MULTI_HOP"


class DependencyStatus(str, Enum):
    CONFIRMED_DEPENDENCY = "CONFIRMED_DEPENDENCY"
    POSSIBLE_DEPENDENCY = "POSSIBLE_DEPENDENCY"
    UNCERTAIN_INVESTIGATING = "UNCERTAIN_INVESTIGATING"
    REFUTED_CONTRADICTED = "REFUTED_CONTRADICTED"
    INSUFFICIENT_CONNECTIVITY = "INSUFFICIENT_CONNECTIVITY"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class FloodRuleConfig:
    """Configurable thresholds for physical propagation and sensor validation."""
    # Rainfall thresholds
    heavy_rainfall_15min_mm: float = 5.0
    heavy_rainfall_1hr_mm: float = 20.0
    torrential_rainfall_1hr_mm: float = 50.0

    # Water level thresholds
    water_level_rise_rate_cm: float = 2.0
    water_level_surcharge_cm: float = 30.0
    water_level_critical_cm: float = 50.0
    underpass_closure_depth_cm: float = 35.0

    # Drain blockage thresholds
    min_blockage_probability: float = 0.40
    reduced_flow_capacity_pct: float = 40.0

    # Traffic disruption thresholds
    traffic_speed_drop_pct: float = 40.0
    traffic_severe_delay_min: float = 20.0

    # Temporal propagation window (minutes)
    rainfall_to_water_rise_max_delay_min: float = 60.0
    blockage_to_overflow_max_delay_min: float = 90.0
    power_to_pump_failure_max_delay_min: float = 30.0
    pump_failure_to_flooding_max_delay_min: float = 120.0
    water_to_road_flood_max_delay_min: float = 45.0
    road_flood_to_traffic_delay_min: float = 45.0

    # Spatial tolerance for unlinked but co-located assets (km)
    colocation_tolerance_km: float = 0.50

    # Sensor quality penalty multipliers
    quality_penalty_suspect: float = 0.35
    quality_penalty_imputed: float = 0.15
    health_penalty_degraded: float = 0.25
    health_penalty_offline: float = 0.50


@dataclass
class DependencyEvidenceItem:
    source_id: str
    source_type: str  # SENSOR, OBSERVATION, ASSET, DEPENDENCY_RECORD
    timestamp: Optional[str]
    is_supporting: bool
    summary: str
    data_quality: Optional[str] = "VALID"
    confidence_weight: float = 1.0


@dataclass
class DependencyAnalysisResult:
    dependency_type: DependencyType
    cause_asset_id: str
    effect_asset_id: str
    status: DependencyStatus
    confidence_score: float  # [0.0, 1.0]
    severity_level: SeverityLevel  # Independent from confidence
    rule_id: str
    explanation: str
    is_causation_proven: bool = False
    supporting_evidence: List[DependencyEvidenceItem] = field(default_factory=list)
    contradictory_evidence: List[DependencyEvidenceItem] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    alternative_hypotheses: List[str] = field(default_factory=list)
    temporal_delay_minutes: Optional[float] = None
    edge_attributes: Dict[str, Any] = field(default_factory=dict)
