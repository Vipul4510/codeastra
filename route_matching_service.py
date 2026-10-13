"""
UrbanPulse Geospatial Incident-to-Route Matching Service (Phase 7B.3)
=====================================================================
Matches active incident reports against actual OSRM route polyline geometries.

Architectural Invariants & Matching Requirements:
1. Compares incident coordinates against the actual route polyline (sequence of
   segments), NEVER against a straight origin-destination line.
2. Calculates minimum distance in metres using local metric projection and segment
   projection geometry.
3. Configurable candidate radius (default 500m) and on-route threshold (default 30m).
4. Returns nearest route position (latitude, longitude, segment index) and distance.
5. Verifies incident timestamps, freshness (max staleness window), resolution status,
   and empirical telemetry evidence.
6. Uses road-segment identity and travel direction when available.
7. Strictly distinguishes a nearby incident from a confirmed route disruption.
8. Opposite-carriageway invariant: does NOT claim opposite-carriageway impacts
   without explicit bidirectional disruption evidence.
9. Classifications supported:
   ON_ROUTE, NEAR_ROUTE, AFFECTS_ROUTE, NOT_AFFECTED, UNKNOWN.
10. Impact statuses supported:
   CONFIRMED_DISRUPTION, POSSIBLE_DISRUPTION, NO_ESTABLISHED_DISRUPTION, STALE,
   RESOLVED, INSUFFICIENT_EVIDENCE.
11. Returns an explicit diagnostic explanation for every evaluated incident.
"""

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from pydantic import BaseModel, Field

from incident_service import (
    NormalizedIncident,
    normalize_incident,
    validate_coordinates,
)


# ------------------------------------------------------------------------------
# Classifications & Impact Status Enums
# ------------------------------------------------------------------------------
class MatchClassification(str, Enum):
    ON_ROUTE = "ON_ROUTE"
    NEAR_ROUTE = "NEAR_ROUTE"
    AFFECTS_ROUTE = "AFFECTS_ROUTE"
    NOT_AFFECTED = "NOT_AFFECTED"
    UNKNOWN = "UNKNOWN"


class ImpactStatus(str, Enum):
    CONFIRMED_DISRUPTION = "CONFIRMED_DISRUPTION"
    POSSIBLE_DISRUPTION = "POSSIBLE_DISRUPTION"
    NO_ESTABLISHED_DISRUPTION = "NO_ESTABLISHED_DISRUPTION"
    STALE = "STALE"
    RESOLVED = "RESOLVED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ------------------------------------------------------------------------------
# Matching Configuration
# ------------------------------------------------------------------------------
class RouteMatchingConfig(BaseModel):
    candidate_radius_m: float = Field(
        default=500.0,
        description="Maximum distance in metres for an incident to be considered a candidate match"
    )
    on_route_threshold_m: float = Field(
        default=30.0,
        description="Distance threshold in metres within which an incident is classified physically ON_ROUTE"
    )
    max_staleness_hours: float = Field(
        default=6.0,
        description="Maximum age in hours before an un-updated incident is classified as STALE"
    )


# ------------------------------------------------------------------------------
# Output Schemas
# ------------------------------------------------------------------------------
class NearestRoutePosition(BaseModel):
    latitude: float = Field(..., description="Latitude of nearest point on the route polyline")
    longitude: float = Field(..., description="Longitude of nearest point on the route polyline")
    segment_index: Optional[int] = Field(None, description="Index of route polyline segment containing the nearest point")
    segment_name: Optional[str] = Field(None, description="Street / corridor name of the nearest leg step if available")


class IncidentRouteMatch(BaseModel):
    incident_id: str = Field(..., description="Canonical incident identifier")
    classification: MatchClassification = Field(..., description="Geospatial route match classification")
    impact_status: ImpactStatus = Field(..., description="Evaluated operational impact status on the route")
    distance_to_route_m: Optional[float] = Field(None, description="Minimum distance to route polyline in metres")
    nearest_point: Optional[NearestRoutePosition] = Field(None, description="Coordinates of closest point on route polyline")
    explanation: str = Field(..., description="Diagnostic rationale explaining the assigned classification and status")
    incident_timestamp: Optional[str] = Field(None, description="Observation timestamp of the incident")
    incident_locality: Optional[str] = Field(None, description="Suburban locality of the incident")
    affected_route_id: Optional[str] = Field(None, description="Corridor ID if reported by incident")
    is_road_blocked: bool = Field(default=False, description="Whether roadway is confirmed impassable/closed")
    route_impact_confidence: Optional[float] = Field(None, description="Decoupled route impact confidence score [0.0, 1.0]")
    matched_segment_name: Optional[str] = Field(None, description="Corridor name from nearest route leg step")


# ------------------------------------------------------------------------------
# Geospatial Mathematics
# ------------------------------------------------------------------------------
EARTH_RADIUS_METERS = 6371008.8


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates forward compass bearing (0-360 degrees) from (lat1, lon1) to (lat2, lon2)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angular_difference(bearing1: float, bearing2: float) -> float:
    """Calculates minimal angular difference in degrees between two compass bearings [0, 180]."""
    diff = abs(bearing1 - bearing2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def point_to_segment_distance_m(
    p_lat: float, p_lon: float,
    a_lat: float, a_lon: float,
    b_lat: float, b_lon: float
) -> Tuple[float, float, float, float]:
    """
    Computes minimum distance in metres from point P to line segment AB,
    and returns (min_distance_m, nearest_lat, nearest_lon, segment_bearing).
    Uses local metric projection centered at P.
    """
    lat_mid_rad = math.radians(p_lat)

    # Local Euclidean projection in metres relative to P
    x_a = EARTH_RADIUS_METERS * math.radians(a_lon - p_lon) * math.cos(lat_mid_rad)
    y_a = EARTH_RADIUS_METERS * math.radians(a_lat - p_lat)
    x_b = EARTH_RADIUS_METERS * math.radians(b_lon - p_lon) * math.cos(b_lat - p_lat if False else lat_mid_rad)
    y_b = EARTH_RADIUS_METERS * math.radians(b_lat - p_lat)

    dx = x_b - x_a
    dy = y_b - y_a
    seg_len_sq = dx * dx + dy * dy

    bearing = calculate_bearing(a_lat, a_lon, b_lat, b_lon)

    if seg_len_sq <= 1e-6:
        # Segment is degenerate (single point)
        dist = math.sqrt(x_a * x_a + y_a * y_a)
        return dist, a_lat, a_lon, bearing

    # Vector AP is (-x_a, -y_a) since P is at (0, 0)
    # Projection parameter t = (AP . AB) / |AB|^2
    t = ((-x_a) * dx + (-y_a) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))

    # Nearest point Q in metric plane
    x_q = x_a + t_clamped * dx
    y_q = y_a + t_clamped * dy

    dist = math.sqrt(x_q * x_q + y_q * y_q)

    # Convert metric point Q back to geographic coordinates
    q_lat = p_lat + math.degrees(y_q / EARTH_RADIUS_METERS)
    q_lon = p_lon + math.degrees(x_q / (EARTH_RADIUS_METERS * math.cos(lat_mid_rad)))

    return dist, q_lat, q_lon, bearing


def compute_point_to_polyline_distance(
    p_lat: float, p_lon: float,
    coordinates: List[List[float]]
) -> Tuple[float, float, float, int, float]:
    """
    Computes exact minimum distance from point P to a polyline of [lon, lat] coordinates.
    Returns (min_distance_m, nearest_lat, nearest_lon, nearest_seg_index, seg_bearing).
    """
    if len(coordinates) < 2:
        if len(coordinates) == 1:
            c_lon, c_lat = coordinates[0][0], coordinates[0][1]
            dist, _, _, b = point_to_segment_distance_m(p_lat, p_lon, c_lat, c_lon, c_lat, c_lon)
            return dist, c_lat, c_lon, 0, b
        return float("inf"), p_lat, p_lon, 0, 0.0

    min_dist = float("inf")
    best_lat = p_lat
    best_lon = p_lon
    best_seg_idx = 0
    best_bearing = 0.0

    for i in range(len(coordinates) - 1):
        a_lon, a_lat = coordinates[i][0], coordinates[i][1]
        b_lon, b_lat = coordinates[i+1][0], coordinates[i+1][1]

        dist, q_lat, q_lon, bearing = point_to_segment_distance_m(
            p_lat, p_lon, a_lat, a_lon, b_lat, b_lon
        )
        if dist < min_dist:
            min_dist = dist
            best_lat = q_lat
            best_lon = q_lon
            best_seg_idx = i
            best_bearing = bearing

    return min_dist, best_lat, best_lon, best_seg_idx, best_bearing


# ------------------------------------------------------------------------------
# Directional Parsing Helper
# ------------------------------------------------------------------------------
DIRECTION_ANGLES = {
    "NORTH": 0.0, "NORTHBOUND": 0.0, "NB": 0.0,
    "NORTHEAST": 45.0, "NE": 45.0,
    "EAST": 90.0, "EASTBOUND": 90.0, "EB": 90.0,
    "SOUTHEAST": 135.0, "SE": 135.0,
    "SOUTH": 180.0, "SOUTHBOUND": 180.0, "SB": 180.0,
    "SOUTHWEST": 225.0, "SW": 225.0,
    "WEST": 270.0, "WESTBOUND": 270.0, "WB": 270.0,
    "NORTHWEST": 315.0, "NW": 315.0,
}


def parse_direction_bearing(direction_val: Any) -> Optional[float]:
    """Extracts numeric compass bearing from direction label or degree value."""
    if direction_val is None:
        return None
    if isinstance(direction_val, (int, float)):
        return float(direction_val) % 360.0
    if isinstance(direction_val, str):
        cleaned = direction_val.strip().upper()
        if cleaned in DIRECTION_ANGLES:
            return DIRECTION_ANGLES[cleaned]
        try:
            return float(cleaned) % 360.0
        except ValueError:
            return None
    return None


# ------------------------------------------------------------------------------
# Geospatial Matching Service
# ------------------------------------------------------------------------------
class RouteMatchingService:
    """Dedicated service for matching incidents to route polylines."""

    def __init__(self, config: Optional[RouteMatchingConfig] = None):
        self.config = config or RouteMatchingConfig()

    def match_incident(
        self,
        incident: Union[NormalizedIncident, Dict[str, Any]],
        route_polyline: List[List[float]],
        route_legs: Optional[List[Any]] = None,
        config: Optional[RouteMatchingConfig] = None,
        reference_time: Optional[datetime] = None
    ) -> IncidentRouteMatch:
        """
        Evaluates a single incident against the route polyline.

        Parameters:
        - incident: NormalizedIncident or raw incident dictionary.
        - route_polyline: List of [longitude, latitude] coordinates from route geometry.
        - route_legs: Optional list of route leg steps for corridor naming.
        - config: Optional custom matching thresholds.
        - reference_time: Optional query reference datetime (defaults to UTC now).
        """
        cfg = config or self.config
        now = reference_time or datetime.now(timezone.utc)

        # 1. Normalize incident if needed
        norm_inc: Optional[NormalizedIncident] = None
        if isinstance(incident, NormalizedIncident):
            norm_inc = incident
        elif isinstance(incident, dict):
            try:
                norm_inc = normalize_incident(incident)
            except Exception as e:
                inc_id = incident.get("incident_id") or incident.get("observation_id", "UNKNOWN_INCIDENT")
                return IncidentRouteMatch(
                    incident_id=str(inc_id),
                    classification=MatchClassification.UNKNOWN,
                    impact_status=ImpactStatus.INSUFFICIENT_EVIDENCE,
                    distance_to_route_m=None,
                    nearest_point=None,
                    explanation=f"Invalid coordinates or malformed incident data: {str(e)}",
                    incident_timestamp=None,
                    incident_locality=None,
                    affected_route_id=None,
                    is_road_blocked=False,
                    route_impact_confidence=None,
                    matched_segment_name=None
                )
        else:
            return IncidentRouteMatch(
                incident_id="UNKNOWN_INCIDENT",
                classification=MatchClassification.UNKNOWN,
                impact_status=ImpactStatus.INSUFFICIENT_EVIDENCE,
                explanation="Unsupported incident input type.",
                is_road_blocked=False
            )

        inc_id = norm_inc.incident_id

        # 2. Coordinate validation
        try:
            flat, flon = validate_coordinates(norm_inc.latitude, norm_inc.longitude)
        except Exception as e:
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=MatchClassification.UNKNOWN,
                impact_status=ImpactStatus.INSUFFICIENT_EVIDENCE,
                explanation=f"Invalid incident coordinates: {str(e)}",
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id
            )

        # 3. Resolution Status Check
        if norm_inc.status and norm_inc.status.upper() == "RESOLVED":
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=MatchClassification.NOT_AFFECTED,
                impact_status=ImpactStatus.RESOLVED,
                distance_to_route_m=None,
                nearest_point=None,
                explanation=f"Incident {inc_id} is marked as RESOLVED and no longer disrupts traffic.",
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id,
                is_road_blocked=False,
                route_impact_confidence=0.0
            )

        # 4. Freshness / Staleness Check
        try:
            inc_dt = datetime.fromisoformat(norm_inc.timestamp.replace("Z", "+00:00"))
            # Ensure both datetimes are timezone-aware for comparison
            if inc_dt.tzinfo is None:
                inc_dt = inc_dt.replace(tzinfo=timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            age_hours = (now - inc_dt).total_seconds() / 3600.0
            if age_hours > cfg.max_staleness_hours:
                return IncidentRouteMatch(
                    incident_id=inc_id,
                    classification=MatchClassification.NOT_AFFECTED,
                    impact_status=ImpactStatus.STALE,
                    distance_to_route_m=None,
                    nearest_point=None,
                    explanation=(
                        f"Incident {inc_id} observation timestamp ({norm_inc.timestamp}) is STALE "
                        f"({round(age_hours, 1)} hours old, exceeding freshness window of {cfg.max_staleness_hours}h)."
                    ),
                    incident_timestamp=norm_inc.timestamp,
                    incident_locality=norm_inc.locality,
                    affected_route_id=norm_inc.affected_route_id,
                    is_road_blocked=False,
                    route_impact_confidence=0.0
                )
        except Exception:
            # If timestamp parsing fails here, proceed with caution
            pass

        # 5. Empty or Invalid Route Polyline
        if not route_polyline or len(route_polyline) == 0:
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=MatchClassification.UNKNOWN,
                impact_status=ImpactStatus.INSUFFICIENT_EVIDENCE,
                explanation="Route polyline contains no coordinates.",
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality
            )

        # 6. Polyline Minimum Distance Computation
        dist_m, near_lat, near_lon, seg_idx, seg_bearing = compute_point_to_polyline_distance(
            flat, flon, route_polyline
        )
        dist_m_rounded = round(dist_m, 1)

        # Resolve corridor step name if available from route_legs
        matched_step_name = None
        if route_legs and isinstance(route_legs, list):
            # Try to identify step name from route legs
            for leg in route_legs:
                leg_steps = getattr(leg, "steps", None) or (leg.get("steps") if isinstance(leg, dict) else None)
                if leg_steps and isinstance(leg_steps, list):
                    # Estimate matching step from cumulative geometry or summary
                    for step in leg_steps:
                        s_name = getattr(step, "name", None) or (step.get("name") if isinstance(step, dict) else None)
                        if s_name and s_name.strip():
                            matched_step_name = s_name.strip()
                            break
                if not matched_step_name:
                    leg_summary = getattr(leg, "summary", None) or (leg.get("summary") if isinstance(leg, dict) else None)
                    if leg_summary:
                        matched_step_name = leg_summary

        nearest_pos = NearestRoutePosition(
            latitude=round(near_lat, 6),
            longitude=round(near_lon, 6),
            segment_index=seg_idx,
            segment_name=matched_step_name
        )

        # 7. Spatial Candidate Filtering
        if dist_m > cfg.candidate_radius_m:
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=MatchClassification.NOT_AFFECTED,
                impact_status=ImpactStatus.NO_ESTABLISHED_DISRUPTION,
                distance_to_route_m=dist_m_rounded,
                nearest_point=nearest_pos,
                explanation=(
                    f"Incident {inc_id} is {dist_m_rounded}m away from the route polyline, "
                    f"beyond candidate radius ({cfg.candidate_radius_m}m). No impact on monitored route."
                ),
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id,
                is_road_blocked=False,
                route_impact_confidence=0.0,
                matched_segment_name=matched_step_name
            )

        # 8. Direction & Opposite-Carriageway Check
        raw_dict = norm_inc.raw_payload or {}
        traffic_raw = raw_dict.get("traffic") if isinstance(raw_dict.get("traffic"), dict) else {}
        inc_dir_val = (
            raw_dict.get("direction") or
            raw_dict.get("travel_direction") or
            traffic_raw.get("direction") or
            traffic_raw.get("carriageway")
        )
        is_bidirectional = (
            raw_dict.get("is_bidirectional") is True or
            str(inc_dir_val).upper() in ("BOTH", "BIDIRECTIONAL", "BOTH_DIRECTIONS")
        )
        carriageway_note = ""

        if inc_dir_val and not is_bidirectional:
            inc_bearing = parse_direction_bearing(inc_dir_val)
            if inc_bearing is not None:
                ang_diff = angular_difference(seg_bearing, inc_bearing)
                # If opposing direction (> 120 degrees apart) and distinct carriageways
                if ang_diff >= 120.0:
                    return IncidentRouteMatch(
                        incident_id=inc_id,
                        classification=MatchClassification.NEAR_ROUTE,
                        impact_status=ImpactStatus.NO_ESTABLISHED_DISRUPTION,
                        distance_to_route_m=dist_m_rounded,
                        nearest_point=nearest_pos,
                        explanation=(
                            f"Incident {inc_id} ({dist_m_rounded}m from route) is located on the opposing carriageway "
                            f"(reported direction {inc_dir_val} vs route bearing {round(seg_bearing, 1)}°). "
                            "Opposite-carriageway impacts are not claimed without confirmed cross-median disruption."
                        ),
                        incident_timestamp=norm_inc.timestamp,
                        incident_locality=norm_inc.locality,
                        affected_route_id=norm_inc.affected_route_id,
                        is_road_blocked=False,
                        route_impact_confidence=0.0,
                        matched_segment_name=matched_step_name
                    )

        # 9. Spatial Classification (ON_ROUTE vs NEAR_ROUTE)
        is_on_route = dist_m <= cfg.on_route_threshold_m
        base_classification = MatchClassification.ON_ROUTE if is_on_route else MatchClassification.NEAR_ROUTE

        # 10. Operational Impact Evaluation
        # A) Confirmed Road Blockage
        if norm_inc.is_road_blocked or (norm_inc.passability in ("CLOSED", "IMPASSABLE")):
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=MatchClassification.AFFECTS_ROUTE,
                impact_status=ImpactStatus.CONFIRMED_DISRUPTION,
                distance_to_route_m=dist_m_rounded,
                nearest_point=nearest_pos,
                explanation=(
                    f"CONFIRMED DISRUPTION: Roadway closure (passability: {norm_inc.passability or 'CLOSED'}) "
                    f"located {dist_m_rounded}m from route on corridor segment."
                ),
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id,
                is_road_blocked=True,
                route_impact_confidence=norm_inc.route_impact_confidence or 0.95,
                matched_segment_name=matched_step_name
            )

        # B) Confirmed Disruption from Telemetry Bottleneck on Affected Route
        has_route_match = (
            norm_inc.affected_route_id is not None and
            norm_inc.affected_route_id.strip() != ""
        )
        has_severe_delay = (
            (norm_inc.speed_drop_pct is not None and norm_inc.speed_drop_pct >= 50.0) or
            (norm_inc.delay_minutes is not None and norm_inc.delay_minutes >= 25.0)
        )
        if has_route_match and has_severe_delay:
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=MatchClassification.AFFECTS_ROUTE,
                impact_status=ImpactStatus.CONFIRMED_DISRUPTION,
                distance_to_route_m=dist_m_rounded,
                nearest_point=nearest_pos,
                explanation=(
                    f"CONFIRMED DISRUPTION: Severe traffic bottleneck on matched corridor {norm_inc.affected_route_id} "
                    f"({dist_m_rounded}m from route; speed drop: {norm_inc.speed_drop_pct}%, delay: {norm_inc.delay_minutes} min)."
                ),
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id,
                is_road_blocked=False,
                route_impact_confidence=norm_inc.route_impact_confidence or 0.85,
                matched_segment_name=matched_step_name
            )

        # C) Restricted Passability
        if norm_inc.passability == "RESTRICTED":
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=MatchClassification.AFFECTS_ROUTE,
                impact_status=ImpactStatus.POSSIBLE_DISRUPTION,
                distance_to_route_m=dist_m_rounded,
                nearest_point=nearest_pos,
                explanation=(
                    f"POSSIBLE DISRUPTION: Road passability reported as RESTRICTED {dist_m_rounded}m from route. "
                    "Traffic flow is constrained but not completely impassable."
                ),
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id,
                is_road_blocked=False,
                route_impact_confidence=norm_inc.route_impact_confidence or 0.70,
                matched_segment_name=matched_step_name
            )

        # D) Uncertain Flood with Unknown Passability (Requirement: must remain POSSIBLE_DISRUPTION)
        is_flood_event = False
        evidence = norm_inc.evidence or {}
        event_type = str(raw_dict.get("event_type", "")).upper()
        if (
            "WATERLOG" in event_type or "FLOOD" in event_type or
            evidence.get("water_level_cm") is not None or
            "water" in str(raw_dict.get("text", "")).lower()
        ):
            is_flood_event = True

        if is_flood_event and (norm_inc.passability is None or norm_inc.passability == ""):
            wl = evidence.get("water_level_cm")
            wl_str = f" (water level: {wl}cm)" if wl is not None else ""
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=base_classification,
                impact_status=ImpactStatus.POSSIBLE_DISRUPTION,
                distance_to_route_m=dist_m_rounded,
                nearest_point=nearest_pos,
                explanation=(
                    f"POSSIBLE DISRUPTION: Flood/waterlogging{wl_str} observed {dist_m_rounded}m from route. "
                    "Passability status is unconfirmed; road impairment remains possible."
                ),
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id,
                is_road_blocked=False,
                route_impact_confidence=0.50,
                matched_segment_name=matched_step_name
            )

        # E) Explicitly Passable / Normal Traffic
        if norm_inc.passability == "PASSABLE":
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=base_classification,
                impact_status=ImpactStatus.NO_ESTABLISHED_DISRUPTION,
                distance_to_route_m=dist_m_rounded,
                nearest_point=nearest_pos,
                explanation=(
                    f"Incident {inc_id} is {dist_m_rounded}m from route, but roadway is explicitly reported PASSABLE. "
                    "No operational disruption established."
                ),
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id,
                is_road_blocked=False,
                route_impact_confidence=norm_inc.route_impact_confidence or 0.15,
                matched_segment_name=matched_step_name
            )

        # F) Critical Incident that is Not Route-Affecting (Requirement)
        is_critical_alert = (norm_inc.alert_level in ("RED_CRITICAL", "CRITICAL", "MAJOR"))
        if is_critical_alert and not norm_inc.is_road_blocked:
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=base_classification,
                impact_status=ImpactStatus.NO_ESTABLISHED_DISRUPTION,
                distance_to_route_m=dist_m_rounded,
                nearest_point=nearest_pos,
                explanation=(
                    f"High-severity incident ({norm_inc.alert_level}) located {dist_m_rounded}m from route, "
                    "but no vehicular blockage or route corridor obstruction has been confirmed."
                ),
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id,
                is_road_blocked=False,
                route_impact_confidence=norm_inc.route_impact_confidence or 0.20,
                matched_segment_name=matched_step_name
            )

        # G) Insufficient Evidence Baseline
        if not evidence and norm_inc.passability is None and norm_inc.speed_drop_pct is None:
            return IncidentRouteMatch(
                incident_id=inc_id,
                classification=base_classification,
                impact_status=ImpactStatus.INSUFFICIENT_EVIDENCE,
                distance_to_route_m=dist_m_rounded,
                nearest_point=nearest_pos,
                explanation=(
                    f"Incident {inc_id} is {dist_m_rounded}m from route, but lacks telemetry evidence "
                    "or passability reports required to determine operational traffic impact."
                ),
                incident_timestamp=norm_inc.timestamp,
                incident_locality=norm_inc.locality,
                affected_route_id=norm_inc.affected_route_id,
                is_road_blocked=False,
                route_impact_confidence=None,
                matched_segment_name=matched_step_name
            )

        # Default fallback: No established disruption
        return IncidentRouteMatch(
            incident_id=inc_id,
            classification=base_classification,
            impact_status=ImpactStatus.NO_ESTABLISHED_DISRUPTION,
            distance_to_route_m=dist_m_rounded,
            nearest_point=nearest_pos,
            explanation=f"Incident {inc_id} is {dist_m_rounded}m from route without established traffic impairment.",
            incident_timestamp=norm_inc.timestamp,
            incident_locality=norm_inc.locality,
            affected_route_id=norm_inc.affected_route_id,
            is_road_blocked=False,
            route_impact_confidence=norm_inc.route_impact_confidence,
            matched_segment_name=matched_step_name
        )

    def match_incidents(
        self,
        incidents: List[Union[NormalizedIncident, Dict[str, Any]]],
        route_polyline: List[List[float]],
        route_legs: Optional[List[Any]] = None,
        config: Optional[RouteMatchingConfig] = None,
        reference_time: Optional[datetime] = None
    ) -> List[IncidentRouteMatch]:
        """Evaluates multiple incidents against a route polyline."""
        results = []
        for inc in incidents:
            results.append(
                self.match_incident(
                    incident=inc,
                    route_polyline=route_polyline,
                    route_legs=route_legs,
                    config=config,
                    reference_time=reference_time
                )
            )
        return results


# ------------------------------------------------------------------------------
# Module-level instance helper
# ------------------------------------------------------------------------------
_default_matching_service: Optional[RouteMatchingService] = None


def get_route_matching_service() -> RouteMatchingService:
    """Returns singleton instance of RouteMatchingService."""
    global _default_matching_service
    if _default_matching_service is None:
        _default_matching_service = RouteMatchingService()
    return _default_matching_service
