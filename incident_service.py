"""
UrbanPulse Incident Integration & Normalization Service (Phase 7B.2)
===================================================================
Provides backend integration between the Intelligent Incident Agent and
route-monitoring services.

Architectural Invariants:
1. Reuses existing incident sources (in-memory live_events and canonical dataset).
   Never creates duplicate incident storage.
2. Normalizes heterogeneous incident structures into a canonical schema:
   - incident ID and timestamp
   - locality and validated latitude/longitude
   - predicted relationship and model confidence
   - alert level and urgency
   - traffic passability, speed-drop estimate, reported delay, affected route ID
   - recommended actions, nearby assets, evidence, status and provenance
3. Validates coordinates (lat [-90, 90], lon [-180, 180]), ISO 8601 timestamps,
   and numerical bounds (speed drop [0, 100], delay >= 0, confidences [0, 1]).
4. Strictly preserves original information; avoids inventing missing values.
5. Strict Decoupling Invariants:
   - ML classification confidence is separate from route-impact confidence.
   - An alert level of RED_CRITICAL or model confidence of 1.0 does NOT
     automatically mean a road is blocked.
   - A nearby physical asset is strictly an observed spatial proximity,
     NOT automatically assumed to be the root cause.
"""

import os
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------
class IncidentValidationError(Exception):
    """Base exception for incident validation failures."""
    pass


class CoordinateValidationError(IncidentValidationError):
    """Raised when latitude or longitude is invalid or out of bounds."""
    pass


class TimestampValidationError(IncidentValidationError):
    """Raised when timestamp is missing or not a valid ISO 8601 string."""
    pass


class NumericalFieldValidationError(IncidentValidationError):
    """Raised when a numerical field violates domain bounds or is non-finite."""
    pass


# ------------------------------------------------------------------------------
# Field Validation Helpers
# ------------------------------------------------------------------------------
def validate_coordinates(lat: Any, lon: Any, label: str = "Incident") -> Tuple[float, float]:
    """
    Validates geographic coordinates.
    Raises CoordinateValidationError if invalid, non-numeric, or non-finite.
    """
    if lat is None or lon is None:
        raise CoordinateValidationError(f"{label} coordinates cannot be None; got lat={lat!r}, lon={lon!r}")

    try:
        flat = float(lat)
        flon = float(lon)
    except (TypeError, ValueError) as e:
        raise CoordinateValidationError(f"{label} coordinates must be numeric; got lat={lat!r}, lon={lon!r}") from e

    if not math.isfinite(flat) or not math.isfinite(flon):
        raise CoordinateValidationError(f"{label} coordinates must be finite real numbers; got lat={flat}, lon={flon}")

    if not (-90.0 <= flat <= 90.0):
        raise CoordinateValidationError(f"{label} latitude must be between -90.0 and 90.0 degrees; got {flat}")

    if not (-180.0 <= flon <= 180.0):
        raise CoordinateValidationError(f"{label} longitude must be between -180.0 and 180.0 degrees; got {flon}")

    return flat, flon


def validate_timestamp(ts: Any, label: str = "Incident") -> str:
    """
    Validates that ts is a valid ISO 8601 timestamp string.
    Raises TimestampValidationError if missing or unparseable.
    """
    if not ts or not isinstance(ts, str):
        raise TimestampValidationError(f"{label} timestamp must be a non-empty string; got {ts!r}")

    cleaned = ts.strip()
    try:
        # Standard ISO 8601 parsing; replace trailing 'Z' with UTC offset
        datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as e:
        raise TimestampValidationError(f"{label} timestamp '{ts}' is not valid ISO 8601 format: {str(e)}") from e

    return cleaned


def validate_numerical_field(
    val: Any,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    field_name: str = "Numerical field"
) -> Optional[float]:
    """
    Validates optional numerical field bounds.
    Returns float or None.
    Raises NumericalFieldValidationError if value is out of bounds or non-finite.
    """
    if val is None:
        return None

    try:
        fval = float(val)
    except (TypeError, ValueError) as e:
        raise NumericalFieldValidationError(f"{field_name} must be numeric; got {val!r}") from e

    if not math.isfinite(fval):
        raise NumericalFieldValidationError(f"{field_name} must be a finite real number; got {fval}")

    if min_val is not None and fval < min_val:
        raise NumericalFieldValidationError(
            f"{field_name} cannot be less than {min_val}; got {fval}"
        )

    if max_val is not None and fval > max_val:
        raise NumericalFieldValidationError(
            f"{field_name} cannot be greater than {max_val}; got {fval}"
        )

    return fval


# ------------------------------------------------------------------------------
# Decoupled Route-Impact Confidence Calculator
# ------------------------------------------------------------------------------
def calculate_route_impact_confidence(
    passability: Optional[str],
    speed_drop_pct: Optional[float],
    delay_minutes: Optional[float]
) -> Tuple[Optional[float], bool]:
    """
    Calculates route impact confidence strictly decoupled from incident classification confidence.

    INVARIANTS:
    1. A critical alert or classification model_confidence == 1.0 does NOT mean route_impact_confidence is 1.0.
    2. A road is only marked blocked (is_road_blocked=True) if passability is explicitly
       'CLOSED' or 'IMPASSABLE'.
    3. Missing traffic telemetry yields None; values are never invented.
    """
    is_road_blocked = False

    # Explicit physical road closure/impassability
    if passability in ("CLOSED", "IMPASSABLE"):
        is_road_blocked = True
        return (0.95, is_road_blocked)

    has_traffic_evidence = (
        passability is not None or
        speed_drop_pct is not None or
        delay_minutes is not None
    )

    if not has_traffic_evidence:
        # Without traffic telemetry, route impact cannot be claimed with confidence
        return (None, False)

    score_components = []

    if passability == "RESTRICTED":
        score_components.append(0.70)
    elif passability == "PASSABLE":
        score_components.append(0.15)

    if speed_drop_pct is not None:
        if speed_drop_pct >= 70.0:
            score_components.append(0.85)
        elif speed_drop_pct >= 40.0:
            score_components.append(0.60)
        elif speed_drop_pct >= 20.0:
            score_components.append(0.35)
        else:
            score_components.append(0.10)

    if delay_minutes is not None:
        if delay_minutes >= 45.0:
            score_components.append(0.85)
        elif delay_minutes >= 20.0:
            score_components.append(0.60)
        elif delay_minutes >= 10.0:
            score_components.append(0.35)
        else:
            score_components.append(0.10)

    if not score_components:
        return (None, False)

    final_score = round(sum(score_components) / len(score_components), 3)
    return (final_score, is_road_blocked)


# ------------------------------------------------------------------------------
# Normalized Incident Schema
# ------------------------------------------------------------------------------
class NormalizedIncident(BaseModel):
    """Normalized incident representation integrating agent output with route-monitoring."""
    incident_id: str = Field(..., description="Unique canonical incident identifier", json_schema_extra={"example": "EVT-LIVE-00001"})
    timestamp: str = Field(..., description="ISO 8601 event observation timestamp", json_schema_extra={"example": "2026-10-13T04:20:00+05:30"})
    locality: Optional[str] = Field(None, description="Suburban locality or ward name", json_schema_extra={"example": "Andheri West"})
    latitude: float = Field(..., description="Latitude coordinate (-90 to 90)", json_schema_extra={"example": 19.11985})
    longitude: float = Field(..., description="Longitude coordinate (-180 to 180)", json_schema_extra={"example": 72.8465})

    # Incident classification confidence (ML / Causal)
    predicted_relationship: Optional[str] = Field(None, description="Predicted ML relationship label (e.g. NEW_PRIMARY_INCIDENT, SAME_INCIDENT)")
    model_confidence: Optional[float] = Field(None, description="ML classifier confidence score [0.0, 1.0]", json_schema_extra={"example": 0.95})

    # Alert & Urgency
    alert_level: Optional[str] = Field(None, description="Categorical alert level (e.g. GREEN_NORMAL, AMBER_WARNING, RED_CRITICAL, MODERATE)")
    urgency: Optional[str] = Field(None, description="Urgency priority (e.g. LOW, HIGH, IMMEDIATE)")

    # Traffic Passability & Impact (Strictly Decoupled)
    passability: Optional[str] = Field(None, description="Reported road passability (PASSABLE, RESTRICTED, IMPASSABLE, CLOSED)")
    speed_drop_pct: Optional[float] = Field(None, description="Speed drop percentage compared to baseline [0, 100]", json_schema_extra={"example": 65.0})
    delay_minutes: Optional[float] = Field(None, description="Estimated traffic delay in minutes (>= 0)", json_schema_extra={"example": 35.0})
    affected_route_id: Optional[str] = Field(None, description="Identifier of the corridor or route affected", json_schema_extra={"example": "ROUTE-004"})
    route_impact_confidence: Optional[float] = Field(None, description="Assessed route-impact confidence [0.0, 1.0] decoupled from model_confidence")
    is_road_blocked: bool = Field(default=False, description="Whether the roadway is explicitly blocked (only True if CLOSED or IMPASSABLE)")

    # Context, Evidence, Status & Provenance
    recommended_actions: List[str] = Field(default_factory=list, description="Actionable response recommendations")
    nearby_assets: List[Dict[str, Any]] = Field(default_factory=list, description="Nearby infrastructure assets (spatial proximity; NOT confirmed root cause)")
    root_cause_hypothesis: Optional[str] = Field(None, description="Hypothetical root cause explanation (explicitly a hypothesis)")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting telemetry and field evidence")
    status: str = Field(default="ACTIVE", description="Current incident lifecycle status (ACTIVE, INVESTIGATING, CANDIDATE, RESOLVED)")
    provenance: Optional[str] = Field(None, description="Data source or modality provenance")
    raw_payload: Optional[Dict[str, Any]] = Field(default=None, description="Preserved original un-normalized raw event dictionary")


# ------------------------------------------------------------------------------
# Normalization Logic
# ------------------------------------------------------------------------------
def normalize_incident(raw: Dict[str, Any]) -> NormalizedIncident:
    """
    Normalizes heterogeneous incident structures from the Intelligent Incident Agent:
    - Agent API response (ingest_telemetry_and_evaluate / query_location_response)
    - In-memory event dicts from live_events
    - Canonical records from incident_clusters.csv / observations.csv

    Validates coordinates, timestamps, and numerical fields.
    Preserves original information in raw_payload; avoids inventing missing values.
    """
    if not isinstance(raw, dict):
        raise IncidentValidationError(f"Incident payload must be a dictionary; got {type(raw)}")

    # 1. Incident ID
    incident_id = (
        raw.get("incident_id") or
        raw.get("observation_id") or
        raw.get("event_id") or
        raw.get("candidate_incident_id")
    )
    if not incident_id or not isinstance(incident_id, str):
        raise IncidentValidationError(f"Missing or invalid incident identifier: {incident_id!r}")

    # 2. Timestamp
    raw_ts = (
        raw.get("timestamp") or
        raw.get("observation_time") or
        raw.get("timestamp_iso") or
        raw.get("first_seen") or
        raw.get("event_time")
    )
    validated_ts = validate_timestamp(raw_ts, label=f"Incident {incident_id}")

    # 3. Locality
    locality = (
        raw.get("locality") or
        (raw.get("incident_location", {}).get("locality") if isinstance(raw.get("incident_location"), dict) else None) or
        raw.get("location_text")
    )

    # 4. Latitude and Longitude
    raw_lat = raw.get("latitude")
    raw_lon = raw.get("longitude")
    if raw_lat is None or raw_lon is None:
        # Check nested location coordinates
        inc_loc = raw.get("incident_location")
        if isinstance(inc_loc, dict):
            coords = inc_loc.get("coordinates")
            if isinstance(coords, dict):
                raw_lat = coords.get("latitude")
                raw_lon = coords.get("longitude")

    lat, lon = validate_coordinates(raw_lat, raw_lon, label=f"Incident {incident_id}")

    # 5. Classification Model Evaluation
    model_eval = raw.get("model_evaluation") if isinstance(raw.get("model_evaluation"), dict) else {}
    gen_resp = raw.get("generated_response") if isinstance(raw.get("generated_response"), dict) else {}
    ml_ctx = gen_resp.get("ml_inference_context") if isinstance(gen_resp.get("ml_inference_context"), dict) else {}

    predicted_rel = (
        model_eval.get("predicted_relationship") or
        ml_ctx.get("predicted_relationship") or
        raw.get("predicted_relationship")
    )

    raw_conf = (
        model_eval.get("confidence") or
        ml_ctx.get("model_confidence") or
        raw.get("model_confidence")
    )
    model_conf = validate_numerical_field(
        raw_conf,
        min_val=0.0,
        max_val=1.0,
        field_name=f"Incident {incident_id} model_confidence"
    )

    # 6. Alert Level & Urgency
    alert_level = (
        gen_resp.get("alert_level") or
        raw.get("alert_level") or
        raw.get("severity")
    )
    urgency = gen_resp.get("urgency") or raw.get("urgency")

    # 7. Traffic Impact & Telemetry
    traffic_dict = raw.get("traffic") if isinstance(raw.get("traffic"), dict) else {}
    traffic_impact = gen_resp.get("traffic_impact") if isinstance(gen_resp.get("traffic_impact"), dict) else {}

    passability = (
        traffic_dict.get("road_passability") or
        traffic_impact.get("passability") or
        raw.get("road_passability") or
        raw.get("passability")
    )

    raw_speed_drop = (
        traffic_dict.get("speed_drop_pct") or
        traffic_impact.get("speed_drop_pct") or
        raw.get("speed_drop_pct")
    )
    speed_drop_pct = validate_numerical_field(
        raw_speed_drop,
        min_val=0.0,
        max_val=100.0,
        field_name=f"Incident {incident_id} speed_drop_pct"
    )

    raw_delay = (
        traffic_dict.get("traffic_delay_minutes") or
        traffic_impact.get("delay_minutes") or
        raw.get("traffic_delay_minutes") or
        raw.get("delay_minutes")
    )
    delay_minutes = validate_numerical_field(
        raw_delay,
        min_val=0.0,
        max_val=None,
        field_name=f"Incident {incident_id} delay_minutes"
    )

    affected_route_id = (
        traffic_dict.get("affected_route_id") or
        traffic_impact.get("affected_route_id") or
        raw.get("affected_route_id")
    )

    # Calculate decoupled route impact confidence
    route_impact_conf, is_blocked = calculate_route_impact_confidence(
        passability=passability,
        speed_drop_pct=speed_drop_pct,
        delay_minutes=delay_minutes
    )

    # 8. Recommended Actions
    actions = gen_resp.get("recommended_actions") or raw.get("recommended_actions") or []
    if not isinstance(actions, list):
        actions = [str(actions)]

    # 9. Nearby Assets & Root Cause Hypothesis (Decoupled)
    nearby_assets = (
        raw.get("nearby_assets") or
        (raw.get("incident_location", {}).get("nearby_critical_assets") if isinstance(raw.get("incident_location"), dict) else None) or
        []
    )
    if not isinstance(nearby_assets, list):
        nearby_assets = []

    root_cause_hyp = (
        gen_resp.get("root_cause_hypothesis") or
        raw.get("root_cause_hypothesis")
    )

    # 10. Evidence Dictionary
    evidence = {}
    for key in ("water_level_cm", "rainfall_rate_mm", "pump_status", "power_status", "text", "event_type", "modality"):
        if key in raw and raw[key] is not None:
            evidence[key] = raw[key]

    # 11. Status & Provenance
    status = raw.get("status") or raw.get("cluster_status") or "ACTIVE"
    provenance = (
        raw.get("provenance") or
        raw.get("source_modality") or
        raw.get("modality") or
        raw.get("source_type")
    )

    return NormalizedIncident(
        incident_id=incident_id,
        timestamp=validated_ts,
        locality=locality,
        latitude=lat,
        longitude=lon,
        predicted_relationship=predicted_rel,
        model_confidence=model_conf,
        alert_level=alert_level,
        urgency=urgency,
        passability=passability,
        speed_drop_pct=speed_drop_pct,
        delay_minutes=delay_minutes,
        affected_route_id=affected_route_id,
        route_impact_confidence=route_impact_conf,
        is_road_blocked=is_blocked,
        recommended_actions=actions,
        nearby_assets=nearby_assets,
        root_cause_hypothesis=root_cause_hyp,
        evidence=evidence,
        status=status,
        provenance=provenance,
        raw_payload=raw
    )


# ------------------------------------------------------------------------------
# Active Incident Retrieval
# ------------------------------------------------------------------------------
def get_active_incidents(
    include_live: bool = True,
    include_dataset: bool = False,
    locality: Optional[str] = None,
    dataset_csv_path: Optional[str] = None
) -> List[NormalizedIncident]:
    """
    Retrieves and normalizes active incidents using existing incident sources.
    Avoids duplicating incident storage by directly reading live agent storage
    and/or canonical cluster records.

    Parameters:
    - include_live: Query in-memory live_events from api_service.
    - include_dataset: Query canonical active clusters from incident_clusters.csv.
    - locality: Optional filter by suburban locality name.
    - dataset_csv_path: Optional explicit path to incident_clusters.csv.
    """
    normalized_list: List[NormalizedIncident] = []
    seen_ids = set()

    # 1. Retrieve from live in-memory storage (api_service.live_events)
    if include_live:
        try:
            import api_service
            live_events = getattr(api_service, "live_events", [])
            for evt in live_events:
                try:
                    norm = normalize_incident(evt)
                    if norm.incident_id not in seen_ids:
                        if locality is None or (norm.locality and norm.locality.lower() == locality.lower()):
                            normalized_list.append(norm)
                            seen_ids.add(norm.incident_id)
                except Exception:
                    continue
        except ImportError:
            pass

    # 2. Retrieve from canonical incident clusters if requested or live events empty
    if include_dataset:
        csv_path = dataset_csv_path
        if not csv_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(current_dir, "incident_clusters.csv")
            if os.path.exists(candidate):
                csv_path = candidate

        if csv_path and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Active statuses in canonical dataset
                active_statuses = {"INVESTIGATING", "DETECTED", "CANDIDATE"}
                if "cluster_status" in df.columns:
                    active_df = df[df["cluster_status"].isin(active_statuses)]
                else:
                    active_df = df

                if locality and "locality" in active_df.columns:
                    active_df = active_df[active_df["locality"].str.lower() == locality.lower()]

                for _, row in active_df.iterrows():
                    row_dict = row.dropna().to_dict()
                    inc_id = str(row_dict.get("incident_id", ""))
                    if inc_id and inc_id not in seen_ids:
                        try:
                            norm = normalize_incident(row_dict)
                            normalized_list.append(norm)
                            seen_ids.add(norm.incident_id)
                        except Exception:
                            continue
            except Exception:
                pass

    return normalized_list
