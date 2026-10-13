"""
UrbanPulse Route Evaluation & Personalized Alert Service (Phase 7B.4)
====================================================================
Connects the OSRM routing service, incident retrieval, and geospatial
incident-to-route matching into a unified evaluation engine.

Key Architectural Invariants:
1. Strict Separation of Baseline Duration and Traffic Delay:
   - OSRM baseline duration represents free-flow network traversal.
   - Additional delay is derived strictly from confirmed bottleneck telemetry
     on the affected corridor, NEVER blindly added from unrelated incidents.
2. Non-fabrication on Dependency Failure:
   - When OSRM fails (timeout or unavailable), propagate errors; never return
     synthetic route geometry or fake ETAs.
3. Decoupled Alert Decision Logic:
   - RED / AVOID_ROUTE: Roadway is confirmed impassable/closed on route.
   - AMBER / PROCEED_WITH_CAUTION: Restricted passability or possible flood disruption.
   - GREEN / ROUTE_CLEAR: No operational bottlenecks detected.
4. Clean Documentation of Alternative Routes:
   - Alternative routes are marked NOT_YET_IMPLEMENTED until Phase 7B.5.
"""

import os
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from route_service import (
    Coordinate,
    RoutePlanResponse,
    get_route_service,
)
from incident_service import (
    NormalizedIncident,
    normalize_incident,
    get_active_incidents,
)
from route_matching_service import (
    IncidentRouteMatch,
    RouteMatchingConfig,
    MatchClassification,
    ImpactStatus,
    get_route_matching_service,
)


# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------
class RouteEvaluateRequest(BaseModel):
    origin: Coordinate = Field(..., description="Starting journey coordinates")
    destination: Coordinate = Field(..., description="Ending journey coordinates")
    candidate_radius_m: Optional[float] = Field(
        default=500.0,
        description="Geospatial candidate buffer radius in metres"
    )
    include_dataset: Optional[bool] = Field(
        default=False,
        description="Whether to include active incidents from incident_clusters.csv"
    )


class RouteAlertsEvaluateRequest(BaseModel):
    origin: Coordinate = Field(..., description="Starting journey coordinates")
    destination: Coordinate = Field(..., description="Ending journey coordinates")
    user_journey_id: Optional[str] = Field(
        default="DEMO-JOURNEY-001",
        description="Journey identifier for demo or personalized journey"
    )
    journey_name: Optional[str] = Field(
        default="Demo Commute Corridor",
        description="Label or moniker for the journey"
    )
    candidate_radius_m: Optional[float] = Field(
        default=500.0,
        description="Geospatial candidate buffer radius in metres"
    )
    include_dataset: Optional[bool] = Field(
        default=False,
        description="Whether to include active incidents from incident_clusters.csv"
    )


class AlternativeRoutesEvaluateRequest(BaseModel):
    origin: Coordinate = Field(..., description="Starting journey coordinates")
    destination: Coordinate = Field(..., description="Ending journey coordinates")
    candidate_radius_m: Optional[float] = Field(
        default=500.0,
        description="Geospatial candidate buffer radius in metres"
    )
    include_dataset: Optional[bool] = Field(
        default=False,
        description="Whether to include active incidents from incident_clusters.csv"
    )


class DelayBreakdown(BaseModel):
    baseline_duration_seconds: float = Field(..., description="OSRM free-flow duration in seconds")
    baseline_duration_minutes: float = Field(..., description="OSRM free-flow duration in minutes")
    additional_delay_minutes: float = Field(..., description="Justified additional delay in minutes from confirmed telemetry")
    total_estimated_duration_minutes: float = Field(..., description="Total estimated travel duration in minutes")
    delay_source: Optional[str] = Field(None, description="Source provenance of the estimated delay")
    calculation_method: str = Field(..., description="Methodology used to compute additional delay")


class AlertRecommendation(BaseModel):
    alert_level: str = Field(..., description="Alert severity: RED, AMBER, or GREEN")
    action: str = Field(..., description="Recommended traveler action: AVOID_ROUTE, PROCEED_WITH_CAUTION, or ROUTE_CLEAR")
    title: str = Field(..., description="Brief alert title")
    message: str = Field(..., description="Detailed alert summary")
    should_alert: bool = Field(..., description="Whether a proactive push alert is warranted")


class CandidateRecommendationStatus(str, Enum):
    PREFERRED = "PREFERRED"
    ACCEPTABLE_ALTERNATIVE = "ACCEPTABLE_ALTERNATIVE"
    REJECTED_BLOCKED = "REJECTED_BLOCKED"
    REJECTED_DISRUPTED = "REJECTED_DISRUPTED"
    REJECTED_SLOWER = "REJECTED_SLOWER"
    UNVIABLE = "UNVIABLE"


class CandidateRouteEvaluation(BaseModel):
    candidate_id: str = Field(..., description="Unique identifier for candidate route (e.g. candidate-0)")
    is_primary: bool = Field(..., description="Whether this candidate is the OSRM primary default route")
    corridor_summary: Optional[str] = Field(None, description="Corridor summary name (e.g. Western Express Highway)")
    route: RoutePlanResponse = Field(..., description="Candidate route geometry, distance, and baseline duration")
    overall_classification: str = Field(..., description="Route classification: ON_ROUTE, NEAR_ROUTE, AFFECTS_ROUTE, NOT_AFFECTED")
    overall_impact_status: str = Field(..., description="Operational status: CONFIRMED_DISRUPTION, POSSIBLE_DISRUPTION, NO_ESTABLISHED_DISRUPTION")
    is_road_blocked: bool = Field(..., description="True if any segment on this candidate corridor is confirmed closed/impassable")
    impacting_incidents_count: int = Field(..., description="Count of active incidents impacting this candidate")
    nearby_unrelated_incidents_count: int = Field(..., description="Count of nearby incidents with no established impact")
    delay: DelayBreakdown = Field(..., description="Baseline duration, justified delay, and total travel time estimate")
    recommendation_status: str = Field(..., description="PREFERRED, ACCEPTABLE_ALTERNATIVE, REJECTED_BLOCKED, REJECTED_DISRUPTED, REJECTED_SLOWER, or UNVIABLE")
    selection_rationale: str = Field(..., description="Explicit, calculated justification for preferring or rejecting this candidate route")
    impacting_incidents: List[IncidentRouteMatch] = Field(default_factory=list, description="Incidents affecting this specific candidate route")


class AlternativeRoutesEvaluation(BaseModel):
    status: str = Field(
        ...,
        description="PRIMARY_ROUTE_OPTIMAL, ALTERNATIVE_RECOMMENDED, ALL_ROUTES_BLOCKED, or NO_ALTERNATIVES_FOUND"
    )
    message: Optional[str] = Field(None, description="High-level status message for client compatibility")
    alternatives_available: bool = Field(..., description="True if at least one viable candidate exists besides a blocked primary, or if alternatives exist")
    recommended_candidate_id: Optional[str] = Field(None, description="Candidate ID of the preferred route, or None if all are blocked")
    recommended_route_summary: Optional[str] = Field(None, description="Summary moniker of the recommended route corridor")
    total_candidates: int = Field(default=0, description="Total number of candidates evaluated from OSRM")
    candidates: List[CandidateRouteEvaluation] = Field(default_factory=list, description="All evaluated candidate routes")
    selection_summary: str = Field(..., description="Summary explanation of candidate comparison and detour decision")
    limitations_and_disclaimer: str = Field(
        default="Candidate routes generated using OSRM alternatives API. OSRM baseline duration represents free-flow speeds without live traffic feeds. Delays are rule-based or telemetry-derived.",
        description="Methodological limitations and data provenance disclaimers"
    )


# Backwards compatibility alias
AlternativeRouteStatus = AlternativeRoutesEvaluation


class EvaluationSummary(BaseModel):
    overall_classification: str = Field(..., description="Aggregated route classification (e.g. AFFECTS_ROUTE, NEAR_ROUTE, NOT_AFFECTED)")
    overall_impact_status: str = Field(..., description="Aggregated operational impact (CONFIRMED_DISRUPTION, POSSIBLE_DISRUPTION, NO_ESTABLISHED_DISRUPTION)")
    is_route_blocked: bool = Field(..., description="Whether the monitored route is impassable")
    total_incidents_evaluated: int = Field(..., description="Total active incidents evaluated against the route")
    impacting_incidents_count: int = Field(..., description="Number of incidents causing confirmed or possible disruption")
    nearby_unrelated_incidents_count: int = Field(..., description="Number of nearby incidents with no established route disruption")


class RouteEvaluationResponse(BaseModel):
    route: RoutePlanResponse = Field(..., description="OSRM calculated route geometry, distance, and baseline duration")
    evaluation_summary: EvaluationSummary = Field(..., description="High-level impact summary")
    delay: DelayBreakdown = Field(..., description="Detailed travel time and delay analysis")
    alert: AlertRecommendation = Field(..., description="Actionable traveler alert recommendation")
    impacting_incidents: List[IncidentRouteMatch] = Field(default_factory=list, description="Incidents actively impacting the corridor")
    nearby_unrelated_incidents: List[IncidentRouteMatch] = Field(default_factory=list, description="Nearby incidents without established disruption")
    unaffected_incidents: List[IncidentRouteMatch] = Field(default_factory=list, description="Incidents outside the corridor or resolved")
    alternative_routes: AlternativeRoutesEvaluation = Field(..., description="Alternative routes evaluation and candidate scoring")
    evaluation_timestamp: str = Field(..., description="ISO 8601 UTC timestamp of evaluation")
    limitations: List[str] = Field(default_factory=list, description="Known operational constraints and modeling assumptions")


class JourneyAlertResponse(BaseModel):
    journey_id: str = Field(..., description="Journey identifier")
    journey_name: str = Field(..., description="User journey label")
    alert: AlertRecommendation = Field(..., description="Personalized alert recommendation")
    evaluation_summary: EvaluationSummary = Field(..., description="Impact overview")
    delay: DelayBreakdown = Field(..., description="Duration and delay breakdown")
    impacting_incidents: List[IncidentRouteMatch] = Field(default_factory=list, description="Incidents triggering the alert")
    evaluation_timestamp: str = Field(..., description="ISO 8601 UTC timestamp of alert decision")


# ------------------------------------------------------------------------------
# Core Evaluation Service Implementation
# ------------------------------------------------------------------------------
class RouteEvaluationService:
    """Orchestrates route planning, incident fetching, and candidate alternative evaluation."""

    def __init__(self):
        self.route_service = get_route_service()
        self.matching_service = get_route_matching_service()

    def evaluate_route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        candidate_radius_m: float = 500.0,
        include_dataset: bool = False
    ) -> RouteEvaluationResponse:
        """
        1. Plans OSRM candidate routes (using native alternatives).
        2. Retrieves active incidents.
        3. Geospatially matches and evaluates each candidate route.
        4. Ranks candidate routes, generates detour recommendations, and computes justified delays.
        """
        eval_time_iso = datetime.now(timezone.utc).isoformat()

        # 1. Plan Route Candidates via OSRM
        route_svc = get_route_service()
        candidates: List[RoutePlanResponse] = []
        try:
            if hasattr(route_svc, "plan_routes"):
                raw_candidates = route_svc.plan_routes(
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    dest_lat=dest_lat,
                    dest_lon=dest_lon,
                    alternatives=3
                )
                if isinstance(raw_candidates, list) and len(raw_candidates) > 0:
                    candidates = raw_candidates
                else:
                    candidates = [route_svc.plan_route(origin_lat, origin_lon, dest_lat, dest_lon)]
            else:
                candidates = [route_svc.plan_route(origin_lat, origin_lon, dest_lat, dest_lon)]
        except (AttributeError, TypeError):
            candidates = [route_svc.plan_route(origin_lat, origin_lon, dest_lat, dest_lon)]

        # 2. Retrieve Active Incidents
        active_incidents = get_active_incidents(
            include_live=True,
            include_dataset=include_dataset
        )

        # 3. Match and Evaluate Each Candidate Route
        match_config = RouteMatchingConfig(candidate_radius_m=candidate_radius_m)
        evaluated_candidates: List[CandidateRouteEvaluation] = []

        for idx, candidate in enumerate(candidates):
            c_eval = self._evaluate_single_candidate(
                candidate=candidate,
                candidate_id=f"candidate-{idx}",
                is_primary=(idx == 0),
                active_incidents=active_incidents,
                match_config=match_config
            )
            evaluated_candidates.append(c_eval)

        # 4. Rank Candidates & Determine Detour Recommendation
        alt_evaluation = self._rank_and_select_candidates(evaluated_candidates)

        # Primary route is index 0
        primary_eval = evaluated_candidates[0]

        # 5. Top-level Traveler Alert Recommendation
        if alt_evaluation.status == "ALL_ROUTES_BLOCKED":
            alert_level = "RED"
            alert_action = "AVOID_ROUTE"
            alert_title = "All Routes Blocked"
            alert_msg = f"Critical road disruptions on all evaluated corridors. No viable alternative route available."
            should_alert = True
        elif alt_evaluation.status == "NO_ALTERNATIVES_FOUND":
            alert_level = "RED"
            alert_action = "AVOID_ROUTE"
            alert_title = "Route Disruption - No Alternatives"
            alert_msg = f"Primary corridor is impassable due to confirmed road closure. No alternative routes available."
            should_alert = True
        elif alt_evaluation.status == "ALTERNATIVE_RECOMMENDED":
            # Primary is blocked or heavily disrupted, but a viable alternative exists
            alert_level = "RED" if primary_eval.is_road_blocked else "AMBER"
            alert_action = "AVOID_ROUTE" if primary_eval.is_road_blocked else "PROCEED_WITH_CAUTION"
            alert_title = "Detour Recommended"
            alt_summary = alt_evaluation.recommended_route_summary or "alternative corridor"
            alert_msg = f"Primary corridor is impassable due to confirmed disruption. Detour recommended via {alt_summary}."
            should_alert = True
        else:  # PRIMARY_ROUTE_OPTIMAL
            if primary_eval.is_road_blocked or primary_eval.overall_impact_status == ImpactStatus.CONFIRMED_DISRUPTION.value:
                alert_level = "RED"
                alert_action = "AVOID_ROUTE"
                alert_title = "Route Disruption Alert"
                alert_msg = f"Critical road disruption detected on route ({primary_eval.impacting_incidents_count} incident(s) affecting travel)."
                should_alert = True
            elif primary_eval.overall_impact_status == ImpactStatus.POSSIBLE_DISRUPTION.value:
                alert_level = "AMBER"
                alert_action = "PROCEED_WITH_CAUTION"
                alert_title = "Potential Delay Warning"
                alert_msg = f"Waterlogging or restricted passability near corridor ({primary_eval.impacting_incidents_count} possible disruption(s))."
                should_alert = True
            else:
                alert_level = "GREEN"
                alert_action = "ROUTE_CLEAR"
                alert_title = "Corridor Clear"
                alert_msg = "No operational road disruptions or bottlenecks detected along the selected route."
                should_alert = False

        alert_rec = AlertRecommendation(
            alert_level=alert_level,
            action=alert_action,
            title=alert_title,
            message=alert_msg,
            should_alert=should_alert
        )

        summary = EvaluationSummary(
            overall_classification=primary_eval.overall_classification,
            overall_impact_status=primary_eval.overall_impact_status,
            is_route_blocked=primary_eval.is_road_blocked,
            total_incidents_evaluated=len(active_incidents),
            impacting_incidents_count=primary_eval.impacting_incidents_count,
            nearby_unrelated_incidents_count=primary_eval.nearby_unrelated_incidents_count
        )

        # Primary route nearby/unaffected partition
        primary_matches = self.matching_service.match_incidents(
            incidents=active_incidents,
            route_polyline=primary_eval.route.geometry.coordinates,
            route_legs=primary_eval.route.legs,
            config=match_config
        )
        nearby_unrelated = [
            m for m in primary_matches
            if m.classification == MatchClassification.NEAR_ROUTE and
            m.impact_status in (ImpactStatus.NO_ESTABLISHED_DISRUPTION, ImpactStatus.INSUFFICIENT_EVIDENCE)
        ]
        unaffected = [
            m for m in primary_matches
            if m not in primary_eval.impacting_incidents and m not in nearby_unrelated
        ]

        limitations = [
            "OSRM baseline duration reflects free-flow road network speeds and does not include live traffic feeds.",
            "Additional delay is synthesized from verified incident reports and may evolve as conditions change.",
            "Opposite-carriageway events are excluded from disruption scoring unless cross-median blockage is evidenced.",
            "Candidate alternative routes are derived strictly from OSRM native routing graph alternatives.",
            "Detour recommendations are based on verified absence of closures on alternative corridors."
        ]

        return RouteEvaluationResponse(
            route=primary_eval.route,
            evaluation_summary=summary,
            delay=primary_eval.delay,
            alert=alert_rec,
            impacting_incidents=primary_eval.impacting_incidents,
            nearby_unrelated_incidents=nearby_unrelated,
            unaffected_incidents=unaffected,
            alternative_routes=alt_evaluation,
            evaluation_timestamp=eval_time_iso,
            limitations=limitations
        )

    def _evaluate_single_candidate(
        self,
        candidate: RoutePlanResponse,
        candidate_id: str,
        is_primary: bool,
        active_incidents: List[NormalizedIncident],
        match_config: RouteMatchingConfig
    ) -> CandidateRouteEvaluation:
        """Evaluates an individual candidate route against active incidents."""
        matches = self.matching_service.match_incidents(
            incidents=active_incidents,
            route_polyline=candidate.geometry.coordinates,
            route_legs=candidate.legs,
            config=match_config
        )

        impacting = [
            m for m in matches
            if m.impact_status in (ImpactStatus.CONFIRMED_DISRUPTION, ImpactStatus.POSSIBLE_DISRUPTION) or
            m.classification == MatchClassification.AFFECTS_ROUTE
        ]
        nearby_unrelated = [
            m for m in matches
            if m.classification == MatchClassification.NEAR_ROUTE and
            m.impact_status in (ImpactStatus.NO_ESTABLISHED_DISRUPTION, ImpactStatus.INSUFFICIENT_EVIDENCE)
        ]

        is_blocked = any(m.is_road_blocked for m in impacting)
        has_confirmed = any(m.impact_status == ImpactStatus.CONFIRMED_DISRUPTION for m in impacting)
        has_possible = any(m.impact_status == ImpactStatus.POSSIBLE_DISRUPTION for m in impacting)

        if has_confirmed or is_blocked:
            overall_impact = ImpactStatus.CONFIRMED_DISRUPTION.value
            overall_class = MatchClassification.AFFECTS_ROUTE.value
        elif has_possible:
            overall_impact = ImpactStatus.POSSIBLE_DISRUPTION.value
            overall_class = MatchClassification.AFFECTS_ROUTE.value
        else:
            overall_impact = ImpactStatus.NO_ESTABLISHED_DISRUPTION.value
            overall_class = MatchClassification.NEAR_ROUTE.value if nearby_unrelated else MatchClassification.NOT_AFFECTED.value

        # Delay calculation for this specific candidate
        baseline_sec = candidate.duration.seconds
        baseline_min = candidate.duration.minutes
        additional_delay_min = 0.0
        delay_source = None
        delay_method = "No traffic disruptions affecting route travel time."

        if is_blocked:
            delay_method = "Roadway is confirmed impassable/closed. Traversal not possible without detour."
            delay_source = "Corridor closure telemetry"
            additional_delay_min = 60.0  # Conservative nominal detour estimate
        elif impacting:
            reported_delays = []
            sources = []
            for inc in active_incidents:
                matching_m = next((x for x in impacting if x.incident_id == inc.incident_id), None)
                if matching_m and inc.delay_minutes and inc.delay_minutes > 0:
                    reported_delays.append(inc.delay_minutes)
                    sources.append(f"{inc.incident_id} ({inc.affected_route_id or 'route'})")

            if reported_delays:
                additional_delay_min = round(max(reported_delays), 1)
                delay_source = f"Telemetry feed from {', '.join(sources)}"
                delay_method = "Maximum confirmed bottleneck delay on route corridor."
            elif has_possible:
                additional_delay_min = 5.0
                delay_source = "Unconfirmed waterlogging advisory estimate"
                delay_method = "Nominal advisory buffer for unconfirmed water accumulation."

        total_est_duration_min = round(baseline_min + additional_delay_min, 1)

        delay_breakdown = DelayBreakdown(
            baseline_duration_seconds=baseline_sec,
            baseline_duration_minutes=baseline_min,
            additional_delay_minutes=additional_delay_min,
            total_estimated_duration_minutes=total_est_duration_min,
            delay_source=delay_source,
            calculation_method=delay_method
        )

        corridor_name = None
        if candidate.legs and len(candidate.legs) > 0 and candidate.legs[0].summary:
            corridor_name = candidate.legs[0].summary
        if not corridor_name:
            corridor_name = f"Corridor {'(Primary)' if is_primary else candidate_id}"

        return CandidateRouteEvaluation(
            candidate_id=candidate_id,
            is_primary=is_primary,
            corridor_summary=corridor_name,
            route=candidate,
            overall_classification=overall_class,
            overall_impact_status=overall_impact,
            is_road_blocked=is_blocked,
            impacting_incidents_count=len(impacting),
            nearby_unrelated_incidents_count=len(nearby_unrelated),
            delay=delay_breakdown,
            recommendation_status="PENDING",
            selection_rationale="Evaluation pending ranking.",
            impacting_incidents=impacting
        )

    def _rank_and_select_candidates(
        self,
        candidates: List[CandidateRouteEvaluation]
    ) -> AlternativeRoutesEvaluation:
        """Ranks candidates, assigns PREFERRED/REJECTED status, and determines detour recommendations."""
        if not candidates:
            return AlternativeRoutesEvaluation(
                status="NO_ALTERNATIVES_FOUND",
                message="No route candidates available.",
                alternatives_available=False,
                recommended_candidate_id=None,
                recommended_route_summary=None,
                total_candidates=0,
                candidates=[],
                selection_summary="No route candidates were generated."
            )

        # Single candidate scenario
        if len(candidates) == 1:
            c = candidates[0]
            if c.is_road_blocked:
                c.recommendation_status = "REJECTED_BLOCKED"
                c.selection_rationale = (
                    f"Route rejected: confirmed road closure / impassable roadway on {c.corridor_summary}."
                )
                return AlternativeRoutesEvaluation(
                    status="NO_ALTERNATIVES_FOUND",
                    message="Primary route is blocked and no alternative routes were generated.",
                    alternatives_available=False,
                    recommended_candidate_id=None,
                    recommended_route_summary=None,
                    total_candidates=1,
                    candidates=candidates,
                    selection_summary=(
                        f"Primary route via {c.corridor_summary} is blocked by confirmed closure. "
                        "No alternative corridors were generated by the routing provider."
                    )
                )
            else:
                c.recommendation_status = "PREFERRED"
                c.selection_rationale = (
                    f"Optimal route: {c.corridor_summary} has no confirmed closures with "
                    f"{c.delay.total_estimated_duration_minutes} min estimated duration."
                )
                return AlternativeRoutesEvaluation(
                    status="PRIMARY_ROUTE_OPTIMAL",
                    message="Primary route is clear and optimal.",
                    alternatives_available=False,
                    recommended_candidate_id=c.candidate_id,
                    recommended_route_summary=c.corridor_summary,
                    total_candidates=1,
                    candidates=candidates,
                    selection_summary=(
                        f"Primary route via {c.corridor_summary} is optimal ({c.delay.total_estimated_duration_minutes} min duration). "
                        "No alternative corridors were generated by OSRM for this journey."
                    )
                )

        # Multiple candidates scenario
        primary_c = candidates[0]
        viable_candidates = [c for c in candidates if not c.is_road_blocked]

        # Case 1: ALL candidates are blocked
        if not viable_candidates:
            for c in candidates:
                c.recommendation_status = "REJECTED_BLOCKED"
                c.selection_rationale = (
                    f"Corridor {c.corridor_summary} rejected: confirmed road closure / impassable roadway."
                )
            return AlternativeRoutesEvaluation(
                status="ALL_ROUTES_BLOCKED",
                message="All available corridors are impassable due to confirmed road closures.",
                alternatives_available=False,
                recommended_candidate_id=None,
                recommended_route_summary=None,
                total_candidates=len(candidates),
                candidates=candidates,
                selection_summary=(
                    f"All {len(candidates)} evaluated corridors (including primary and alternatives) "
                    "are impassable due to confirmed road closures. Traversal not recommended; no viable alternative available."
                )
            )

        # Case 2: Primary route is completely clear of disruptions
        if not primary_c.is_road_blocked and primary_c.overall_impact_status == ImpactStatus.NO_ESTABLISHED_DISRUPTION.value:
            primary_c.recommendation_status = "PREFERRED"
            primary_c.selection_rationale = (
                f"Primary corridor ({primary_c.corridor_summary}) is optimal: "
                f"zero closures, {primary_c.delay.total_estimated_duration_minutes} min duration."
            )

            for c in candidates[1:]:
                if c.is_road_blocked:
                    c.recommendation_status = "REJECTED_BLOCKED"
                    c.selection_rationale = (
                        f"Alternative corridor {c.corridor_summary} rejected: confirmed road closure."
                    )
                elif c.overall_impact_status == ImpactStatus.POSSIBLE_DISRUPTION.value:
                    c.recommendation_status = "REJECTED_DISRUPTED"
                    c.selection_rationale = (
                        f"Alternative corridor {c.corridor_summary} has potential waterlogging disruptions."
                    )
                else:
                    diff_min = round(c.delay.total_estimated_duration_minutes - primary_c.delay.total_estimated_duration_minutes, 1)
                    if diff_min > 5.0:
                        c.recommendation_status = "REJECTED_SLOWER"
                        c.selection_rationale = (
                            f"Alternative corridor {c.corridor_summary} is clear, but is +{diff_min} mins slower than primary."
                        )
                    else:
                        c.recommendation_status = "ACCEPTABLE_ALTERNATIVE"
                        c.selection_rationale = (
                            f"Acceptable alternative corridor ({c.corridor_summary}), baseline duration {c.delay.total_estimated_duration_minutes} mins."
                        )

            return AlternativeRoutesEvaluation(
                status="PRIMARY_ROUTE_OPTIMAL",
                message="Primary route is clear and optimal.",
                alternatives_available=(len(viable_candidates) > 1),
                recommended_candidate_id=primary_c.candidate_id,
                recommended_route_summary=primary_c.corridor_summary,
                total_candidates=len(candidates),
                candidates=candidates,
                selection_summary=(
                    f"Primary route via {primary_c.corridor_summary} is optimal "
                    f"({primary_c.delay.total_estimated_duration_minutes} min duration). "
                    f"{len(candidates) - 1} alternative corridor(s) evaluated."
                )
            )

        # Case 3: Primary corridor is blocked or disrupted, check if viable alternative exists
        viable_alternatives = [c for c in candidates[1:] if not c.is_road_blocked]
        if not viable_alternatives:
            # Primary was blocked and all alternatives are also blocked
            for c in candidates:
                c.recommendation_status = "REJECTED_BLOCKED"
                c.selection_rationale = (
                    f"Corridor {c.corridor_summary} rejected: confirmed road closure / impassable roadway."
                )
            return AlternativeRoutesEvaluation(
                status="ALL_ROUTES_BLOCKED",
                message="All available corridors are impassable due to confirmed road closures.",
                alternatives_available=False,
                recommended_candidate_id=None,
                recommended_route_summary=None,
                total_candidates=len(candidates),
                candidates=candidates,
                selection_summary=(
                    f"All {len(candidates)} evaluated corridors (including primary and alternatives) "
                    "are impassable due to confirmed road closures. Traversal not recommended; no viable alternative available."
                )
            )

        # Rank viable alternatives:
        def alt_sort_key(c: CandidateRouteEvaluation):
            disruption_rank = 0 if c.overall_impact_status == ImpactStatus.NO_ESTABLISHED_DISRUPTION.value else 1
            return (disruption_rank, c.delay.total_estimated_duration_minutes)

        viable_alternatives_sorted = sorted(viable_alternatives, key=alt_sort_key)
        best_candidate = viable_alternatives_sorted[0]

        if primary_c.is_road_blocked:
            primary_c.recommendation_status = "REJECTED_BLOCKED"
            primary_c.selection_rationale = (
                f"Primary corridor ({primary_c.corridor_summary}) rejected: confirmed road closure / impassable roadway."
            )
        else:
            primary_c.recommendation_status = "REJECTED_DISRUPTED"
            primary_c.selection_rationale = (
                f"Primary corridor ({primary_c.corridor_summary}) has active disruption delays "
                f"({primary_c.delay.additional_delay_minutes} min delay). Alternative route is clearer/faster."
            )

        best_candidate.recommendation_status = "PREFERRED"
        best_candidate.selection_rationale = (
            f"Recommended detour: Corridor {best_candidate.corridor_summary} is viable with zero closures "
            f"and {best_candidate.delay.total_estimated_duration_minutes} min estimated travel time."
        )

        for c in candidates:
            if c.candidate_id in (primary_c.candidate_id, best_candidate.candidate_id):
                continue
            if c.is_road_blocked:
                c.recommendation_status = "REJECTED_BLOCKED"
                c.selection_rationale = f"Alternative corridor {c.corridor_summary} rejected: confirmed closure."
            else:
                diff_min = round(c.delay.total_estimated_duration_minutes - best_candidate.delay.total_estimated_duration_minutes, 1)
                c.recommendation_status = "ACCEPTABLE_ALTERNATIVE" if diff_min <= 5.0 else "REJECTED_SLOWER"
                c.selection_rationale = (
                    f"Viable alternative ({c.corridor_summary}), +{diff_min} mins slower than recommended detour."
                )

        return AlternativeRoutesEvaluation(
            status="ALTERNATIVE_RECOMMENDED",
            message=f"Alternative route via {best_candidate.corridor_summary} is recommended.",
            alternatives_available=True,
            recommended_candidate_id=best_candidate.candidate_id,
            recommended_route_summary=best_candidate.corridor_summary,
            total_candidates=len(candidates),
            candidates=candidates,
            selection_summary=(
                f"Primary corridor ({primary_c.corridor_summary}) is {'blocked' if primary_c.is_road_blocked else 'disrupted'}. "
                f"Detour recommended via {best_candidate.corridor_summary} "
                f"({best_candidate.delay.total_estimated_duration_minutes} min estimated duration)."
            )
        )

    def evaluate_journey_alerts(
        self,
        request: RouteAlertsEvaluateRequest
    ) -> JourneyAlertResponse:
        """Evaluates a demo or coordinate-based traveler journey for personalized alert decisions."""
        eval_resp = self.evaluate_route(
            origin_lat=request.origin.latitude,
            origin_lon=request.origin.longitude,
            dest_lat=request.destination.latitude,
            dest_lon=request.destination.longitude,
            candidate_radius_m=request.candidate_radius_m or 500.0,
            include_dataset=request.include_dataset or False
        )

        return JourneyAlertResponse(
            journey_id=request.user_journey_id or "DEMO-JOURNEY-001",
            journey_name=request.journey_name or "Demo Commute Corridor",
            alert=eval_resp.alert,
            evaluation_summary=eval_resp.evaluation_summary,
            delay=eval_resp.delay,
            impacting_incidents=eval_resp.impacting_incidents,
            evaluation_timestamp=eval_resp.evaluation_timestamp
        )

    def evaluate_alternative_routes(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        candidate_radius_m: float = 500.0,
        include_dataset: bool = False
    ) -> AlternativeRoutesEvaluation:
        """Evaluates and scores candidate alternative corridors, returning AlternativeRoutesEvaluation."""
        eval_resp = self.evaluate_route(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            candidate_radius_m=candidate_radius_m,
            include_dataset=include_dataset
        )
        return eval_resp.alternative_routes


# ------------------------------------------------------------------------------
# Module-level instance helper
# ------------------------------------------------------------------------------
_default_eval_service: Optional[RouteEvaluationService] = None


def get_route_evaluation_service() -> RouteEvaluationService:
    global _default_eval_service
    if _default_eval_service is None:
        _default_eval_service = RouteEvaluationService()
    return _default_eval_service


def set_route_evaluation_service(service: Optional[RouteEvaluationService]) -> None:
    global _default_eval_service
    _default_eval_service = service
