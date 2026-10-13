"""
UrbanPulse Flood Intelligence & Infrastructure Dependency Engine (Phase 6)
============================================================================
Performs rule-based causal and dependency reasoning across sensor telemetry,
flood observations, physical infrastructure assets, and dependency topology.

Enforces Core Architectural Invariants:
1. Verify asset connectivity before propagating dependencies.
2. Ensure temporal ordering is physically plausible (cause precedes effect within propagation limits).
3. Consider sensor health, data quality, water level trends, flow rate, and pump status.
4. Do NOT infer a drain blockage from high water level alone.
5. Do NOT infer that power failure caused pump failure unless supply dependency is verified AND evidence exists.
6. Represent inferred relationships as POSSIBLE_DEPENDENCY unless causation is independently established.
7. Record supporting evidence, contradictory evidence, missing information, and rules used.
8. Assign severity strictly separately from dependency confidence.
9. Test alternative explanations and contradictory observations.
10. Ensure missing evidence results in UNCERTAIN_INVESTIGATING status rather than confirmed claims.
"""

import os
import sys
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx

try:
    from .dependency_rules import (
        DependencyType,
        DependencyStatus,
        SeverityLevel,
        FloodRuleConfig,
        DependencyEvidenceItem,
        DependencyAnalysisResult
    )
    from .graph_validation import NodeType, EdgeType
except (ImportError, ValueError):
    from dependency_rules import (
        DependencyType,
        DependencyStatus,
        SeverityLevel,
        FloodRuleConfig,
        DependencyEvidenceItem,
        DependencyAnalysisResult
    )
    from graph_validation import NodeType, EdgeType


class FloodDependencyEngine:
    """
    Core causal and dependency reasoning engine for urban flood intelligence.
    """

    def __init__(
        self,
        config: Optional[FloodRuleConfig] = None,
        assets: Optional[List[Dict[str, Any]]] = None,
        dependencies: Optional[List[Dict[str, Any]]] = None
    ):
        self.config = config or FloodRuleConfig()
        self.asset_registry: Dict[str, Dict[str, Any]] = {}
        self.asset_graph = nx.DiGraph()

        if assets:
            self.load_assets(assets)
        if dependencies:
            self.load_dependencies(dependencies)

    def load_assets(self, assets: List[Dict[str, Any]]) -> None:
        """Loads physical asset records into asset graph and registry."""
        for ast in assets:
            a_id = ast.get("asset_id")
            if not a_id:
                continue
            self.asset_registry[a_id] = ast
            self.asset_graph.add_node(a_id, **ast)

    def load_dependencies(self, dependencies: List[Dict[str, Any]]) -> None:
        """Loads verified and inferred topological dependencies into asset graph."""
        for dep in dependencies:
            from_id = dep.get("from_asset_id")
            to_id = dep.get("to_asset_id")
            if not from_id or not to_id:
                continue
            # Ensure endpoints exist
            if not self.asset_graph.has_node(from_id):
                self.asset_graph.add_node(from_id, asset_id=from_id)
            if not self.asset_graph.has_node(to_id):
                self.asset_graph.add_node(to_id, asset_id=to_id)

            rel_type = dep.get("relationship_type", "POSSIBLE_DEPENDENCY")
            status = dep.get("relationship_status", "UNVERIFIED")
            conf = float(dep.get("confidence_score", 0.5))

            self.asset_graph.add_edge(
                from_id,
                to_id,
                relationship_type=rel_type,
                relationship_status=status,
                confidence=conf,
                evidence_id=dep.get("evidence_id")
            )

    @staticmethod
    def parse_iso_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
        """Parses ISO timestamp string to timezone-aware UTC datetime."""
        if not ts_str or not isinstance(ts_str, str):
            return None
        clean = ts_str.strip()
        try:
            if clean.endswith("Z"):
                clean = clean[:-1] + "+00:00"
            dt = datetime.fromisoformat(clean)
            if dt.tzinfo is None:
                # Default to IST (+05:30) if no timezone offset
                dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes great-circle distance between two GPS coordinates in kilometers."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return 6371.0088 * c

    def verify_asset_connectivity(
        self,
        from_asset_id: str,
        to_asset_id: str,
        expected_relationship: Optional[str] = None
    ) -> Tuple[bool, Optional[str], float]:
        """
        Requirement 1: Verify asset connectivity before propagating dependencies.
        Returns: (is_connected, path_or_relation_summary, topological_confidence)
        """
        if from_asset_id == to_asset_id:
            return True, "SAME_ASSET", 1.0

        # 1. Direct edge in asset topology
        if self.asset_graph.has_edge(from_asset_id, to_asset_id):
            edge_data = self.asset_graph[from_asset_id][to_asset_id]
            rel = edge_data.get("relationship_type", "CONNECTED")
            status = edge_data.get("relationship_status", "UNVERIFIED")
            conf = edge_data.get("confidence", 0.7)

            if expected_relationship and rel != expected_relationship:
                # Check reverse or symmetric adjacency
                pass
            else:
                return True, f"DIRECT_EDGE:{rel}:{status}", conf

        # 2. Multi-hop reachability in asset graph (up to 3 hops)
        try:
            if nx.has_path(self.asset_graph, from_asset_id, to_asset_id):
                path = nx.shortest_path(self.asset_graph, from_asset_id, to_asset_id)
                if len(path) <= 4:  # up to 3 hops
                    return True, f"MULTI_HOP_PATH:{'->'.join(path)}", 0.75
        except Exception:
            pass

        # 3. Catchment / Geographic Co-location Fallback
        ast1 = self.asset_registry.get(from_asset_id)
        ast2 = self.asset_registry.get(to_asset_id)
        if ast1 and ast2:
            catch1 = ast1.get("catchment_id")
            catch2 = ast2.get("catchment_id")
            if catch1 and catch2 and catch1 == catch2:
                return True, f"SHARED_CATCHMENT:{catch1}", 0.65

            lat1, lon1 = ast1.get("latitude"), ast1.get("longitude")
            lat2, lon2 = ast2.get("latitude"), ast2.get("longitude")
            if lat1 and lon1 and lat2 and lon2:
                dist = self.haversine_km(lat1, lon1, lat2, lon2)
                if dist <= self.config.colocation_tolerance_km:
                    return True, f"SPATIAL_PROXIMITY:{round(dist, 3)}km", 0.55

        return False, None, 0.0

    def check_temporal_plausibility(
        self,
        cause_time_iso: Optional[str],
        effect_time_iso: Optional[str],
        max_delay_minutes: float
    ) -> Tuple[bool, Optional[float], str]:
        """
        Requirement 2: Ensure timestamps are compatible and sequence is physically plausible.
        Cause must precede effect (or occur near-simultaneously within margin).
        """
        if not cause_time_iso or not effect_time_iso:
            return False, None, "MISSING_TIMESTAMP: Cannot determine temporal ordering."

        dt_cause = self.parse_iso_timestamp(cause_time_iso)
        dt_effect = self.parse_iso_timestamp(effect_time_iso)

        if not dt_cause or not dt_effect:
            return False, None, "UNPARSEABLE_TIMESTAMP: Timestamp format invalid."

        delta_sec = (dt_effect - dt_cause).total_seconds()
        delta_min = delta_sec / 60.0

        # Physical impossibility check: effect occurs substantially BEFORE cause
        if delta_min < -5.0:  # Allow 5-minute clock drift margin
            return (
                False,
                delta_min,
                f"PHYSICALLY_IMPLAUSIBLE: Proposed effect occurred {abs(round(delta_min, 1))} min before cause."
            )

        if delta_min > max_delay_minutes:
            return (
                False,
                delta_min,
                f"EXCEEDS_MAX_DELAY: Time gap of {round(delta_min, 1)} min exceeds threshold of {max_delay_minutes} min."
            )

        return True, max(0.0, delta_min), f"PLAUSIBLE: Propagated in {round(delta_min, 1)} min."

    def evaluate_sensor_quality_penalty(self, reading: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Requirement 3: Adjusts confidence based on sensor health and data quality.
        """
        penalty = 0.0
        warnings = []

        data_qual = reading.get("data_quality", "VALID")
        if data_qual == "SUSPECT":
            penalty += self.config.quality_penalty_suspect
            warnings.append("Data quality flagged SUSPECT.")
        elif data_qual == "IMPUTED":
            penalty += self.config.quality_penalty_imputed
            warnings.append("Data quality flagged IMPUTED.")

        sensor_health = reading.get("sensor_health", "HEALTHY")
        if sensor_health == "OFFLINE":
            penalty += self.config.health_penalty_offline
            warnings.append("Sensor health reported OFFLINE.")
        elif sensor_health == "DEGRADED":
            penalty += self.config.health_penalty_degraded
            warnings.append("Sensor health reported DEGRADED.")

        return min(0.85, penalty), warnings

    # -------------------------------------------------------------------------
    # Rule 1: Heavy Rainfall -> Rising Water Level
    # -------------------------------------------------------------------------
    def analyze_rainfall_to_water_rise(
        self,
        rain_sensor_reading: Dict[str, Any],
        water_level_reading: Dict[str, Any]
    ) -> DependencyAnalysisResult:
        """
        Rule 1: Evaluates heavy rainfall causing rising water level.
        """
        rule_id = "RULE_RAINFALL_TO_WATER_RISE"
        cause_id = rain_sensor_reading.get("sensor_id", "UNKNOWN_RAIN_GAUGE")
        effect_id = water_level_reading.get("sensor_id", "UNKNOWN_WATER_SENSOR")

        supporting: List[DependencyEvidenceItem] = []
        contradictory: List[DependencyEvidenceItem] = []
        missing: List[str] = []
        alt_hypotheses: List[str] = []

        # 1. Connectivity Check
        is_conn, conn_desc, conn_conf = self.verify_asset_connectivity(cause_id, effect_id)
        if not is_conn:
            return DependencyAnalysisResult(
                dependency_type=DependencyType.RAINFALL_TO_WATER_RISE,
                cause_asset_id=cause_id,
                effect_asset_id=effect_id,
                status=DependencyStatus.INSUFFICIENT_CONNECTIVITY,
                confidence_score=0.1,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=f"No catchment or spatial proximity link between rain gauge {cause_id} and sensor {effect_id}.",
                missing_evidence=["Verified hydrological link between rain gauge and water level sensor."]
            )

        # 2. Temporal Check
        t_cause = rain_sensor_reading.get("timestamp")
        t_effect = water_level_reading.get("timestamp")
        is_plausible, delay_min, temp_msg = self.check_temporal_plausibility(
            t_cause, t_effect, self.config.rainfall_to_water_rise_max_delay_min
        )

        if not is_plausible:
            return DependencyAnalysisResult(
                dependency_type=DependencyType.RAINFALL_TO_WATER_RISE,
                cause_asset_id=cause_id,
                effect_asset_id=effect_id,
                status=DependencyStatus.REFUTED_CONTRADICTED,
                confidence_score=0.1,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=temp_msg,
                contradictory_evidence=[DependencyEvidenceItem(
                    source_id=cause_id,
                    source_type="SENSOR",
                    timestamp=t_cause,
                    is_supporting=False,
                    summary=f"Incompatible timing: {temp_msg}"
                )]
            )

        # 3. Physical Signal Analysis
        rain_15m = float(rain_sensor_reading.get("rainfall_15min_mm", 0.0) or 0.0)
        rain_1h = float(rain_sensor_reading.get("rainfall_1hr_mm", 0.0) or 0.0)
        water_lvl = float(water_level_reading.get("water_level_cm", 0.0) or 0.0)
        water_change = float(water_level_reading.get("water_level_change_cm", 0.0) or 0.0)

        is_heavy_rain = (
            rain_15m >= self.config.heavy_rainfall_15min_mm
            or rain_1h >= self.config.heavy_rainfall_1hr_mm
        )
        is_water_rising = water_change >= self.config.water_level_rise_rate_cm or water_lvl >= self.config.water_level_surcharge_cm

        # Sensor health & quality penalty
        q_pen_rain, q_warn_rain = self.evaluate_sensor_quality_penalty(rain_sensor_reading)
        q_pen_water, q_warn_water = self.evaluate_sensor_quality_penalty(water_level_reading)
        total_pen = (q_pen_rain + q_pen_water) / 2.0

        if is_heavy_rain and is_water_rising:
            supporting.append(DependencyEvidenceItem(
                source_id=cause_id,
                source_type="SENSOR",
                timestamp=t_cause,
                is_supporting=True,
                summary=f"Rainfall intensity {rain_15m}mm/15m ({rain_1h}mm/1h) exceeds heavy rain threshold."
            ))
            supporting.append(DependencyEvidenceItem(
                source_id=effect_id,
                source_type="SENSOR",
                timestamp=t_effect,
                is_supporting=True,
                summary=f"Water level reached {water_lvl}cm with rise rate of +{water_change}cm."
            ))

            raw_conf = min(0.95, conn_conf * 0.90)
            final_conf = max(0.25, raw_conf - total_pen)

            sev = SeverityLevel.LOW
            if water_lvl >= self.config.water_level_critical_cm:
                sev = SeverityLevel.CRITICAL
            elif water_lvl >= self.config.water_level_surcharge_cm:
                sev = SeverityLevel.HIGH
            elif water_change > 5.0:
                sev = SeverityLevel.MODERATE

            return DependencyAnalysisResult(
                dependency_type=DependencyType.RAINFALL_TO_WATER_RISE,
                cause_asset_id=cause_id,
                effect_asset_id=effect_id,
                status=DependencyStatus.POSSIBLE_DEPENDENCY,
                confidence_score=round(final_conf, 3),
                severity_level=sev,
                rule_id=rule_id,
                explanation=f"Heavy localized precipitation ({rain_1h}mm/hr) propagated into +{water_change}cm water level rise within {round(delay_min or 0, 1)} min ({conn_desc}).",
                supporting_evidence=supporting,
                temporal_delay_minutes=delay_min,
                alternative_hypotheses=["Upstream catchment discharge", "Drain culvert capacity bottleneck"],
                edge_attributes={"relationship_type": EdgeType.POSSIBLE_DEPENDENCY.value, "confidence": round(final_conf, 3)}
            )

        elif not is_heavy_rain and is_water_rising:
            # Water rising without rain -> Alternative hypothesis required!
            alt_hypotheses.append("High tide marine backflow")
            alt_hypotheses.append("Upstream culvert release or pump backsurge")
            alt_hypotheses.append("Downstream drainage blockage causing localized backup")
            contradictory.append(DependencyEvidenceItem(
                source_id=cause_id,
                source_type="SENSOR",
                timestamp=t_cause,
                is_supporting=False,
                summary=f"Rainfall was negligible ({rain_15m}mm), contradicting pluvial rainfall flood hypothesis."
            ))

            return DependencyAnalysisResult(
                dependency_type=DependencyType.RAINFALL_TO_WATER_RISE,
                cause_asset_id=cause_id,
                effect_asset_id=effect_id,
                status=DependencyStatus.REFUTED_CONTRADICTED,
                confidence_score=0.15,
                severity_level=SeverityLevel.MODERATE,
                rule_id=rule_id,
                explanation="Water level rose without significant rainfall. Pluvial causation rejected; investigate alternative hypotheses.",
                contradictory_evidence=contradictory,
                alternative_hypotheses=alt_hypotheses
            )

        else:
            missing.append("Rainfall or water rise signals below threshold.")
            return DependencyAnalysisResult(
                dependency_type=DependencyType.RAINFALL_TO_WATER_RISE,
                cause_asset_id=cause_id,
                effect_asset_id=effect_id,
                status=DependencyStatus.UNCERTAIN_INVESTIGATING,
                confidence_score=0.35,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation="Insufficient rainfall or water level change to establish dependency.",
                missing_evidence=missing
            )

    # -------------------------------------------------------------------------
    # Rule 2: Drain Blockage -> Reduced Flow / Water Accumulation
    # -------------------------------------------------------------------------
    def analyze_drain_blockage_to_accumulation(
        self,
        drain_reading: Dict[str, Any],
        downstream_or_sump_reading: Dict[str, Any],
        blockage_observation: Optional[Dict[str, Any]] = None
    ) -> DependencyAnalysisResult:
        """
        Rule 2 & Invariant 4: Evaluates drain blockage causing reduced flow or accumulation.
        CRITICAL: Never infer drain blockage from high water level alone!
        """
        rule_id = "RULE_DRAIN_BLOCKAGE_TO_ACCUMULATION"
        drain_id = drain_reading.get("sensor_id") or drain_reading.get("asset_id", "DRAIN_UNKNOWN")
        sump_id = downstream_or_sump_reading.get("sensor_id") or downstream_or_sump_reading.get("asset_id", "SUMP_UNKNOWN")

        supporting: List[DependencyEvidenceItem] = []
        contradictory: List[DependencyEvidenceItem] = []
        missing: List[str] = []
        alt_hypotheses: List[str] = []

        # 1. Connectivity Check
        is_conn, conn_desc, conn_conf = self.verify_asset_connectivity(drain_id, sump_id)
        if not is_conn:
            return DependencyAnalysisResult(
                dependency_type=DependencyType.DRAIN_BLOCKAGE_TO_ACCUMULATION,
                cause_asset_id=drain_id,
                effect_asset_id=sump_id,
                status=DependencyStatus.INSUFFICIENT_CONNECTIVITY,
                confidence_score=0.1,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=f"No drainage connectivity link verified between {drain_id} and {sump_id}.",
                missing_evidence=["Drainage network connectivity record."]
            )

        # 2. Blockage Indicator Check (INVARIANT 4)
        blockage_prob = float(drain_reading.get("blockage_probability", 0.0) or 0.0)
        flow_pct = float(drain_reading.get("drainage_capacity_pct", 100.0) or 100.0)
        water_lvl = float(downstream_or_sump_reading.get("water_level_cm", 0.0) or 0.0)

        has_explicit_blockage = (
            blockage_prob >= self.config.min_blockage_probability
            or (blockage_observation and blockage_observation.get("blockage_status") in ("SUSPECTED", "CONFIRMED"))
        )

        # Invariant 4 safeguard: High water level alone CANNOT infer blockage!
        if not has_explicit_blockage and water_lvl >= self.config.water_level_surcharge_cm:
            alt_hypotheses.append("Rainfall volume simply exceeded nominal hydraulic design capacity.")
            alt_hypotheses.append("Pump station clearance rate is below inflow rate.")
            alt_hypotheses.append("High downstream tidal head preventing gravity discharge.")

            return DependencyAnalysisResult(
                dependency_type=DependencyType.DRAIN_BLOCKAGE_TO_ACCUMULATION,
                cause_asset_id=drain_id,
                effect_asset_id=sump_id,
                status=DependencyStatus.UNCERTAIN_INVESTIGATING,
                confidence_score=0.20,
                severity_level=SeverityLevel.MODERATE,
                rule_id=rule_id,
                explanation="INVARIANT ENFORCED: High water level alone is insufficient to infer a drain blockage. No explicit blockage sensor telemetry or inspection record present.",
                missing_evidence=["Physical drain inspection record", "Acoustic / ultrasonic pipe blockage telemetry"],
                alternative_hypotheses=alt_hypotheses
            )

        if has_explicit_blockage:
            supporting.append(DependencyEvidenceItem(
                source_id=drain_id,
                source_type="SENSOR",
                timestamp=drain_reading.get("timestamp"),
                is_supporting=True,
                summary=f"Explicit blockage detected: probability {blockage_prob:.2f}, drainage capacity down to {flow_pct:.1f}%."
            ))
            if blockage_observation:
                supporting.append(DependencyEvidenceItem(
                    source_id=blockage_observation.get("flood_observation_id", "OBS-BLOCKAGE"),
                    source_type="OBSERVATION",
                    timestamp=blockage_observation.get("observed_time"),
                    is_supporting=True,
                    summary=f"Field observation confirmed blockage status: {blockage_observation.get('blockage_status')}."
                ))

            # Severity assessment (independent from confidence)
            sev = SeverityLevel.MODERATE
            if flow_pct <= 20.0 or water_lvl >= self.config.water_level_critical_cm:
                sev = SeverityLevel.CRITICAL
            elif flow_pct <= 40.0:
                sev = SeverityLevel.HIGH

            conf = min(0.92, conn_conf * (0.60 + (blockage_prob * 0.35)))

            return DependencyAnalysisResult(
                dependency_type=DependencyType.DRAIN_BLOCKAGE_TO_ACCUMULATION,
                cause_asset_id=drain_id,
                effect_asset_id=sump_id,
                status=DependencyStatus.POSSIBLE_DEPENDENCY,
                confidence_score=round(conf, 3),
                severity_level=sev,
                rule_id=rule_id,
                explanation=f"Verified blockage (prob {blockage_prob:.2f}) reduced drainage capacity to {flow_pct:.1f}%, driving water accumulation in {sump_id} ({conn_desc}).",
                supporting_evidence=supporting,
                edge_attributes={"relationship_type": EdgeType.POSSIBLE_DEPENDENCY.value, "confidence": round(conf, 3)}
            )

        # Neither blockage nor high water level
        return DependencyAnalysisResult(
            dependency_type=DependencyType.DRAIN_BLOCKAGE_TO_ACCUMULATION,
            cause_asset_id=drain_id,
            effect_asset_id=sump_id,
            status=DependencyStatus.UNCERTAIN_INVESTIGATING,
            confidence_score=0.15,
            severity_level=SeverityLevel.LOW,
            rule_id=rule_id,
            explanation="Drain operating within normal clearance parameters; no blockage indicators detected.",
            missing_evidence=["Blockage indicators absent."]
        )

    # -------------------------------------------------------------------------
    # Rule 3: Pump Failure -> Reduced Drainage Capability
    # -------------------------------------------------------------------------
    def analyze_pump_failure_to_drainage_impairment(
        self,
        pump_reading: Dict[str, Any],
        sump_reading: Dict[str, Any],
        pump_observation: Optional[Dict[str, Any]] = None
    ) -> DependencyAnalysisResult:
        """
        Rule 3: Evaluates pump failure causing sump/underpass inundation.
        """
        rule_id = "RULE_PUMP_FAILURE_TO_DRAINAGE_IMPAIRMENT"
        pump_id = pump_reading.get("sensor_id") or pump_reading.get("asset_id", "PUMP_UNKNOWN")
        sump_id = sump_reading.get("sensor_id") or sump_reading.get("asset_id", "SUMP_UNKNOWN")

        supporting: List[DependencyEvidenceItem] = []
        contradictory: List[DependencyEvidenceItem] = []
        missing: List[str] = []

        # 1. Connectivity Check
        is_conn, conn_desc, conn_conf = self.verify_asset_connectivity(pump_id, sump_id)
        if not is_conn:
            return DependencyAnalysisResult(
                dependency_type=DependencyType.PUMP_FAILURE_TO_DRAINAGE_IMPAIRMENT,
                cause_asset_id=pump_id,
                effect_asset_id=sump_id,
                status=DependencyStatus.INSUFFICIENT_CONNECTIVITY,
                confidence_score=0.1,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=f"Pump station {pump_id} is not connected to drainage catchment for {sump_id}.",
                missing_evidence=["Pump-to-sump dependency record."]
            )

        # 2. Pump Status Check
        p_status = pump_reading.get("pump_status", "UNKNOWN")
        is_failed = p_status in ("FAILED", "OFFLINE") or (
            pump_observation and pump_observation.get("event_type") == "PUMP_FAILURE"
        )

        water_lvl = float(sump_reading.get("water_level_cm", 0.0) or 0.0)

        # Temporal check
        t_pump = pump_reading.get("timestamp")
        t_sump = sump_reading.get("timestamp")
        is_plausible, delay_min, temp_msg = self.check_temporal_plausibility(
            t_pump, t_sump, self.config.pump_failure_to_flooding_max_delay_min
        )

        if is_failed and is_plausible:
            supporting.append(DependencyEvidenceItem(
                source_id=pump_id,
                source_type="SENSOR",
                timestamp=t_pump,
                is_supporting=True,
                summary=f"Pump telemetry reports status: {p_status}."
            ))
            supporting.append(DependencyEvidenceItem(
                source_id=sump_id,
                source_type="SENSOR",
                timestamp=t_sump,
                is_supporting=True,
                summary=f"Sump standing water level recorded at {water_lvl}cm."
            ))

            sev = SeverityLevel.MODERATE
            if water_lvl >= self.config.water_level_critical_cm:
                sev = SeverityLevel.CRITICAL
            elif water_lvl >= self.config.water_level_surcharge_cm:
                sev = SeverityLevel.HIGH

            conf = min(0.94, conn_conf * 0.90)

            return DependencyAnalysisResult(
                dependency_type=DependencyType.PUMP_FAILURE_TO_DRAINAGE_IMPAIRMENT,
                cause_asset_id=pump_id,
                effect_asset_id=sump_id,
                status=DependencyStatus.POSSIBLE_DEPENDENCY,
                confidence_score=round(conf, 3),
                severity_level=sev,
                rule_id=rule_id,
                explanation=f"Pump stoppage ({p_status}) directly impaired mechanical dewatering in {sump_id}, leading to {water_lvl}cm accumulation ({conn_desc}).",
                supporting_evidence=supporting,
                temporal_delay_minutes=delay_min,
                alternative_hypotheses=["Inflow volume surged beyond rated pumping capacity even if active"],
                edge_attributes={"relationship_type": EdgeType.POSSIBLE_DEPENDENCY.value, "confidence": round(conf, 3)}
            )

        elif not is_failed and water_lvl >= self.config.water_level_surcharge_cm:
            contradictory.append(DependencyEvidenceItem(
                source_id=pump_id,
                source_type="SENSOR",
                timestamp=t_pump,
                is_supporting=False,
                summary=f"Pump is actively RUNNING; accumulation caused by extreme inflow volume exceeding rated capacity."
            ))
            return DependencyAnalysisResult(
                dependency_type=DependencyType.PUMP_FAILURE_TO_DRAINAGE_IMPAIRMENT,
                cause_asset_id=pump_id,
                effect_asset_id=sump_id,
                status=DependencyStatus.REFUTED_CONTRADICTED,
                confidence_score=0.15,
                severity_level=SeverityLevel.MODERATE,
                rule_id=rule_id,
                explanation="Pump is operating normally. Inundation is due to hydrologic overload, not pump failure.",
                contradictory_evidence=contradictory,
                alternative_hypotheses=["Inflow exceeds design pump discharge capacity (extreme cloudburst)"]
            )

        else:
            missing.append("Pump telemetry status uncertain or water level normal.")
            return DependencyAnalysisResult(
                dependency_type=DependencyType.PUMP_FAILURE_TO_DRAINAGE_IMPAIRMENT,
                cause_asset_id=pump_id,
                effect_asset_id=sump_id,
                status=DependencyStatus.UNCERTAIN_INVESTIGATING,
                confidence_score=0.30,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation="Insufficient telemetry to confirm pump failure impact on drainage.",
                missing_evidence=missing
            )

    # -------------------------------------------------------------------------
    # Rule 4: Rising Water Level -> Road Waterlogging
    # -------------------------------------------------------------------------
    def analyze_water_level_to_road_waterlogging(
        self,
        water_sensor_reading: Dict[str, Any],
        road_observation: Dict[str, Any]
    ) -> DependencyAnalysisResult:
        """
        Rule 4: Evaluates culvert/drain water surcharge causing surface road waterlogging.
        """
        rule_id = "RULE_WATER_LEVEL_TO_ROAD_WATERLOGGING"
        sensor_id = water_sensor_reading.get("sensor_id", "UNKNOWN_SENSOR")
        road_id = road_observation.get("asset_id") or road_observation.get("flood_observation_id", "ROAD_UNKNOWN")

        supporting: List[DependencyEvidenceItem] = []
        contradictory: List[DependencyEvidenceItem] = []

        is_conn, conn_desc, conn_conf = self.verify_asset_connectivity(sensor_id, road_id)
        if not is_conn:
            return DependencyAnalysisResult(
                dependency_type=DependencyType.WATER_LEVEL_TO_ROAD_WATERLOGGING,
                cause_asset_id=sensor_id,
                effect_asset_id=road_id,
                status=DependencyStatus.INSUFFICIENT_CONNECTIVITY,
                confidence_score=0.1,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=f"Sensor {sensor_id} is not connected to road segment {road_id}.",
                missing_evidence=["Spatial or drainage connection between sensor and road segment."]
            )

        t_water = water_sensor_reading.get("timestamp")
        t_road = road_observation.get("observed_time") or road_observation.get("reported_time")
        is_plausible, delay_min, temp_msg = self.check_temporal_plausibility(
            t_water, t_road, self.config.water_to_road_flood_max_delay_min
        )

        water_lvl = float(water_sensor_reading.get("water_level_cm", 0.0) or 0.0)
        road_depth = float(road_observation.get("water_depth_cm", 0.0) or 0.0)

        if water_lvl >= self.config.water_level_surcharge_cm and is_plausible:
            supporting.append(DependencyEvidenceItem(
                source_id=sensor_id,
                source_type="SENSOR",
                timestamp=t_water,
                is_supporting=True,
                summary=f"Culvert water level reached {water_lvl}cm exceeding surcharge threshold."
            ))
            supporting.append(DependencyEvidenceItem(
                source_id=road_id,
                source_type="OBSERVATION",
                timestamp=t_road,
                is_supporting=True,
                summary=f"Road waterlogging reported with surface water depth of {road_depth}cm."
            ))

            sev = SeverityLevel.MODERATE
            if road_depth >= 40.0:
                sev = SeverityLevel.CRITICAL
            elif road_depth >= 20.0:
                sev = SeverityLevel.HIGH

            conf = min(0.92, conn_conf * 0.90)

            return DependencyAnalysisResult(
                dependency_type=DependencyType.WATER_LEVEL_TO_ROAD_WATERLOGGING,
                cause_asset_id=sensor_id,
                effect_asset_id=road_id,
                status=DependencyStatus.POSSIBLE_DEPENDENCY,
                confidence_score=round(conf, 3),
                severity_level=sev,
                rule_id=rule_id,
                explanation=f"Subsurface drain surcharge ({water_lvl}cm) backed up onto surface carriage-way, producing {road_depth}cm waterlogging ({conn_desc}).",
                supporting_evidence=supporting,
                temporal_delay_minutes=delay_min,
                edge_attributes={"relationship_type": EdgeType.POSSIBLE_DEPENDENCY.value, "confidence": round(conf, 3)}
            )

        return DependencyAnalysisResult(
            dependency_type=DependencyType.WATER_LEVEL_TO_ROAD_WATERLOGGING,
            cause_asset_id=sensor_id,
            effect_asset_id=road_id,
            status=DependencyStatus.UNCERTAIN_INVESTIGATING,
            confidence_score=0.25,
            severity_level=SeverityLevel.LOW,
            rule_id=rule_id,
            explanation="Drain water level is below surcharge threshold; surface waterlogging not propagated from drain.",
            missing_evidence=["Water level surcharge evidence."]
        )

    # -------------------------------------------------------------------------
    # Rule 5: Road Waterlogging -> Traffic Slowdown
    # -------------------------------------------------------------------------
    def analyze_road_waterlogging_to_traffic_slowdown(
        self,
        road_observation: Dict[str, Any],
        traffic_data: Dict[str, Any]
    ) -> DependencyAnalysisResult:
        """
        Rule 5: Evaluates road waterlogging propagating to vehicular slowdown.
        """
        rule_id = "RULE_ROAD_WATERLOGGING_TO_TRAFFIC_SLOWDOWN"
        road_id = road_observation.get("asset_id") or road_observation.get("flood_observation_id", "ROAD_UNKNOWN")
        route_id = traffic_data.get("route_id") or traffic_data.get("asset_id", "ROUTE_UNKNOWN")

        supporting: List[DependencyEvidenceItem] = []
        contradictory: List[DependencyEvidenceItem] = []

        is_conn, conn_desc, conn_conf = self.verify_asset_connectivity(road_id, route_id)
        if not is_conn:
            return DependencyAnalysisResult(
                dependency_type=DependencyType.ROAD_WATERLOGGING_TO_TRAFFIC_SLOWDOWN,
                cause_asset_id=road_id,
                effect_asset_id=route_id,
                status=DependencyStatus.INSUFFICIENT_CONNECTIVITY,
                confidence_score=0.1,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=f"Road waterlogged segment {road_id} does not intersect route corridor {route_id}.",
                missing_evidence=["Road corridor mapping."]
            )

        passability = road_observation.get("road_passability", "UNKNOWN")
        speed_drop = float(traffic_data.get("speed_drop_pct", 0.0) or 0.0)
        delay_min_traffic = float(traffic_data.get("congestion_delay_min", 0.0) or 0.0)

        is_disrupted = (
            passability in ("SLOW", "RESTRICTED", "CLOSED")
            or speed_drop >= self.config.traffic_speed_drop_pct
            or delay_min_traffic >= self.config.traffic_severe_delay_min
        )

        t_road = road_observation.get("observed_time")
        t_traf = traffic_data.get("timestamp")
        is_plausible, delay_min, temp_msg = self.check_temporal_plausibility(
            t_road, t_traf, self.config.road_flood_to_traffic_delay_min
        )

        # Contradiction check: Traffic moving normally despite flood claim
        if traffic_data.get("traffic_status") == "FREE_FLOW" and passability == "CLOSED":
            contradictory.append(DependencyEvidenceItem(
                source_id=route_id,
                source_type="SENSOR",
                timestamp=t_traf,
                is_supporting=False,
                summary="Traffic telemetry shows normal free flow speeds, contradicting complete road closure claim."
            ))
            return DependencyAnalysisResult(
                dependency_type=DependencyType.ROAD_WATERLOGGING_TO_TRAFFIC_SLOWDOWN,
                cause_asset_id=road_id,
                effect_asset_id=route_id,
                status=DependencyStatus.REFUTED_CONTRADICTED,
                confidence_score=0.20,
                severity_level=SeverityLevel.MODERATE,
                rule_id=rule_id,
                explanation="Contradiction flagged: Traffic monitoring reports normal free flow speeds.",
                contradictory_evidence=contradictory
            )

        if is_disrupted and is_plausible:
            supporting.append(DependencyEvidenceItem(
                source_id=road_id,
                source_type="OBSERVATION",
                timestamp=t_road,
                is_supporting=True,
                summary=f"Road passability reported as {passability}."
            ))
            supporting.append(DependencyEvidenceItem(
                source_id=route_id,
                source_type="SENSOR",
                timestamp=t_traf,
                is_supporting=True,
                summary=f"Traffic speed dropped by {speed_drop}%, adding +{delay_min_traffic} min delay."
            ))

            sev = SeverityLevel.MODERATE
            if passability == "CLOSED" or delay_min_traffic >= 30.0:
                sev = SeverityLevel.CRITICAL
            elif passability == "RESTRICTED" or delay_min_traffic >= 15.0:
                sev = SeverityLevel.HIGH

            conf = min(0.95, conn_conf * 0.92)

            return DependencyAnalysisResult(
                dependency_type=DependencyType.ROAD_WATERLOGGING_TO_TRAFFIC_SLOWDOWN,
                cause_asset_id=road_id,
                effect_asset_id=route_id,
                status=DependencyStatus.POSSIBLE_DEPENDENCY,
                confidence_score=round(conf, 3),
                severity_level=sev,
                rule_id=rule_id,
                explanation=f"Carriageway inundation ({passability}) caused severe traffic friction: speed dropped {speed_drop}% (+{delay_min_traffic} min bottleneck delay).",
                supporting_evidence=supporting,
                temporal_delay_minutes=delay_min,
                edge_attributes={"relationship_type": EdgeType.IMPACTS_ROUTE.value, "confidence": round(conf, 3)}
            )

        return DependencyAnalysisResult(
            dependency_type=DependencyType.ROAD_WATERLOGGING_TO_TRAFFIC_SLOWDOWN,
            cause_asset_id=road_id,
            effect_asset_id=route_id,
            status=DependencyStatus.UNCERTAIN_INVESTIGATING,
            confidence_score=0.30,
            severity_level=SeverityLevel.LOW,
            rule_id=rule_id,
            explanation="Traffic flow within normal tolerance; waterlogging impact not confirmed.",
            missing_evidence=["Traffic speed sensor telemetry"]
        )

    # -------------------------------------------------------------------------
    # Rule 6: Underpass Inundation -> Travel Disruption
    # -------------------------------------------------------------------------
    def analyze_underpass_inundation_to_travel_disruption(
        self,
        underpass_observation: Dict[str, Any],
        transit_corridor_data: Dict[str, Any]
    ) -> DependencyAnalysisResult:
        """
        Rule 6: Evaluates subway underpass submergence forcing vehicular & transit route shutdown.
        """
        rule_id = "RULE_UNDERPASS_INUNDATION_TO_TRAVEL_DISRUPTION"
        subway_id = underpass_observation.get("asset_id", "UNDERPASS_UNKNOWN")
        transit_id = transit_corridor_data.get("transit_id") or transit_corridor_data.get("route_id", "CORRIDOR_UNKNOWN")

        supporting: List[DependencyEvidenceItem] = []
        contradictory: List[DependencyEvidenceItem] = []

        is_conn, conn_desc, conn_conf = self.verify_asset_connectivity(subway_id, transit_id)
        if not is_conn:
            return DependencyAnalysisResult(
                dependency_type=DependencyType.UNDERPASS_INUNDATION_TO_TRAVEL_DISRUPTION,
                cause_asset_id=subway_id,
                effect_asset_id=transit_id,
                status=DependencyStatus.INSUFFICIENT_CONNECTIVITY,
                confidence_score=0.1,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=f"Underpass {subway_id} does not intersect transit corridor {transit_id}.",
                missing_evidence=["Underpass-to-transit route mapping."]
            )

        depth = float(underpass_observation.get("water_depth_cm", 0.0) or 0.0)
        passability = underpass_observation.get("road_passability", "UNKNOWN")

        is_closed = depth >= self.config.underpass_closure_depth_cm or passability == "CLOSED"

        if is_closed:
            supporting.append(DependencyEvidenceItem(
                source_id=subway_id,
                source_type="OBSERVATION",
                timestamp=underpass_observation.get("observed_time"),
                is_supporting=True,
                summary=f"Underpass barrel submerged with {depth}cm standing water (passability: {passability})."
            ))
            supporting.append(DependencyEvidenceItem(
                source_id=transit_id,
                source_type="TRANSIT_SERVICE",
                timestamp=transit_corridor_data.get("timestamp"),
                is_supporting=True,
                summary=f"Transit corridor closed / diverted: {transit_corridor_data.get('status', 'BLOCKED')}."
            ))

            sev = SeverityLevel.CRITICAL if depth >= 50.0 else SeverityLevel.HIGH
            conf = min(0.98, conn_conf * 0.95)

            return DependencyAnalysisResult(
                dependency_type=DependencyType.UNDERPASS_INUNDATION_TO_TRAVEL_DISRUPTION,
                cause_asset_id=subway_id,
                effect_asset_id=transit_id,
                status=DependencyStatus.POSSIBLE_DEPENDENCY,
                confidence_score=round(conf, 3),
                severity_level=sev,
                rule_id=rule_id,
                explanation=f"Underpass barrel inundated ({depth}cm depth) enforced physical barrier deployment and complete corridor closure on {transit_id}.",
                supporting_evidence=supporting,
                edge_attributes={"relationship_type": EdgeType.IMPACTS_ROUTE.value, "confidence": round(conf, 3), "impact_type": "TOTAL_CLOSURE"}
            )

        return DependencyAnalysisResult(
            dependency_type=DependencyType.UNDERPASS_INUNDATION_TO_TRAVEL_DISRUPTION,
            cause_asset_id=subway_id,
            effect_asset_id=transit_id,
            status=DependencyStatus.UNCERTAIN_INVESTIGATING,
            confidence_score=0.30,
            severity_level=SeverityLevel.LOW,
            rule_id=rule_id,
            explanation="Underpass water level is below closure threshold; corridor remains passable.",
            missing_evidence=["Water depth exceeding closure threshold."]
        )

    # -------------------------------------------------------------------------
    # Rule 7: Power Failure -> Pump Failure (CRITICAL INVARIANT 5)
    # -------------------------------------------------------------------------
    def analyze_power_to_pump_failure(
        self,
        power_feeder_data: Dict[str, Any],
        pump_station_data: Dict[str, Any]
    ) -> DependencyAnalysisResult:
        """
        Rule 7 & Invariant 5:
        Do NOT infer that a power failure caused a pump failure unless:
        1. The supply dependency (POWERED_BY) is verified, AND
        2. Supporting power failure evidence exists and temporally precedes pump stoppage.
        """
        rule_id = "RULE_POWER_FAILURE_TO_PUMP_FAILURE"
        feeder_id = power_feeder_data.get("asset_id", "FEEDER_UNKNOWN")
        pump_id = pump_station_data.get("asset_id", "PUMP_UNKNOWN")

        supporting: List[DependencyEvidenceItem] = []
        contradictory: List[DependencyEvidenceItem] = []
        missing: List[str] = []
        alt_hypotheses: List[str] = []

        # 1. VERIFY POWER SUPPLY DEPENDENCY (INVARIANT 5)
        # Check direct POWERED_BY edge in asset topology
        has_power_edge = False
        edge_status = "UNVERIFIED"
        edge_conf = 0.5

        if self.asset_graph.has_edge(feeder_id, pump_id):
            e_data = self.asset_graph[feeder_id][pump_id]
            if e_data.get("relationship_type") == "POWERED_BY":
                has_power_edge = True
                edge_status = e_data.get("relationship_status", "UNVERIFIED")
                edge_conf = e_data.get("confidence", 0.7)
        # In case edge is pump -> feeder
        elif self.asset_graph.has_edge(pump_id, feeder_id):
            e_data = self.asset_graph[pump_id][feeder_id]
            if e_data.get("relationship_type") == "POWERED_BY":
                has_power_edge = True
                edge_status = e_data.get("relationship_status", "UNVERIFIED")
                edge_conf = e_data.get("confidence", 0.7)

        # INVARIANT 5: If supply dependency does NOT exist, REJECT IMMEDIATELY!
        if not has_power_edge:
            alt_hypotheses.append("Mechanical pump impeller jamming (debris/silt)")
            alt_hypotheses.append("Thermal motor trip due to continuous high-load running")
            alt_hypotheses.append("Auxiliary diesel generator failure (independent of utility grid)")

            return DependencyAnalysisResult(
                dependency_type=DependencyType.POWER_FAILURE_TO_PUMP_FAILURE,
                cause_asset_id=feeder_id,
                effect_asset_id=pump_id,
                status=DependencyStatus.REFUTED_CONTRADICTED,
                confidence_score=0.10,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=f"INVARIANT ENFORCED: No verified electric supply dependency between power feeder {feeder_id} and pump station {pump_id}. Unrelated power outages cannot be inferred as pump failure causes.",
                missing_evidence=[f"Verified electrical single-line diagram linking {feeder_id} to {pump_id}"],
                alternative_hypotheses=alt_hypotheses
            )

        # If edge exists but is unverified in asset register
        if edge_status != "VERIFIED":
            missing.append("Electric utility supply relationship is UNVERIFIED; awaiting official feeder registry confirmation.")

        # 2. Check Feeder Outage and Pump Failure Status
        feeder_status = power_feeder_data.get("status", "ACTIVE")
        is_feeder_out = feeder_status in ("OFFLINE", "POWER_OUTAGE", "TRIPPED")

        pump_status = pump_station_data.get("pump_status", "ACTIVE")
        is_pump_stopped = pump_status in ("FAILED", "OFFLINE")

        if not is_feeder_out:
            contradictory.append(DependencyEvidenceItem(
                source_id=feeder_id,
                source_type="SENSOR",
                timestamp=power_feeder_data.get("timestamp"),
                is_supporting=False,
                summary=f"Power feeder {feeder_id} is energized (status: {feeder_status}). Pump failure was NOT caused by grid loss."
            ))
            alt_hypotheses.append("Mechanical impeller blockage")
            alt_hypotheses.append("Localized circuit breaker trip inside pump house")

            return DependencyAnalysisResult(
                dependency_type=DependencyType.POWER_FAILURE_TO_PUMP_FAILURE,
                cause_asset_id=feeder_id,
                effect_asset_id=pump_id,
                status=DependencyStatus.REFUTED_CONTRADICTED,
                confidence_score=0.15,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=f"Feeder {feeder_id} telemetry confirms electrical grid is normal ({feeder_status}). Pump failure was mechanical, not grid loss.",
                contradictory_evidence=contradictory,
                alternative_hypotheses=alt_hypotheses
            )

        # 3. Temporal plausibility check
        t_feeder = power_feeder_data.get("timestamp")
        t_pump = pump_station_data.get("timestamp")
        is_plausible, delay_min, temp_msg = self.check_temporal_plausibility(
            t_feeder, t_pump, self.config.power_to_pump_failure_max_delay_min
        )

        if not is_plausible:
            contradictory.append(DependencyEvidenceItem(
                source_id=feeder_id,
                source_type="SENSOR",
                timestamp=t_feeder,
                is_supporting=False,
                summary=f"Timing incompatible: {temp_msg}"
            ))
            return DependencyAnalysisResult(
                dependency_type=DependencyType.POWER_FAILURE_TO_PUMP_FAILURE,
                cause_asset_id=feeder_id,
                effect_asset_id=pump_id,
                status=DependencyStatus.REFUTED_CONTRADICTED,
                confidence_score=0.20,
                severity_level=SeverityLevel.LOW,
                rule_id=rule_id,
                explanation=temp_msg,
                contradictory_evidence=contradictory
            )

        # If both failed and supply link exists
        supporting.append(DependencyEvidenceItem(
            source_id=feeder_id,
            source_type="ASSET",
            timestamp=t_feeder,
            is_supporting=True,
            summary=f"Power feeder {feeder_id} experienced blackout (status: {feeder_status})."
        ))
        supporting.append(DependencyEvidenceItem(
            source_id=pump_id,
            source_type="ASSET",
            timestamp=t_pump,
            is_supporting=True,
            summary=f"Pump station {pump_id} de-energized {round(delay_min or 0, 1)} min later."
        ))

        # Check backup generator existence
        has_generator = pump_station_data.get("has_backup_generator", False)
        if has_generator:
            missing.append("Backup generator telemetry missing; verifying why generator did not start.")

        status = DependencyStatus.POSSIBLE_DEPENDENCY
        if edge_status == "VERIFIED" and not has_generator:
            status = DependencyStatus.CONFIRMED_DEPENDENCY
            conf = 0.92
        elif edge_status == "VERIFIED" and has_generator:
            status = DependencyStatus.POSSIBLE_DEPENDENCY
            conf = 0.72
        else:
            status = DependencyStatus.UNCERTAIN_INVESTIGATING
            conf = 0.50

        sev = SeverityLevel.HIGH if is_pump_stopped else SeverityLevel.MODERATE

        return DependencyAnalysisResult(
            dependency_type=DependencyType.POWER_FAILURE_TO_PUMP_FAILURE,
            cause_asset_id=feeder_id,
            effect_asset_id=pump_id,
            status=status,
            confidence_score=round(conf, 3),
            severity_level=sev,
            rule_id=rule_id,
            explanation=f"Power feeder {feeder_id} outage preceded pump station {pump_id} stoppage by {round(delay_min or 0, 1)} min on verified power circuit ({edge_status}).",
            supporting_evidence=supporting,
            missing_evidence=missing,
            temporal_delay_minutes=delay_min,
            edge_attributes={"relationship_type": EdgeType.POSSIBLE_DEPENDENCY.value, "confidence": round(conf, 3)}
        )

    # -------------------------------------------------------------------------
    # Graph Edge Conversion
    # -------------------------------------------------------------------------
    def convert_analysis_to_graph_edge(
        self,
        analysis: DependencyAnalysisResult,
        model_version: str = "FloodDependencyEngine_v6"
    ) -> Optional[Dict[str, Any]]:
        """
        Converts a DependencyAnalysisResult into a validated graph edge dict
        compatible with UrbanPulseEvidenceGraph.
        """
        if analysis.status not in (
            DependencyStatus.CONFIRMED_DEPENDENCY,
            DependencyStatus.POSSIBLE_DEPENDENCY
        ):
            return None

        # Build evidence ID list
        ev_ids = [item.source_id for item in analysis.supporting_evidence]
        if not ev_ids:
            ev_ids = [analysis.cause_asset_id, analysis.effect_asset_id]

        rel_type = analysis.edge_attributes.get("relationship_type", EdgeType.POSSIBLE_DEPENDENCY.value)

        return {
            "source": analysis.cause_asset_id,
            "target": analysis.effect_asset_id,
            "relationship_type": rel_type,
            "confidence": analysis.confidence_score,
            "evidence_ids": ev_ids,
            "rule_or_model_version": model_version,
            "explanation": analysis.explanation,
            "severity_level": analysis.severity_level.value,
            "status": analysis.status.value,
            "rule_id": analysis.rule_id,
            "is_causation_proven": analysis.is_causation_proven,
            "alternative_hypotheses": analysis.alternative_hypotheses
        }
