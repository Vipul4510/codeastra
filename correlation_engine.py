"""
UrbanPulse Rule-Based Multimodal Incident Correlation Engine
============================================================
Decides whether incoming multimodal observations:
- describe the same incident (SAME_INCIDENT),
- are related but represent different incidents (RELATED_BUT_DISTINCT),
- are duplicates (DUPLICATE),
- contradict each other (CONTRADICTORY), or
- are unrelated or uncertain (UNRELATED).

Core Architectural Invariants:
1. Spatial candidate filtering via Haversine distance.
2. Hard compatibility rules applied BEFORE soft scoring.
3. NEVER merge observations solely due to spatio-temporal proximity.
4. Reposts / forwarded copies treated as ONE evidence chain.
5. Contradictions and missing data preserved transparently.
6. Explicit relationship label and explanation produced for every evaluated pair.
7. Candidate incident clusters built without over-merging distinct events.
8. Severity assessment kept strictly separate from incident identity and confidence.
"""

import os
import sys
import math
import json
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import numpy as np

# Fallback imports to support execution as a module or direct script
try:
    from .rule_definitions import (
        CorrelationRuleConfig,
        CAUSAL_RELATIONSHIPS,
        haversine_distance_km,
        compute_token_jaccard_similarity,
        check_hard_incompatibility,
        check_exact_or_repost_duplicate,
        check_contradiction,
        calculate_heuristic_correlation_score,
    )
except (ImportError, ValueError):
    from rule_definitions import (
        CorrelationRuleConfig,
        CAUSAL_RELATIONSHIPS,
        haversine_distance_km,
        compute_token_jaccard_similarity,
        check_hard_incompatibility,
        check_exact_or_repost_duplicate,
        check_contradiction,
        calculate_heuristic_correlation_score,
    )


class IncidentCorrelationEngine:
    """
    Multimodal Incident Correlation & Clustering Engine.
    """

    def __init__(self, config: Optional[CorrelationRuleConfig] = None):
        self.config = config or CorrelationRuleConfig()

    def evaluate_observation_pair(
        self,
        event_a: Dict[str, Any],
        event_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates a single pair of observations (A, B) and assigns:
        - relation_label: SAME_INCIDENT, RELATED_BUT_DISTINCT, DUPLICATE, CONTRADICTORY, UNRELATED
        - confidence_score: heuristic confidence [0.0 - 1.0]
        - rule_id: triggered rule identifier
        - explanation: human-readable rationale
        - evidence_chain_count: independent corroboration sources count
        """
        id_a = str(event_a.get("event_id", event_a.get("raw_id", "A")))
        id_b = str(event_b.get("event_id", event_b.get("raw_id", "B")))

        # 1. Temporal calculation (minutes)
        epoch_a = event_a.get("observation_time_epoch_ms")
        epoch_b = event_b.get("observation_time_epoch_ms")
        
        if epoch_a is not None and epoch_b is not None:
            time_diff_min = abs(epoch_a - epoch_b) / (60.0 * 1000.0)
        else:
            time_diff_min = 0.0  # Missing timestamp penalty handled via flags

        # 2. Geographic distance (Haversine formula in km)
        lat_a, lon_a = event_a.get("latitude"), event_a.get("longitude")
        lat_b, lon_b = event_b.get("latitude"), event_b.get("longitude")
        
        has_coords = (lat_a is not None and lon_a is not None and lat_b is not None and lon_b is not None)
        if has_coords:
            geo_dist_km = haversine_distance_km(float(lat_a), float(lon_a), float(lat_b), float(lon_b))
        else:
            geo_dist_km = None

        # 3. Text and Semantic similarity
        text_a = str(event_a.get("text_cleaned", event_a.get("text_raw", "")))
        text_b = str(event_b.get("text_cleaned", event_b.get("text_raw", "")))
        text_sim = compute_token_jaccard_similarity(text_a, text_b)

        # 4. Source & Reliability Metrics
        source_a = event_a.get("source_channel", "UNKNOWN")
        source_b = event_b.get("source_channel", "UNKNOWN")
        same_source = (source_a == source_b and source_a != "UNKNOWN")
        
        rel_a = float(event_a.get("source_reliability", 0.50))
        rel_b = float(event_b.get("source_reliability", 0.50))
        reliability_product = round(rel_a * rel_b, 4)

        # 5. Locality and Asset alignment
        loc_a = event_a.get("locality_normalized", "")
        loc_b = event_b.get("locality_normalized", "")
        same_locality = bool(loc_a and loc_b and loc_a.lower() == loc_b.lower())

        cat_a = event_a.get("canonical_event_type", "GENERAL_INCIDENT")
        cat_b = event_b.get("canonical_event_type", "GENERAL_INCIDENT")

        # Determine Category Match Type
        if cat_a == cat_b:
            cat_match_type = "EXACT"
        elif cat_b in CAUSAL_RELATIONSHIPS.get(cat_a, set()) or cat_a in CAUSAL_RELATIONSHIPS.get(cat_b, set()):
            cat_match_type = "CAUSAL"
        else:
            cat_match_type = "COMPATIBLE"

        # -----------------------------------------------------------------
        # STEP A: HARD COMPATIBILITY RULES (Evaluated before any scoring)
        # -----------------------------------------------------------------

        # Rule H1: Hard Distance / Time Horizon Exceeded
        is_hard_incomp, hard_lbl, hard_exp = check_hard_incompatibility(
            dist_km=geo_dist_km,
            time_diff_min=time_diff_min,
            config=self.config
        )
        if is_hard_incomp:
            return {
                "pair_id": f"{id_a}__{id_b}",
                "event_id_a": id_a,
                "event_id_b": id_b,
                "relation_label": hard_lbl,
                "confidence_score": 0.0,
                "rule_id": "RULE_HARD_SPATIAL_TEMPORAL_CUTOFF",
                "explanation": hard_exp,
                "evidence_chain_count": 2,
                "geo_distance_km": geo_dist_km,
                "time_diff_minutes": round(time_diff_min, 2),
                "text_similarity": text_sim,
                "category_match_type": cat_match_type,
            }

        # Rule H2: Exact Duplicate & Repost Evidence Chain (Requirement 6)
        is_dupe, dupe_lbl, dupe_exp, chain_count = check_exact_or_repost_duplicate(
            event_a=event_a,
            event_b=event_b,
            dist_km=geo_dist_km,
            time_diff_min=time_diff_min,
            text_sim=text_sim,
            config=self.config
        )
        if is_dupe:
            return {
                "pair_id": f"{id_a}__{id_b}",
                "event_id_a": id_a,
                "event_id_b": id_b,
                "relation_label": dupe_lbl,
                "confidence_score": 0.98,
                "rule_id": "RULE_EXACT_DUPLICATE_OR_REPOST_CHAIN",
                "explanation": dupe_exp,
                "evidence_chain_count": chain_count,
                "geo_distance_km": geo_dist_km,
                "time_diff_minutes": round(time_diff_min, 2),
                "text_similarity": text_sim,
                "category_match_type": cat_match_type,
            }

        # Rule H3: Factual Contradiction Check (Requirement 7)
        is_contra, contra_lbl, contra_exp = check_contradiction(
            event_a=event_a,
            event_b=event_b,
            dist_km=geo_dist_km,
            time_diff_min=time_diff_min,
            config=self.config
        )
        if is_contra:
            return {
                "pair_id": f"{id_a}__{id_b}",
                "event_id_a": id_a,
                "event_id_b": id_b,
                "relation_label": contra_lbl,
                "confidence_score": 0.85,
                "rule_id": "RULE_CONTRADICTORY_ASSERTION",
                "explanation": contra_exp,
                "evidence_chain_count": 2,
                "geo_distance_km": geo_dist_km,
                "time_diff_minutes": round(time_diff_min, 2),
                "text_similarity": text_sim,
                "category_match_type": cat_match_type,
            }

        # -----------------------------------------------------------------
        # STEP B: HEURISTIC MULTIMODAL SCORING
        # -----------------------------------------------------------------
        heuristic_score = calculate_heuristic_correlation_score(
            dist_km=geo_dist_km,
            time_diff_min=time_diff_min,
            text_sim=text_sim,
            cat_match_type=cat_match_type,
            same_source=same_source,
            rel_product=reliability_product
        )

        # -----------------------------------------------------------------
        # STEP C: PRINCIPLE 5 CHECK
        # "Never merge observations solely because they are geographically
        #  close or occur around the same time."
        # -----------------------------------------------------------------
        is_spatially_close = (geo_dist_km is not None and geo_dist_km <= self.config.same_incident_max_distance_km)
        is_temporally_close = (time_diff_min <= self.config.same_incident_max_time_minutes)

        # If geographically & temporally close BUT categories are unrelated AND text similarity is negligible:
        # DO NOT MERGE! Assign RELATED_BUT_DISTINCT or UNRELATED.
        if is_spatially_close and is_temporally_close:
            if cat_match_type not in ("EXACT", "CAUSAL") and text_sim < 0.25:
                return {
                    "pair_id": f"{id_a}__{id_b}",
                    "event_id_a": id_a,
                    "event_id_b": id_b,
                    "relation_label": "RELATED_BUT_DISTINCT",
                    "confidence_score": 0.52,
                    "rule_id": "RULE_CO_LOCATED_INDEPENDENT_EVENTS",
                    "explanation": (
                        f"Principle 5 Safeguard: Events are co-located in {loc_a} within {time_diff_min:.1f} min, "
                        f"but categories '{cat_a}' vs '{cat_b}' lack causal corroboration and text similarity is low ({text_sim:.2f}). "
                        "Kept as distinct incidents."
                    ),
                    "evidence_chain_count": 2,
                    "geo_distance_km": geo_dist_km,
                    "time_diff_minutes": round(time_diff_min, 2),
                    "text_similarity": text_sim,
                    "category_match_type": cat_match_type,
                }

        # -----------------------------------------------------------------
        # STEP D: SAME INCIDENT vs RELATED-BUT-DISTINCT CLASSIFICATION
        # -----------------------------------------------------------------

        # Case 1: SAME_INCIDENT
        # Requires: localized distance (<= 0.85 km), time window (<= 45 min),
        # category compatibility (EXACT or direct CAUSAL), and text overlap (>= 0.40) OR exact same asset
        has_same_asset = (
            event_a.get("asset_id") and
            event_a.get("asset_id") == event_b.get("asset_id")
        )

        meets_same_incident_envelope = (
            (geo_dist_km is None or geo_dist_km <= self.config.same_incident_max_distance_km) and
            time_diff_min <= self.config.same_incident_max_time_minutes and
            (same_locality or geo_dist_km is not None and geo_dist_km <= 0.50)
        )

        has_corroboration = (
            cat_match_type in ("EXACT", "CAUSAL") and
            (text_sim >= self.config.same_incident_min_text_similarity or has_same_asset or cat_match_type == "EXACT")
        )

        if meets_same_incident_envelope and has_corroboration and heuristic_score >= self.config.same_incident_min_heuristic_score:
            exp = (
                f"Corroborating observations of the same localized incident in {loc_a}: "
                f"distance={geo_dist_km or 0.0:.2f}km, time offset={time_diff_min:.1f}min, "
                f"category alignment='{cat_match_type}', heuristic score={heuristic_score:.2f}."
            )
            return {
                "pair_id": f"{id_a}__{id_b}",
                "event_id_a": id_a,
                "event_id_b": id_b,
                "relation_label": "SAME_INCIDENT",
                "confidence_score": heuristic_score,
                "rule_id": "RULE_SAME_INCIDENT_CORROBORATION",
                "explanation": exp,
                "evidence_chain_count": 2,
                "geo_distance_km": geo_dist_km,
                "time_diff_minutes": round(time_diff_min, 2),
                "text_similarity": text_sim,
                "category_match_type": cat_match_type,
            }

        # Case 2: RELATED_BUT_DISTINCT
        # Broader corridor horizon (distance <= 3.5 km, time <= 180 min),
        # causal category link OR shared suburban zone
        meets_related_envelope = (
            (geo_dist_km is None or geo_dist_km <= self.config.related_distinct_max_distance_km) and
            time_diff_min <= self.config.related_distinct_max_time_minutes
        )

        if meets_related_envelope and (cat_match_type in ("CAUSAL", "EXACT") or same_locality) and heuristic_score >= self.config.related_distinct_min_heuristic_score:
            exp = (
                f"Cascading or corridor-related incident: Category '{cat_a}' and '{cat_b}' share causal/regional episode in {loc_a or 'Western Suburbs'} "
                f"(dist={geo_dist_km or 0.0:.2f}km, delta_t={time_diff_min:.1f}min). Kept separate to prevent artificial cluster bloat."
            )
            return {
                "pair_id": f"{id_a}__{id_b}",
                "event_id_a": id_a,
                "event_id_b": id_b,
                "relation_label": "RELATED_BUT_DISTINCT",
                "confidence_score": heuristic_score,
                "rule_id": "RULE_RELATED_DISTINCT_CORRIDOR_CASCADE",
                "explanation": exp,
                "evidence_chain_count": 2,
                "geo_distance_km": geo_dist_km,
                "time_diff_minutes": round(time_diff_min, 2),
                "text_similarity": text_sim,
                "category_match_type": cat_match_type,
            }

        # Case 3: UNRELATED
        exp = (
            f"Unrelated events: Spatio-temporal distance ({geo_dist_km or 0.0:.2f}km, {time_diff_min:.1f}min) "
            f"or disparate categories ('{cat_a}' vs '{cat_b}') confirm independent status."
        )
        return {
            "pair_id": f"{id_a}__{id_b}",
            "event_id_a": id_a,
            "event_id_b": id_b,
            "relation_label": "UNRELATED",
            "confidence_score": heuristic_score,
            "rule_id": "RULE_UNRELATED_INDEPENDENT",
            "explanation": exp,
            "evidence_chain_count": 2,
            "geo_distance_km": geo_dist_km,
            "time_diff_minutes": round(time_diff_min, 2),
            "text_similarity": text_sim,
            "category_match_type": cat_match_type,
        }

    def generate_candidate_pairs(
        self,
        events: List[Dict[str, Any]],
        max_pairs: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Generates candidate observation pairs using temporal and spatial blocking.
        Avoids O(N^2) explosion by indexing observations into locality & temporal buckets.
        """
        # Bucket by locality
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for ev in events:
            loc = ev.get("locality_normalized", "Unknown")
            buckets.setdefault(loc, []).append(ev)

        candidate_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        # Generate candidates within locality buckets (sorted chronologically)
        for loc, ev_list in buckets.items():
            # Sort by epoch
            sorted_evs = sorted(
                ev_list,
                key=lambda x: x.get("observation_time_epoch_ms") or 0
            )
            n = len(sorted_evs)
            for i in range(n):
                ev_a = sorted_evs[i]
                epoch_a = ev_a.get("observation_time_epoch_ms")
                id_a = ev_a.get("event_id")

                for j in range(i + 1, min(n, i + 30)):  # sliding temporal window of 30 nearby events
                    ev_b = sorted_evs[j]
                    epoch_b = ev_b.get("observation_time_epoch_ms")
                    id_b = ev_b.get("event_id")

                    if id_a == id_b:
                        continue

                    # Check temporal window
                    if epoch_a and epoch_b:
                        diff_min = abs(epoch_b - epoch_a) / (60.0 * 1000.0)
                        if diff_min > self.config.candidate_max_time_minutes:
                            break  # Since sorted, all subsequent will be even further

                    pair_key = tuple(sorted([id_a, id_b]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        candidate_pairs.append((ev_a, ev_b))
                        if max_pairs and len(candidate_pairs) >= max_pairs:
                            return candidate_pairs

        return candidate_pairs

    def build_candidate_clusters(
        self,
        events: List[Dict[str, Any]],
        pairwise_evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Builds candidate incident clusters using graph connected components
        from verified SAME_INCIDENT relationship edges.
        
        Strict Guardrails:
        1. CONTRADICTORY edges block merges and are attached as active conflicts.
        2. Singletons remain independent clusters.
        3. Severities are evaluated completely separately from identity.
        """
        # Map event_id -> event
        event_map = {e.get("event_id"): e for e in events}

        # Build adjacency list for SAME_INCIDENT edges
        adj: Dict[str, Set[str]] = {e_id: set() for e_id in event_map}
        conflicts_map: Dict[str, List[Dict[str, Any]]] = {e_id: [] for e_id in event_map}

        for p in pairwise_evaluations:
            e_a = p["event_id_a"]
            e_b = p["event_id_b"]
            label = p["relation_label"]
            score = p.get("confidence_score", 0.0)

            if label == "SAME_INCIDENT" and score >= self.config.clustering_confidence_threshold:
                if e_a in adj and e_b in adj:
                    adj[e_a].add(e_b)
                    adj[e_b].add(e_a)
            elif label == "CONTRADICTORY":
                if e_a in conflicts_map:
                    conflicts_map[e_a].append(p)
                if e_b in conflicts_map:
                    conflicts_map[e_b].append(p)

        # Connected components via BFS
        visited: Set[str] = set()
        raw_clusters: List[List[str]] = []

        for node in adj:
            if node not in visited:
                comp = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                raw_clusters.append(comp)

        # Assemble Standardized Candidate Clusters
        clusters_output: List[Dict[str, Any]] = []
        cluster_idx = 1

        for comp in raw_clusters:
            comp_events = [event_map[ev_id] for ev_id in comp if ev_id in event_map]
            if not comp_events:
                continue

            # Cluster Identity
            cluster_id = f"CAND-INC-{cluster_idx:06d}"
            cluster_idx += 1

            # Primary Category & Modalities
            cat_counts = pd.Series([e.get("canonical_event_type") for e in comp_events]).value_counts()
            primary_cat = cat_counts.index[0]

            modalities = list(set(e.get("source_modality") for e in comp_events))
            sources = list(set(e.get("source_channel") for e in comp_events))

            # Spatial Centroid
            valid_lats = [float(e["latitude"]) for e in comp_events if e.get("latitude") is not None]
            valid_lons = [float(e["longitude"]) for e in comp_events if e.get("longitude") is not None]
            
            centroid_lat = round(float(np.mean(valid_lats)), 6) if valid_lats else None
            centroid_lon = round(float(np.mean(valid_lons)), 6) if valid_lons else None

            # Localities
            localities = list(set(e.get("locality_normalized") for e in comp_events if e.get("locality_normalized")))
            primary_locality = localities[0] if localities else "Western Suburbs"

            # Temporal Horizon
            epochs = [e.get("observation_time_epoch_ms") for e in comp_events if e.get("observation_time_epoch_ms")]
            iso_times = [e.get("observation_time_iso") for e in comp_events if e.get("observation_time_iso")]
            
            first_seen = min(iso_times) if iso_times else "Unknown"
            last_seen = max(iso_times) if iso_times else "Unknown"

            # Gather Active Conflicts
            comp_conflicts = []
            for ev_id in comp:
                comp_conflicts.extend(conflicts_map.get(ev_id, []))

            # Distinct evidence chains
            chain_counts = sum(1 for e in comp_events if not e.get("is_repost_or_copy") and not e.get("is_exact_duplicate"))
            chain_counts = max(1, chain_counts)

            # Correlation Confidence (Separate from Severity)
            # Higher with multi-source diversity, penalized by contradictions
            diversity_bonus = min(0.30, len(modalities) * 0.10)
            avg_rel = float(np.mean([float(e.get("source_reliability", 0.50)) for e in comp_events]))
            base_conf = 0.50 + diversity_bonus + (avg_rel * 0.20)
            if comp_conflicts:
                base_conf -= 0.15  # Penalty for unresolved contradiction
            corr_confidence = round(min(0.99, max(0.40, base_conf)), 2)

            # Severity Assessment (Independent from Identity & Confidence)
            severity = self.evaluate_incident_severity(comp_events, primary_cat)

            clusters_output.append({
                "cluster_id": cluster_id,
                "primary_event_type": primary_cat,
                "locality": primary_locality,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "member_event_count": len(comp_events),
                "independent_evidence_chains": chain_counts,
                "member_event_ids": comp,
                "contributing_modalities": modalities,
                "contributing_sources": sources,
                "correlation_confidence": corr_confidence,
                "severity_level": severity["level"],
                "severity_rationale": severity["rationale"],
                "active_conflicts_count": len(comp_conflicts),
                "has_contradictions": len(comp_conflicts) > 0,
            })

        return clusters_output

    def evaluate_incident_severity(
        self,
        events: List[Dict[str, Any]],
        primary_cat: str
    ) -> Dict[str, str]:
        """
        Requirement 10: Evaluates severity completely independently
        from incident identity and correlation confidence.
        
        Severity Levels:
        - CRITICAL: Water depth > 50cm, road impassable / subway closed, rail tracks submerged
        - MAJOR: Water depth > 30cm, significant traffic gridlock, bus delay > 30min
        - MODERATE: Localized waterlogging, traffic slowdown
        - LOW: Minor localized report, clear drain
        """
        depths = []
        is_closed = False
        has_transit_block = False

        for ev in events:
            text = str(ev.get("text_raw", "")).lower()
            if "closed" in text or "impassable" in text or "halted" in text:
                is_closed = True
            if "subway" in text or "track" in text or "rail" in text:
                has_transit_block = True
            
            # Extract depth if present in text
            if "cm" in text:
                for token in text.split():
                    if "cm" in token:
                        clean_num = token.replace("cm", "").replace("depth:", "").strip()
                        try:
                            depths.append(float(clean_num))
                        except ValueError:
                            pass

        max_depth = max(depths) if depths else 0.0

        if max_depth >= 50.0 or (is_closed and has_transit_block) or primary_cat in ("SUBWAY_INUNDATION", "OHE_WIRE_FAILURE"):
            return {
                "level": "CRITICAL",
                "rationale": f"Severe physical impact: Inundation/blockage on key arterial corridor (depth={max_depth}cm, closed={is_closed})."
            }
        elif max_depth >= 30.0 or is_closed or primary_cat in ("ROAD_CLOSURE", "RAIL_SIGNAL_FAILURE"):
            return {
                "level": "MAJOR",
                "rationale": f"Significant mobility disruption: Road impassable or standing water >= 30cm (depth={max_depth}cm)."
            }
        elif max_depth > 10.0 or primary_cat in ("TRAFFIC_SLOWDOWN", "DRAIN_BLOCKAGE", "WATER_LEVEL_RISE"):
            return {
                "level": "MODERATE",
                "rationale": f"Moderate localized slowdown: Moderate waterlogging or congestion (depth={max_depth}cm)."
            }
        elif primary_cat == "ROAD_WATERLOGGING" and max_depth > 0:
            return {
                "level": "LOW",
                "rationale": f"Minor localized water accumulation without major disruption (depth={max_depth}cm)."
            }
        else:
            return {
                "level": "LOW",
                "rationale": "Minor civic notice without significant transit obstruction."
            }
