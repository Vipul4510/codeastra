"""
UrbanPulse Feature Engineering Module for Incident Relationship Classification
================================================================================
Extracts, enriches, and transforms pairwise observation signals for machine learning.

Enforces Strict Leakage Prevention:
1. Excludes all database keys and identifiers (pair_id, observation_ids).
2. Excludes ground-truth target clusters (candidate_incident_id).
3. Excludes metadata fields (is_synthetic, provenance).
4. Derives physically meaningful multimodal, spatio-temporal, and semantic features.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

try:
    from .rule_definitions import (
        CAUSAL_RELATIONSHIPS,
        haversine_distance_km,
        detect_contradictory_text,
        calculate_heuristic_correlation_score
    )
except (ImportError, ValueError):
    from rule_definitions import (
        CAUSAL_RELATIONSHIPS,
        haversine_distance_km,
        detect_contradictory_text,
        calculate_heuristic_correlation_score
    )


class PairwiseFeatureExtractor:
    """
    Feature extractor and transformer for pairwise multimodal incident relationships.
    """

    FEATURE_NAMES = [
        # Spatio-Temporal Base & Non-Linear
        "geo_distance_km",
        "log_geo_distance",
        "spatial_proximity_decay",
        "time_diff_minutes",
        "log_time_diff",
        "temporal_proximity_decay",
        "spatiotemporal_proximity_product",
        # Spatio-Temporal Envelope Flags
        "is_micro_spatial",
        "is_local_spatial",
        "is_corridor_spatial",
        "is_micro_temporal",
        "is_local_temporal",
        "is_corridor_temporal",
        # Text & Semantic
        "text_similarity",
        "is_high_text_similarity",
        "is_moderate_text_similarity",
        "is_low_text_similarity",
        "has_contradiction_keywords",
        # Source & Reliability
        "same_locality",
        "same_source",
        "source_independence",
        "reliability_product",
        "source_reliability_min",
        "source_reliability_max",
        "both_high_reliability",
        # Quality & Verification
        "location_accuracy_mean",
        "location_accuracy_min",
        "both_verified",
        "is_duplicate_candidate_flag",
        # Category Compatibility & Causality
        "same_event_type",
        "is_causal_category_pair",
        "category_compatibility_score",
        # Domain Heuristic Prior
        "phase3_heuristic_affinity_score",
    ]

    def __init__(self, obs_df: Optional[pd.DataFrame] = None):
        """
        Args:
            obs_df: Optional observations dataframe indexed by observation_id
                    used for category and metadata enrichment.
        """
        self.obs_df = None
        if obs_df is not None:
            if "observation_id" in obs_df.columns:
                self.obs_df = obs_df.set_index("observation_id")
            else:
                self.obs_df = obs_df

    def extract_features(
        self,
        df_pairs: pd.DataFrame,
        obs_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Transforms a DataFrame of pairwise records into an enriched numerical feature matrix.
        """
        if obs_df is not None:
            if "observation_id" in obs_df.columns:
                active_obs = obs_df.set_index("observation_id")
            else:
                active_obs = obs_df
        elif self.obs_df is not None:
            active_obs = self.obs_df
        else:
            active_obs = None

        n = len(df_pairs)
        features = pd.DataFrame(index=df_pairs.index)

        # 1. Base Spatio-Temporal Metrics
        geo_dist = df_pairs["geo_distance_km"].values.astype(float)
        time_diff = df_pairs["time_diff_minutes"].values.astype(float)

        features["geo_distance_km"] = geo_dist
        features["log_geo_distance"] = np.log1p(geo_dist)
        features["spatial_proximity_decay"] = np.exp(-geo_dist / 1.0)

        features["time_diff_minutes"] = time_diff
        features["log_time_diff"] = np.log1p(time_diff)
        features["temporal_proximity_decay"] = np.exp(-time_diff / 60.0)
        features["spatiotemporal_proximity_product"] = (
            features["spatial_proximity_decay"] * features["temporal_proximity_decay"]
        )

        # 2. Envelope Threshold Binary Indicators
        features["is_micro_spatial"] = (geo_dist <= 0.15).astype(float)
        features["is_local_spatial"] = (geo_dist <= 0.85).astype(float)
        features["is_corridor_spatial"] = (geo_dist <= 3.50).astype(float)

        features["is_micro_temporal"] = (time_diff <= 20.0).astype(float)
        features["is_local_temporal"] = (time_diff <= 45.0).astype(float)
        features["is_corridor_temporal"] = (time_diff <= 180.0).astype(float)

        # 3. Text Similarity Signals
        text_sim = df_pairs["text_similarity"].values.astype(float)
        features["text_similarity"] = text_sim
        features["is_high_text_similarity"] = (text_sim >= 0.80).astype(float)
        features["is_moderate_text_similarity"] = (text_sim >= 0.40).astype(float)
        features["is_low_text_similarity"] = (text_sim < 0.25).astype(float)

        # 4. Source & Locality Signals
        same_loc = df_pairs["same_locality"].values.astype(float)
        same_src = df_pairs["same_source"].values.astype(float)
        rel_prod = df_pairs["reliability_product"].values.astype(float)

        features["same_locality"] = same_loc
        features["same_source"] = same_src
        features["source_independence"] = 1.0 - same_src
        features["reliability_product"] = rel_prod

        # 5. Joined Observations Enrichment (if observations dataframe is provided)
        if active_obs is not None:
            id_a_list = df_pairs["observation_id_a"].values
            id_b_list = df_pairs["observation_id_b"].values

            obs_a = active_obs.loc[id_a_list]
            obs_b = active_obs.loc[id_b_list]

            # Source Reliabilities
            rel_a = obs_a["source_reliability_score"].fillna(0.5).values.astype(float)
            rel_b = obs_b["source_reliability_score"].fillna(0.5).values.astype(float)
            features["source_reliability_min"] = np.minimum(rel_a, rel_b)
            features["source_reliability_max"] = np.maximum(rel_a, rel_b)
            features["both_high_reliability"] = ((rel_a >= 0.70) & (rel_b >= 0.70)).astype(float)

            # Location Quality
            acc_a = obs_a["location_accuracy_m"].fillna(50.0).values.astype(float)
            acc_b = obs_b["location_accuracy_m"].fillna(50.0).values.astype(float)
            features["location_accuracy_mean"] = (acc_a + acc_b) / 2.0
            features["location_accuracy_min"] = np.minimum(acc_a, acc_b)

            # Verification Status
            ver_a = (obs_a["verification_status"].values == "VERIFIED").astype(float)
            ver_b = (obs_b["verification_status"].values == "VERIFIED").astype(float)
            features["both_verified"] = (ver_a * ver_b)

            # Ingestion Candidate Duplicate Flag
            dupe_a = obs_a["is_duplicate_candidate"].fillna(False).values.astype(bool)
            dupe_b = obs_b["is_duplicate_candidate"].fillna(False).values.astype(bool)
            features["is_duplicate_candidate_flag"] = (dupe_a | dupe_b).astype(float)

            # Category & Causality
            types_a = obs_a["event_type"].values
            types_b = obs_b["event_type"].values

            same_cat = (types_a == types_b).astype(float)
            features["same_event_type"] = same_cat

            # Causal and Compatibility Evaluator
            is_causal_list = []
            compat_score_list = []
            contra_list = []

            texts_a = obs_a["text_original"].fillna("").values
            texts_b = obs_b["text_original"].fillna("").values

            for t_a, t_b, txt_a, txt_b in zip(types_a, types_b, texts_a, texts_b):
                # Category relation
                if t_a == t_b:
                    is_causal_list.append(0.0)
                    compat_score_list.append(1.0)
                elif t_b in CAUSAL_RELATIONSHIPS.get(t_a, set()) or t_a in CAUSAL_RELATIONSHIPS.get(t_b, set()):
                    is_causal_list.append(1.0)
                    compat_score_list.append(0.75)
                else:
                    is_causal_list.append(0.0)
                    compat_score_list.append(0.40)

                # Contradiction keywords check
                has_contra, _ = detect_contradictory_text(str(txt_a), str(txt_b))
                contra_list.append(1.0 if has_contra else 0.0)

            features["is_causal_category_pair"] = np.array(is_causal_list)
            features["category_compatibility_score"] = np.array(compat_score_list)
            features["has_contradiction_keywords"] = np.array(contra_list)

        else:
            # Fallback if raw observations are omitted
            features["source_reliability_min"] = np.sqrt(rel_prod)
            features["source_reliability_max"] = np.sqrt(rel_prod)
            features["both_high_reliability"] = (rel_prod >= 0.50).astype(float)
            features["location_accuracy_mean"] = 25.0
            features["location_accuracy_min"] = 25.0
            features["both_verified"] = 0.0
            features["is_duplicate_candidate_flag"] = (text_sim >= 0.85).astype(float)
            features["same_event_type"] = (text_sim >= 0.50).astype(float)
            features["is_causal_category_pair"] = 0.0
            features["category_compatibility_score"] = 0.50
            features["has_contradiction_keywords"] = 0.0

        # 6. Composite Heuristic Affinity Prior from Phase 3
        # Combines spatial, temporal, text, and reliability as a baseline prior feature
        features["phase3_heuristic_affinity_score"] = (
            0.35 * features["spatial_proximity_decay"] +
            0.25 * features["temporal_proximity_decay"] +
            0.20 * features["category_compatibility_score"] +
            0.15 * features["text_similarity"] +
            0.05 * features["reliability_product"]
        ).round(4)

        # Enforce column order and completeness
        return features[self.FEATURE_NAMES].fillna(0.0)
