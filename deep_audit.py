"""
Deep audit script to inspect entity relations, source IDs, artificial patterns,
label correlations, missing references, and split designs.
"""

import os
import json
import pandas as pd
import numpy as np

DATASET_DIR = r"d:\Chaitrali\SEM 7\Urban flow\CodeAstra\UrbanPulse_Expanded_Dataset"

def run_deep_audit():
    # Load all files
    dfs = {}
    for f in os.listdir(DATASET_DIR):
        if f.endswith(".csv"):
            dfs[f] = pd.read_csv(os.path.join(DATASET_DIR, f), low_memory=False)

    results = {}

    # 1. Inspect observations.csv source_id patterns
    obs = dfs["observations.csv"]
    print("=== OBSERVATIONS SOURCE_ID AUDIT ===")
    src_audit = {}
    for stype, grp in obs.groupby("source_type"):
        sample_ids = grp["source_id"].head(5).tolist()
        src_audit[stype] = {
            "count": len(grp),
            "sample_source_ids": sample_ids,
            "id_prefix": list(set(grp["source_id"].str.split("-").str[0]))
        }
    results["observations_source_id_breakdown"] = src_audit

    # Check cross-referencing of source_id
    # Does source_id match social_media_posts post_id?
    soc_posts = dfs["social_media_posts.csv"]
    soc_obs = obs[obs["source_type"] == "SOCIAL_MEDIA"]
    soc_matches = soc_obs["source_id"].isin(soc_posts["post_id"])
    soc_plat_matches = soc_obs["source_id"].isin(soc_posts["platform_post_id"])
    results["social_source_id_cross_ref"] = {
        "social_obs_count": len(soc_obs),
        "post_id_matches": int(soc_matches.sum()),
        "platform_post_id_matches": int(soc_plat_matches.sum()),
        "sample_soc_obs_source_ids": soc_obs["source_id"].head(5).tolist(),
        "sample_social_posts_post_ids": soc_posts["post_id"].head(5).tolist(),
        "sample_social_posts_platform_ids": soc_posts["platform_post_id"].head(5).tolist(),
    }

    # Does source_id match sensor_readings reading_id or sensor_id?
    sensors = dfs["sensor_readings.csv"]
    sensor_obs = obs[obs["source_type"] == "SENSOR"]
    sen_reading_matches = sensor_obs["source_id"].isin(sensors["reading_id"])
    sen_sensor_matches = sensor_obs["source_id"].isin(sensors["sensor_id"])
    results["sensor_source_id_cross_ref"] = {
        "sensor_obs_count": len(sensor_obs),
        "reading_id_matches": int(sen_reading_matches.sum()),
        "sensor_id_matches": int(sen_sensor_matches.sum()),
        "sample_sensor_obs_source_ids": sensor_obs["source_id"].head(5).tolist(),
        "sample_sensor_readings_reading_ids": sensors["reading_id"].head(5).tolist(),
        "sample_sensor_readings_sensor_ids": sensors["sensor_id"].head(5).tolist()
    }

    # 2. Inspect infrastructure_assets asset_type
    assets = dfs["infrastructure_assets.csv"]
    results["asset_types"] = assets["asset_type"].value_counts().to_dict()
    # Check if sensor_id in sensor_readings match asset_id in assets
    sensor_ids_in_readings = set(sensors["sensor_id"])
    asset_ids = set(assets["asset_id"])
    results["sensor_to_asset_id_check"] = {
        "unique_sensors_in_readings": len(sensor_ids_in_readings),
        "sensor_ids_found_in_assets": len(sensor_ids_in_readings.intersection(asset_ids)),
        "sensor_ids_missing_in_assets": len(sensor_ids_in_readings - asset_ids),
        "asset_types_matching_sensor_ids": assets[assets["asset_id"].isin(sensor_ids_in_readings)]["asset_type"].value_counts().to_dict()
    }

    # 3. Graph edges entity check
    edges = dfs["relationship_evidence_graph_edges.csv"]
    obs_ids = set(obs["observation_id"])
    inc_ids = set(dfs["incident_clusters.csv"]["incident_id"])
    soc_ids = set(soc_posts["post_id"])
    route_ids = set(dfs["routes.csv"]["route_id"])
    
    # Check by entity type
    source_stats = {}
    for stype, grp in edges.groupby("source_entity_type"):
        ids = set(grp["source_entity_id"])
        if stype == "OBSERVATION":
            valid = ids.intersection(obs_ids)
        elif stype == "INCIDENT":
            valid = ids.intersection(inc_ids)
        elif stype == "ASSET":
            valid = ids.intersection(asset_ids)
        elif stype == "SOCIAL_POST":
            valid = ids.intersection(soc_ids)
        elif stype == "ROUTE":
            valid = ids.intersection(route_ids)
        else:
            valid = set()
        source_stats[stype] = {
            "total_records": len(grp),
            "unique_ids": len(ids),
            "valid_ids_found": len(valid),
            "missing_ids_count": len(ids - valid),
            "sample_missing": list(ids - valid)[:5] if len(ids - valid) > 0 else []
        }
    results["graph_edges_source_entity_validation"] = source_stats

    target_stats = {}
    for ttype, grp in edges.groupby("target_entity_type"):
        ids = set(grp["target_entity_id"])
        if ttype == "OBSERVATION":
            valid = ids.intersection(obs_ids)
        elif ttype == "INCIDENT":
            valid = ids.intersection(inc_ids)
        elif ttype == "ASSET":
            valid = ids.intersection(asset_ids)
        elif ttype == "SOCIAL_POST":
            valid = ids.intersection(soc_ids)
        elif ttype == "ROUTE":
            valid = ids.intersection(route_ids)
        elif ttype == "LOCATION":
            valid = set() # Location is not in a dedicated table
        else:
            valid = set()
        target_stats[ttype] = {
            "total_records": len(grp),
            "unique_ids": len(ids),
            "valid_ids_found": len(valid),
            "missing_ids_count": len(ids - valid),
            "sample_missing": list(ids - valid)[:5] if len(ids - valid) > 0 else []
        }
    results["graph_edges_target_entity_validation"] = target_stats

    # Check evidence_ids in graph edges
    evidence_obs_missing = 0
    total_evidences = 0
    for ev_str in edges["evidence_ids"].dropna():
        ev_tokens = [t.strip() for t in ev_str.split(";")]
        for ev in ev_tokens:
            total_evidences += 1
            if ev not in obs_ids:
                evidence_obs_missing += 1
    results["graph_evidence_ids_validation"] = {
        "total_evidence_references": total_evidences,
        "missing_evidence_observations": evidence_obs_missing
    }

    # 4. Route impacts segments check
    ri = dfs["route_impacts.csv"]
    results["route_impacts_segments"] = {
        "unique_segment_ids": ri["segment_id"].nunique(),
        "sample_segments": ri["segment_id"].head(5).tolist(),
        "is_segment_table_present": "segments.csv" in dfs or "route_network.csv" in dfs
    }

    # 5. Flood observations check
    flood = dfs["flood_incident_observations.csv"]
    flood_assets = set(flood["asset_id"])
    results["flood_asset_ids_in_infrastructure"] = {
        "unique_flood_asset_ids": len(flood_assets),
        "found_in_assets": len(flood_assets.intersection(asset_ids)),
        "missing_in_assets": len(flood_assets - asset_ids)
    }

    # 6. Pairwise training relationships vs label distributions
    pairs = dfs["pairwise_relationship_training.csv"]
    results["pairwise_stats_by_label"] = {}
    for lbl, grp in pairs.groupby("relation_label"):
        results["pairwise_stats_by_label"][lbl] = {
            "count": len(grp),
            "pct": round(len(grp) / len(pairs) * 100, 2),
            "mean_geo_dist_km": round(grp["geo_distance_km"].mean(), 4),
            "mean_time_diff_min": round(grp["time_diff_minutes"].mean(), 4),
            "mean_text_similarity": round(grp["text_similarity"].mean(), 4),
            "pct_same_locality": round(grp["same_locality"].mean() * 100, 2),
            "pct_same_source": round(grp["same_source"].mean() * 100, 2),
            "mean_reliability_prod": round(grp["reliability_product"].mean(), 4)
        }

    # 7. Check for exact duplicate pairs in pairwise training (e.g. A-B and B-A)
    pair_tuples = set()
    symmetric_dupes = 0
    self_pairs = 0
    for _, row in pairs.iterrows():
        a, b = row["observation_id_a"], row["observation_id_b"]
        if a == b:
            self_pairs += 1
        sorted_pair = tuple(sorted([a, b]))
        if sorted_pair in pair_tuples:
            symmetric_dupes += 1
        pair_tuples.add(sorted_pair)
    results["pairwise_pair_duplication"] = {
        "total_pairs": len(pairs),
        "unique_unordered_pairs": len(pair_tuples),
        "symmetric_duplicates": symmetric_dupes,
        "self_pairs": self_pairs
    }

    # 8. Check observation duplicate candidate flags vs actual text duplicates
    obs_text_dupes = obs["text_original"].duplicated().sum()
    obs_norm_dupes = obs["text_normalized"].duplicated().sum()
    obs_flagged_dupes = obs["is_duplicate_candidate"].sum()
    results["observation_text_duplicates"] = {
        "raw_text_duplicates": int(obs_text_dupes),
        "normalized_text_duplicates": int(obs_norm_dupes),
        "flagged_duplicate_candidate": int(obs_flagged_dupes),
        "flagged_match_raw_text_dupes": int(((obs["text_original"].duplicated(keep=False)) & obs["is_duplicate_candidate"]).sum())
    }

    # 9. Check temporal ranges across datasets
    results["temporal_ranges"] = {
        "observations_event_time": {"min": str(obs["event_time"].min()), "max": str(obs["event_time"].max())},
        "incidents_first_seen": {"min": str(dfs["incident_clusters.csv"]["first_seen"].min()), "max": str(dfs["incident_clusters.csv"]["last_seen"].max())},
        "sensor_timestamps": {"min": str(sensors["timestamp"].min()), "max": str(sensors["timestamp"].max())},
        "social_post_times": {"min": str(soc_posts["post_time"].min()), "max": str(soc_posts["post_time"].max())},
        "flood_observed_times": {"min": str(flood["observed_time"].min()), "max": str(flood["observed_time"].max())},
        "route_impacts_valid_from": {"min": str(ri["valid_from"].min()), "max": str(ri["valid_to"].max())}
    }

    # Save results
    out_path = os.path.join(DATASET_DIR, "deep_audit_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved deep audit results to:", out_path)

if __name__ == "__main__":
    run_deep_audit()
