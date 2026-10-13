"""
UrbanPulse Multimodal Event Preprocessing Pipeline
Processes and standardizes observations, sensor telemetry, social media posts,
and flood observations with robust quality flagging, deduplication, and repost detection.
"""

import os
import time
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

try:
    from .config import DEFAULT_SOURCE_RELIABILITY
    from .normalizers import (
        TimestampNormalizer,
        SpatialNormalizer,
        TaxonomyNormalizer,
        TextNormalizer,
    )
    from .deduplicator import (
        ExactDuplicateDetector,
        SocialRepostDetector,
        IndependenceGuard,
    )
    from .event_schema import StandardizedEvent
except (ImportError, ValueError):
    from config import DEFAULT_SOURCE_RELIABILITY
    from normalizers import (
        TimestampNormalizer,
        SpatialNormalizer,
        TaxonomyNormalizer,
        TextNormalizer,
    )
    from deduplicator import (
        ExactDuplicateDetector,
        SocialRepostDetector,
        IndependenceGuard,
    )
    from event_schema import StandardizedEvent


class MultimodalPreprocessor:
    def __init__(self, dataset_dir: str, output_dir: str):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.exact_detector = ExactDuplicateDetector()
        self.repost_detector = SocialRepostDetector()
        self.stats: Dict[str, Any] = {}

    def process_observations(self) -> List[StandardizedEvent]:
        """
        Processes primary observations stream (observations.csv).
        """
        csv_path = os.path.join(self.dataset_dir, "observations.csv")
        df = pd.read_csv(csv_path, low_memory=False)
        events: List[StandardizedEvent] = []

        for idx, row in df.iterrows():
            raw_id = str(row["observation_id"]).strip()
            event_id = f"EVT-{raw_id}"

            # 1. Normalize timestamps
            obs_iso, obs_epoch, ts_qual = TimestampNormalizer.normalize(row.get("event_time"))
            coll_iso, _, _ = TimestampNormalizer.normalize(row.get("ingested_at"))
            has_missing_ts = (ts_qual != "VALID")

            # 2. Normalize spatial coordinates
            raw_lat = row.get("latitude")
            raw_lon = row.get("longitude")
            loc_text = row.get("location_text")
            spatial = SpatialNormalizer.validate_and_normalize(raw_lat, raw_lon, loc_text)

            # 3. Taxonomies
            canonical_type = TaxonomyNormalizer.normalize_event_type(row.get("event_type"))
            source_chan = TaxonomyNormalizer.normalize_source_type(row.get("source_type"))
            locality_norm = TaxonomyNormalizer.normalize_locality(loc_text)

            # 4. Content & Text
            raw_text = str(row.get("text_original", "")) if pd.notnull(row.get("text_original")) else ""
            cleaned_text = TextNormalizer.clean_text(raw_text)
            norm_text_match = TextNormalizer.normalize_for_matching(raw_text)
            has_missing_text = len(cleaned_text) == 0

            # 5. Reliability & Provenance
            rel_score, rel_imputed = TaxonomyNormalizer.normalize_reliability(
                row.get("source_reliability_score"), source_chan
            )
            provenance = str(row.get("provenance", "SYNTHETIC_GENERATED"))
            is_synthetic = bool(row.get("is_simulated", True))

            # 6. Deduplication Check
            # Note: in observations.csv, source_id is a channel category (e.g. 'SRC-CITIZEN_REPORT'),
            # NOT a unique external record ID. Pass source_record_id=None so exact deduplication
            # correctly triggers only on identical content fingerprints.
            is_dupe, dupe_of, dupe_type = self.exact_detector.check_and_register(
                event_id=event_id,
                source_record_id=None,
                source_type=source_chan,
                normalized_text=norm_text_match,
                locality=locality_norm,
                timestamp_iso=obs_iso,
            )
            if not is_dupe and bool(row.get("is_duplicate_candidate", False)):
                is_dupe = True
                dupe_type = "RAW_STREAM_DUPLICATE_CANDIDATE"

            # 7. Quality Score
            q_score = StandardizedEvent.calculate_quality_score(
                has_ts=not has_missing_ts,
                spatial_quality=spatial["spatial_quality"],
                has_text=not has_missing_text,
                has_reliability=not rel_imputed,
            )

            # 8. Evaluation Ground Truth (Strictly Quarantined)
            cand_inc_id = row.get("candidate_incident_id")
            eval_inc_id = str(cand_inc_id).strip() if pd.notnull(cand_inc_id) else None

            event = StandardizedEvent(
                event_id=event_id,
                raw_id=raw_id,
                source_modality="OBSERVATION",
                source_channel=source_chan,
                canonical_event_type=canonical_type,
                raw_event_type=str(row.get("event_type", "")),
                observation_time_iso=obs_iso,
                observation_time_epoch_ms=obs_epoch,
                publication_time_iso=None,  # Observations stream has event_time & ingested_at
                collection_time_iso=coll_iso,
                latitude=spatial["latitude"],
                longitude=spatial["longitude"],
                latitude_raw=raw_lat,
                longitude_raw=raw_lon,
                locality_normalized=locality_norm,
                locality_zone=spatial["locality_zone"],
                text_raw=raw_text,
                text_cleaned=cleaned_text,
                source_reliability=rel_score,
                reliability_imputed=rel_imputed,
                provenance=provenance,
                is_synthetic=is_synthetic,
                is_exact_duplicate=is_dupe,
                duplicate_of_id=dupe_of,
                duplicate_type=dupe_type,
                is_repost_or_copy=False,
                repost_parent_id=None,
                repost_similarity=None,
                has_missing_timestamp=has_missing_ts,
                has_missing_coordinates=(spatial["latitude"] is None or spatial["longitude"] is None),
                has_missing_text=has_missing_text,
                spatial_quality=spatial["spatial_quality"],
                data_quality_score=q_score,
                is_usable_for_spatial_clustering=spatial["spatial_quality"] in ("VALID_IN_CORRIDOR", "VALID_OUT_OF_BOUNDS"),
                is_usable_for_text_matching=(not has_missing_text and len(cleaned_text) > 3),
                evaluation_candidate_incident_id=eval_inc_id,
            )
            events.append(event)

        return events

    def process_social_media(self) -> List[StandardizedEvent]:
        """
        Processes social media stream (social_media_posts.csv) with repost detection.
        """
        csv_path = os.path.join(self.dataset_dir, "social_media_posts.csv")
        df = pd.read_csv(csv_path, low_memory=False)
        events: List[StandardizedEvent] = []

        # Sort chronologically by post_time for temporal repost resolution
        df_sorted = df.sort_values(by="post_time", na_position="last").reset_index(drop=True)

        for idx, row in df_sorted.iterrows():
            raw_id = str(row["post_id"]).strip()
            event_id = f"EVT-{raw_id}"

            # 1. Timestamps
            pub_iso, pub_epoch, ts_qual = TimestampNormalizer.normalize(row.get("post_time"))
            coll_iso, _, _ = TimestampNormalizer.normalize(row.get("collected_at"))
            has_missing_ts = (ts_qual != "VALID")

            # 2. Spatial coordinates
            raw_lat = row.get("latitude")
            raw_lon = row.get("longitude")
            loc_text = row.get("location_text")
            spatial = SpatialNormalizer.validate_and_normalize(raw_lat, raw_lon, loc_text)

            # 3. Taxonomies
            raw_text = str(row.get("text_original", "")) if pd.notnull(row.get("text_original")) else ""
            cleaned_text = TextNormalizer.clean_text(raw_text)
            has_missing_text = len(cleaned_text) == 0

            # Infer canonical event type from keyword heuristics
            canonical_type = "GENERAL_INCIDENT"
            lower_text = cleaned_text.lower()
            if any(k in lower_text for k in ["waterlog", "flood", "water rise", "water level"]):
                canonical_type = "ROAD_WATERLOGGING"
            elif any(k in lower_text for k in ["subway", "underpass"]):
                canonical_type = "SUBWAY_INUNDATION"
            elif any(k in lower_text for k in ["manhole", "overflow"]):
                canonical_type = "MANHOLE_OVERFLOW"
            elif any(k in lower_text for k in ["drain", "blocked"]):
                canonical_type = "DRAIN_BLOCKAGE"
            elif any(k in lower_text for k in ["traffic", "jam", "slowdown"]):
                canonical_type = "TRAFFIC_SLOWDOWN"
            elif any(k in lower_text for k in ["accident", "crash"]):
                canonical_type = "ROAD_ACCIDENT"
            elif any(k in lower_text for k in ["rain", "downpour"]):
                canonical_type = "HEAVY_RAINFALL"
            elif any(k in lower_text for k in ["ohe", "overhead wire"]):
                canonical_type = "OHE_WIRE_FAILURE"
            elif any(k in lower_text for k in ["tree", "fallen"]):
                canonical_type = "TREE_FALL"

            locality_norm = TaxonomyNormalizer.normalize_locality(loc_text)
            platform = str(row.get("platform", "SOCIAL_MEDIA")).upper()

            # 4. Reliability & Provenance
            rel_score, rel_imputed = TaxonomyNormalizer.normalize_reliability(None, "SOCIAL_MEDIA")
            provenance = str(row.get("provenance", "SYNTHETIC_GENERATED"))
            is_synthetic = bool(row.get("is_synthetic", True))

            # 5. Exact duplicate detection via platform_post_id
            plat_post_id = str(row.get("platform_post_id", ""))
            norm_match = TextNormalizer.normalize_for_matching(raw_text)
            is_dupe, dupe_of, dupe_type = self.exact_detector.check_and_register(
                event_id=event_id,
                source_record_id=plat_post_id,
                source_type="SOCIAL_MEDIA",
                normalized_text=norm_match,
                locality=locality_norm,
                timestamp_iso=pub_iso,
            )

            # 6. Repost / Copied Content Detection
            is_repost, repost_parent, repost_sim = self.repost_detector.check_and_register(
                post_id=raw_id,
                raw_text=raw_text,
                epoch_ms=pub_epoch,
            )

            # 7. Quality Score
            q_score = StandardizedEvent.calculate_quality_score(
                has_ts=not has_missing_ts,
                spatial_quality=spatial["spatial_quality"],
                has_text=not has_missing_text,
                has_reliability=not rel_imputed,
            )

            event = StandardizedEvent(
                event_id=event_id,
                raw_id=raw_id,
                source_modality="SOCIAL_POST",
                source_channel=platform,
                canonical_event_type=canonical_type,
                raw_event_type="SOCIAL_POST",
                observation_time_iso=pub_iso,  # Social observation is when user posted
                observation_time_epoch_ms=pub_epoch,
                publication_time_iso=pub_iso,
                collection_time_iso=coll_iso,
                latitude=spatial["latitude"],
                longitude=spatial["longitude"],
                latitude_raw=raw_lat,
                longitude_raw=raw_lon,
                locality_normalized=locality_norm,
                locality_zone=spatial["locality_zone"],
                text_raw=raw_text,
                text_cleaned=cleaned_text,
                source_reliability=rel_score,
                reliability_imputed=rel_imputed,
                provenance=provenance,
                is_synthetic=is_synthetic,
                is_exact_duplicate=is_dupe,
                duplicate_of_id=dupe_of,
                duplicate_type=dupe_type,
                is_repost_or_copy=is_repost,
                repost_parent_id=repost_parent,
                repost_similarity=repost_sim,
                has_missing_timestamp=has_missing_ts,
                has_missing_coordinates=(spatial["latitude"] is None or spatial["longitude"] is None),
                has_missing_text=has_missing_text,
                spatial_quality=spatial["spatial_quality"],
                data_quality_score=q_score,
                is_usable_for_spatial_clustering=spatial["spatial_quality"] in ("VALID_IN_CORRIDOR", "VALID_OUT_OF_BOUNDS"),
                is_usable_for_text_matching=(not has_missing_text and len(cleaned_text) > 3),
                evaluation_candidate_incident_id=None,
            )
            events.append(event)

        return events

    def process_sensor_readings(self) -> List[StandardizedEvent]:
        """
        Processes physical sensor telemetry (sensor_readings.csv).
        """
        csv_path = os.path.join(self.dataset_dir, "sensor_readings.csv")
        df = pd.read_csv(csv_path, low_memory=False)
        events: List[StandardizedEvent] = []

        for idx, row in df.iterrows():
            raw_id = str(row["reading_id"]).strip()
            event_id = f"EVT-{raw_id}"

            # 1. Timestamps
            obs_iso, obs_epoch, ts_qual = TimestampNormalizer.normalize(row.get("timestamp"))
            has_missing_ts = (ts_qual != "VALID")

            # 2. Coordinates
            raw_lat = row.get("latitude")
            raw_lon = row.get("longitude")
            spatial = SpatialNormalizer.validate_and_normalize(raw_lat, raw_lon)

            # 3. Categorization based on sensor measurements
            canonical_type = "SENSOR_TELEMETRY"
            water_level = row.get("water_level_cm", 0.0)
            rainfall = row.get("rainfall_15min_mm", 0.0)
            blockage_prob = row.get("blockage_probability", 0.0)

            if water_level > 40.0:
                canonical_type = "WATER_LEVEL_RISE"
            elif rainfall > 25.0:
                canonical_type = "HEAVY_RAINFALL"
            elif blockage_prob > 0.70:
                canonical_type = "DRAIN_BLOCKAGE"
            elif str(row.get("pump_status")) == "FAILED":
                canonical_type = "PUMP_FAILURE"

            # 4. Generate structured telemetry descriptive text
            sensor_id = str(row.get("sensor_id", "SENSOR"))
            telemetry_text = (
                f"Telemetry reading from {sensor_id}: rainfall_15min={rainfall}mm, "
                f"water_level={water_level}cm, pump_status={row.get('pump_status')}, "
                f"health={row.get('sensor_health')}"
            )

            # 5. Reliability
            health = str(row.get("sensor_health", "HEALTHY")).upper()
            rel_score = 0.90 if health == "HEALTHY" else (0.60 if health == "DEGRADED" else 0.20)
            provenance = str(row.get("provenance", "SYNTHETIC_GENERATED"))
            is_synthetic = bool(row.get("is_synthetic", True))

            # 6. Quality Score
            q_score = StandardizedEvent.calculate_quality_score(
                has_ts=not has_missing_ts,
                spatial_quality=spatial["spatial_quality"],
                has_text=True,
                has_reliability=True,
            )

            event = StandardizedEvent(
                event_id=event_id,
                raw_id=raw_id,
                source_modality="SENSOR_TELEMETRY",
                source_channel="SENSOR",
                canonical_event_type=canonical_type,
                raw_event_type="SENSOR_TELEMETRY",
                observation_time_iso=obs_iso,
                observation_time_epoch_ms=obs_epoch,
                publication_time_iso=obs_iso,
                collection_time_iso=obs_iso,
                latitude=spatial["latitude"],
                longitude=spatial["longitude"],
                latitude_raw=raw_lat,
                longitude_raw=raw_lon,
                locality_normalized="Corridor Infrastructure",
                locality_zone=spatial["locality_zone"],
                text_raw=telemetry_text,
                text_cleaned=telemetry_text,
                source_reliability=rel_score,
                reliability_imputed=False,
                provenance=provenance,
                is_synthetic=is_synthetic,
                is_exact_duplicate=False,
                duplicate_of_id=None,
                duplicate_type=None,
                is_repost_or_copy=False,
                repost_parent_id=None,
                repost_similarity=None,
                has_missing_timestamp=has_missing_ts,
                has_missing_coordinates=(spatial["latitude"] is None or spatial["longitude"] is None),
                has_missing_text=False,
                spatial_quality=spatial["spatial_quality"],
                data_quality_score=q_score,
                is_usable_for_spatial_clustering=spatial["spatial_quality"] in ("VALID_IN_CORRIDOR", "VALID_OUT_OF_BOUNDS"),
                is_usable_for_text_matching=True,
                evaluation_candidate_incident_id=None,
            )
            events.append(event)

        return events

    def process_flood_observations(self) -> List[StandardizedEvent]:
        """
        Processes flood incident observations stream (flood_incident_observations.csv).
        """
        csv_path = os.path.join(self.dataset_dir, "flood_incident_observations.csv")
        df = pd.read_csv(csv_path, low_memory=False)
        events: List[StandardizedEvent] = []

        for idx, row in df.iterrows():
            raw_id = str(row["flood_observation_id"]).strip()
            event_id = f"EVT-{raw_id}"

            # 1. Timestamps
            obs_iso, obs_epoch, ts_qual = TimestampNormalizer.normalize(row.get("observed_time"))
            pub_iso, _, _ = TimestampNormalizer.normalize(row.get("reported_time"))
            has_missing_ts = (ts_qual != "VALID")

            # 2. Coordinates
            raw_lat = row.get("latitude")
            raw_lon = row.get("longitude")
            loc_text = row.get("locality")
            spatial = SpatialNormalizer.validate_and_normalize(raw_lat, raw_lon, loc_text)

            # 3. Taxonomies
            canonical_type = TaxonomyNormalizer.normalize_event_type(row.get("event_type"))
            locality_norm = TaxonomyNormalizer.normalize_locality(loc_text)

            # 4. Content Text
            depth_cm = row.get("water_depth_cm")
            depth_str = f"{depth_cm}cm" if pd.notnull(depth_cm) else "unmeasured depth"
            passability = row.get("road_passability", "UNKNOWN")
            flood_text = (
                f"Field flood report in {locality_norm}: {canonical_type} with {depth_str}, "
                f"road passability is {passability}, blockage: {row.get('blockage_status')}"
            )

            # 5. Reliability & Provenance
            rel_score, rel_imputed = TaxonomyNormalizer.normalize_reliability(0.85, "CITIZEN_REPORT")
            provenance = str(row.get("provenance", "SYNTHETIC_GENERATED"))
            is_synthetic = bool(row.get("is_synthetic", True))

            # 6. Quality Score
            q_score = StandardizedEvent.calculate_quality_score(
                has_ts=not has_missing_ts,
                spatial_quality=spatial["spatial_quality"],
                has_text=True,
                has_reliability=True,
            )

            event = StandardizedEvent(
                event_id=event_id,
                raw_id=raw_id,
                source_modality="FLOOD_REPORT",
                source_channel="FIELD_OBSERVATION",
                canonical_event_type=canonical_type,
                raw_event_type=str(row.get("event_type", "")),
                observation_time_iso=obs_iso,
                observation_time_epoch_ms=obs_epoch,
                publication_time_iso=pub_iso,
                collection_time_iso=pub_iso,
                latitude=spatial["latitude"],
                longitude=spatial["longitude"],
                latitude_raw=raw_lat,
                longitude_raw=raw_lon,
                locality_normalized=locality_norm,
                locality_zone=spatial["locality_zone"],
                text_raw=flood_text,
                text_cleaned=flood_text,
                source_reliability=rel_score,
                reliability_imputed=False,
                provenance=provenance,
                is_synthetic=is_synthetic,
                is_exact_duplicate=False,
                duplicate_of_id=None,
                duplicate_type=None,
                is_repost_or_copy=False,
                repost_parent_id=None,
                repost_similarity=None,
                has_missing_timestamp=has_missing_ts,
                has_missing_coordinates=(spatial["latitude"] is None or spatial["longitude"] is None),
                has_missing_text=False,
                spatial_quality=spatial["spatial_quality"],
                data_quality_score=q_score,
                is_usable_for_spatial_clustering=spatial["spatial_quality"] in ("VALID_IN_CORRIDOR", "VALID_OUT_OF_BOUNDS"),
                is_usable_for_text_matching=True,
                evaluation_candidate_incident_id=None,
            )
            events.append(event)

        return events

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Executes end-to-end preprocessing, saves standardized datasets, and computes audit metrics.
        """
        start_time = time.time()
        print("[*] Starting Multimodal Event Preprocessing Pipeline...")

        print(" -> Processing unified observations stream (observations.csv)...")
        obs_events = self.process_observations()
        print(f"    Done: {len(obs_events)} events.")

        print(" -> Processing social media stream with repost detector (social_media_posts.csv)...")
        soc_events = self.process_social_media()
        print(f"    Done: {len(soc_events)} events.")

        print(" -> Processing sensor telemetry stream (sensor_readings.csv)...")
        sen_events = self.process_sensor_readings()
        print(f"    Done: {len(sen_events)} events.")

        print(" -> Processing flood observations stream (flood_incident_observations.csv)...")
        fld_events = self.process_flood_observations()
        print(f"    Done: {len(fld_events)} events.")

        all_events = obs_events + soc_events + sen_events + fld_events
        print(f"[+] Total multimodal events processed: {len(all_events)}")

        # Convert to DataFrames and save
        def save_events_to_csv(ev_list: List[StandardizedEvent], filename: str):
            records = [e.to_dict() for e in ev_list]
            df_out = pd.DataFrame(records)
            out_path = os.path.join(self.output_dir, filename)
            df_out.to_csv(out_path, index=False)
            print(f" [+] Saved: {out_path} ({len(df_out)} rows)")
            return df_out

        df_obs = save_events_to_csv(obs_events, "standardized_observations.csv")
        df_soc = save_events_to_csv(soc_events, "standardized_social_posts.csv")
        df_sen = save_events_to_csv(sen_events, "standardized_sensor_readings.csv")
        df_fld = save_events_to_csv(fld_events, "standardized_flood_observations.csv")
        df_all = save_events_to_csv(all_events, "unified_multimodal_events.csv")

        # Compile Quality and Deduplication Statistics
        exact_dupes_total = int(df_all["is_exact_duplicate"].sum())
        reposts_total = int(df_all["is_repost_or_copy"].sum())
        missing_coords_total = int(df_all["has_missing_coordinates"].sum())
        missing_ts_total = int(df_all["has_missing_timestamp"].sum())
        missing_text_total = int(df_all["has_missing_text"].sum())
        
        # Quality score breakdown
        q_scores = df_all["data_quality_score"]
        
        runtime_sec = round(time.time() - start_time, 2)
        print(f"[+] Preprocessing pipeline finished in {runtime_sec}s.")

        pipeline_metrics = {
            "total_events_processed": len(all_events),
            "stream_counts": {
                "observations": len(obs_events),
                "social_media_posts": len(soc_events),
                "sensor_readings": len(sen_events),
                "flood_incident_observations": len(fld_events),
            },
            "deduplication": {
                "exact_duplicates_flagged": exact_dupes_total,
                "social_reposts_or_copies_flagged": reposts_total,
                "reposts_pct_in_social": round(reposts_total / len(soc_events) * 100, 2),
            },
            "data_quality_flags": {
                "missing_timestamps_flagged": missing_ts_total,
                "missing_coordinates_flagged": missing_coords_total,
                "missing_coordinates_pct": round(missing_coords_total / len(all_events) * 100, 2),
                "missing_text_flagged": missing_text_total,
                "usable_for_spatial_clustering": int(df_all["is_usable_for_spatial_clustering"].sum()),
                "usable_for_text_matching": int(df_all["is_usable_for_text_matching"].sum()),
            },
            "quality_score_summary": {
                "mean": round(float(q_scores.mean()), 4),
                "median": round(float(q_scores.median()), 4),
                "min": round(float(q_scores.min()), 4),
                "max": round(float(q_scores.max()), 4),
            },
            "runtime_seconds": runtime_sec,
        }

        # Save metrics JSON
        metrics_path = os.path.join(self.output_dir, "preprocessing_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_metrics, f, indent=2)

        self.stats = pipeline_metrics
        return pipeline_metrics


if __name__ == "__main__":
    base_dir = r"d:\Chaitrali\SEM 7\Urban flow\CodeAstra\UrbanPulse_Expanded_Dataset"
    out_dir = os.path.join(base_dir, "cleaned_output")
    preprocessor = MultimodalPreprocessor(dataset_dir=base_dir, output_dir=out_dir)
    results = preprocessor.run_full_pipeline()
    print("\n--- Preprocessing Metrics Summary ---")
    print(json.dumps(results, indent=2))
