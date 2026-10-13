"""
Script to generate data_dictionary.csv and verify spatial coordinates & distributions.
"""

import os
import glob
import pandas as pd
import numpy as np

DATASET_DIR = r"d:\Chaitrali\SEM 7\Urban flow\CodeAstra\UrbanPulse_Expanded_Dataset"

COLUMN_DESCRIPTIONS = {
    # observations.csv
    ("observations.csv", "observation_id"): ("Unique identifier for each multimodal observation", "Primary Key"),
    ("observations.csv", "source_id"): ("Identifier of the originating source channel/feed", "Foreign Key / Source Ref"),
    ("observations.csv", "source_type"): ("Modality/channel category (e.g. SENSOR, SOCIAL_MEDIA, CITIZEN_REPORT)", "Inference Feature"),
    ("observations.csv", "event_time"): ("Timestamp when the incident or telemetry event occurred", "Inference Feature"),
    ("observations.csv", "ingested_at"): ("Timestamp when the observation was ingested into the pipeline", "System Metadata"),
    ("observations.csv", "event_type"): ("Observed event category (e.g. WATERLOGGING, ACCIDENT, PUMP_FAILURE)", "Inference Feature / Label"),
    ("observations.csv", "text_original"): ("Raw unnormalized text description or report narrative", "Inference Feature (NLP)"),
    ("observations.csv", "text_normalized"): ("Cleaned, normalized text representation for embedding/matching", "Inference Feature (NLP)"),
    ("observations.csv", "latitude"): ("Geographic latitude in decimal degrees (WGS84)", "Inference Feature (Spatial)"),
    ("observations.csv", "longitude"): ("Geographic longitude in decimal degrees (WGS84)", "Inference Feature (Spatial)"),
    ("observations.csv", "location_text"): ("Locality or landmark name in Mumbai Western Suburbs", "Inference Feature"),
    ("observations.csv", "location_accuracy_m"): ("Reported spatial accuracy in meters", "Inference Feature"),
    ("observations.csv", "source_reliability_score"): ("Heuristic trust score of the reporting source [0.0 - 1.0]", "Inference Feature"),
    ("observations.csv", "verification_status"): ("Verification lifecycle state (e.g. VERIFIED, UNVERIFIED)", "Evaluation / Filter"),
    ("observations.csv", "is_duplicate_candidate"): ("Flag indicating potential duplicate observation", "Inference Feature / Label"),
    ("observations.csv", "is_simulated"): ("Boolean flag confirming synthetic simulation provenance", "Provenance Metadata"),
    ("observations.csv", "provenance"): ("Provenance label: SYNTHETIC_GENERATED", "Provenance Metadata"),
    ("observations.csv", "candidate_incident_id"): ("Ground-truth / candidate linked incident cluster ID (LEAKAGE HAZARD)", "Ground-Truth Label (EVAL ONLY)"),

    # incident_clusters.csv
    ("incident_clusters.csv", "incident_id"): ("Unique canonical incident cluster identifier", "Primary Key"),
    ("incident_clusters.csv", "incident_type"): ("Primary incident category", "Cluster Target / Label"),
    ("incident_clusters.csv", "locality"): ("Suburban locality center (e.g. Andheri West, Borivali)", "Spatial Metadata"),
    ("incident_clusters.csv", "latitude"): ("Cluster centroid latitude", "Spatial Feature / Centroid"),
    ("incident_clusters.csv", "longitude"): ("Cluster centroid longitude", "Spatial Feature / Centroid"),
    ("incident_clusters.csv", "first_seen"): ("Earliest observation timestamp contributing to incident", "Temporal Metadata"),
    ("incident_clusters.csv", "last_seen"): ("Latest observation timestamp contributing to incident", "Temporal Metadata"),
    ("incident_clusters.csv", "severity"): ("Categorical incident severity (LOW, MODERATE, MAJOR, CRITICAL)", "Cluster Target / Label"),
    ("incident_clusters.csv", "cluster_status"): ("Incident status (INVESTIGATING, RESOLVED, CANDIDATE, DISPUTED, DETECTED)", "Status Metadata"),
    ("incident_clusters.csv", "is_synthetic"): ("Boolean flag indicating synthetic record", "Provenance Metadata"),
    ("incident_clusters.csv", "provenance"): ("Provenance label: SYNTHETIC_GENERATED", "Provenance Metadata"),

    # observation_incident_links.csv
    ("observation_incident_links.csv", "observation_id"): ("Referenced observation ID", "Foreign Key"),
    ("observation_incident_links.csv", "incident_id"): ("Referenced candidate incident cluster ID", "Foreign Key"),
    ("observation_incident_links.csv", "membership_score"): ("Candidate cluster membership probability / confidence [0.0 - 1.0]", "Candidate Feature / Score"),
    ("observation_incident_links.csv", "decision_type"): ("Clustering decision source (MANUAL_SIMULATED, MODEL_CANDIDATE, RULE_CANDIDATE)", "Method Metadata"),
    ("observation_incident_links.csv", "is_ground_truth"): ("Verification flag indicating confirmed association (all false)", "Evaluation Ground Truth"),
    ("observation_incident_links.csv", "provenance"): ("Provenance label: SYNTHETIC_GENERATED", "Provenance Metadata"),

    # infrastructure_assets.csv
    ("infrastructure_assets.csv", "asset_id"): ("Unique infrastructure asset identifier", "Primary Key"),
    ("infrastructure_assets.csv", "asset_type"): ("Asset class (e.g. DRAIN, PUMP_STATION, RAIN_GAUGE, ROAD_SEGMENT)", "Infrastructure Feature"),
    ("infrastructure_assets.csv", "asset_name"): ("Human-readable name or asset code", "Descriptor Metadata"),
    ("infrastructure_assets.csv", "locality"): ("Suburban locality name", "Spatial Metadata"),
    ("infrastructure_assets.csv", "latitude"): ("Asset installation latitude", "Spatial Coordinate"),
    ("infrastructure_assets.csv", "longitude"): ("Asset installation longitude", "Spatial Coordinate"),
    ("infrastructure_assets.csv", "catchment_id"): ("Hydrological catchment basin identifier", "Hydrology Feature"),
    ("infrastructure_assets.csv", "design_capacity_lps"): ("Design drainage capacity in liters per second", "Engineering Feature"),
    ("infrastructure_assets.csv", "elevation_m"): ("Elevation above mean sea level in meters", "Topographic Feature"),
    ("infrastructure_assets.csv", "asset_status"): ("Operational status of the infrastructure asset", "Status Feature"),
    ("infrastructure_assets.csv", "data_source"): ("Synthetic source generation tag", "Provenance Metadata"),
    ("infrastructure_assets.csv", "is_synthetic"): ("Boolean synthetic flag", "Provenance Metadata"),

    # asset_dependencies.csv
    ("asset_dependencies.csv", "dependency_id"): ("Unique dependency relationship identifier", "Primary Key"),
    ("asset_dependencies.csv", "from_asset_id"): ("Upstream / source asset identifier", "Foreign Key"),
    ("asset_dependencies.csv", "to_asset_id"): ("Downstream / dependent asset identifier", "Foreign Key"),
    ("asset_dependencies.csv", "relationship_type"): ("Graph dependency type (e.g. DRAINS_TO, POWERS, MONITORS)", "Graph Edge Attribute"),
    ("asset_dependencies.csv", "relationship_status"): ("Verification state of dependency edge", "Graph Edge Attribute"),
    ("asset_dependencies.csv", "evidence_id"): ("Rule or telemetry evidence supporting dependency edge", "Graph Rationale"),
    ("asset_dependencies.csv", "confidence_score"): ("Confidence in dependency link [0.0 - 1.0]", "Graph Edge Weight"),
    ("asset_dependencies.csv", "provenance"): ("Provenance label: SYNTHETIC_GENERATED", "Provenance Metadata"),

    # sensor_readings.csv
    ("sensor_readings.csv", "reading_id"): ("Unique telemetry sensor reading identifier", "Primary Key"),
    ("sensor_readings.csv", "sensor_id"): ("Sensor asset ID matching infrastructure_assets.asset_id", "Foreign Key"),
    ("sensor_readings.csv", "timestamp"): ("Telemetry measurement timestamp", "Temporal Feature"),
    ("sensor_readings.csv", "latitude"): ("Sensor latitude coordinate", "Spatial Coordinate"),
    ("sensor_readings.csv", "longitude"): ("Sensor longitude coordinate", "Spatial Coordinate"),
    ("sensor_readings.csv", "rainfall_15min_mm"): ("Rainfall accumulated over past 15 minutes (mm)", "Telemetry Feature"),
    ("sensor_readings.csv", "rainfall_1hr_mm"): ("Rainfall accumulated over past 1 hour (mm)", "Telemetry Feature"),
    ("sensor_readings.csv", "water_level_cm"): ("Measured water level in storm drain/canal (cm)", "Telemetry Feature"),
    ("sensor_readings.csv", "water_level_change_cm"): ("Rate of water level change over previous interval (cm)", "Telemetry Feature"),
    ("sensor_readings.csv", "flow_rate_lps"): ("Calculated stormwater flow rate (liters per second)", "Telemetry Feature"),
    ("sensor_readings.csv", "drainage_capacity_pct"): ("Percentage of design capacity currently utilized", "Telemetry Feature"),
    ("sensor_readings.csv", "blockage_probability"): ("Estimated likelihood of physical drain blockage [0.0 - 1.0]", "Anomaly Target / Feature"),
    ("sensor_readings.csv", "pump_status"): ("Status of adjacent pump (RUNNING, OFFLINE, FAILED, NOT_APPLICABLE)", "Telemetry State"),
    ("sensor_readings.csv", "sensor_health"): ("Sensor hardware operational health (HEALTHY, DEGRADED, OFFLINE)", "Health Metadata"),
    ("sensor_readings.csv", "data_quality"): ("Telemetry quality indicator (GOOD, SUSPECT, INVALID)", "Quality Indicator"),
    ("sensor_readings.csv", "is_synthetic"): ("Boolean synthetic flag", "Provenance Metadata"),
    ("sensor_readings.csv", "provenance"): ("Provenance label: SYNTHETIC_GENERATED", "Provenance Metadata"),

    # social_media_posts.csv
    ("social_media_posts.csv", "post_id"): ("Unique synthetic social media post identifier", "Primary Key"),
    ("social_media_posts.csv", "platform"): ("Simulated social platform name (e.g. X, CivicFeed)", "Source Feature"),
    ("social_media_posts.csv", "platform_post_id"): ("Simulated platform-native post identifier", "Platform ID"),
    ("social_media_posts.csv", "post_time"): ("Post timestamp", "Temporal Feature"),
    ("social_media_posts.csv", "collected_at"): ("Ingestion timestamp", "System Metadata"),
    ("social_media_posts.csv", "text_original"): ("Simulated user post text narrative", "Raw NLP Feature"),
    ("social_media_posts.csv", "language"): ("Language code (e.g. en, mr, hi-en)", "NLP Metadata"),
    ("social_media_posts.csv", "latitude"): ("Reported post geotag latitude (frequently missing)", "Spatial Coordinate"),
    ("social_media_posts.csv", "longitude"): ("Reported post geotag longitude (frequently missing)", "Spatial Coordinate"),
    ("social_media_posts.csv", "location_text"): ("Mentioned landmark or suburban area in post text", "NLP Spatial Feature"),
    ("social_media_posts.csv", "permalink"): ("Simulated permalink URL (entirely null for synthetic integrity)", "Metadata (Null)"),
    ("social_media_posts.csv", "collection_method"): ("Collection mechanism (e.g. STREAM_SIMULATION)", "Pipeline Metadata"),
    ("social_media_posts.csv", "collection_status"): ("Status of collection", "Pipeline Metadata"),
    ("social_media_posts.csv", "is_synthetic"): ("Boolean synthetic flag", "Provenance Metadata"),
    ("social_media_posts.csv", "provenance"): ("Provenance label: SYNTHETIC_GENERATED", "Provenance Metadata"),

    # social_post_analysis.csv
    ("social_post_analysis.csv", "post_id"): ("Referenced post ID matching social_media_posts.post_id", "Primary / Foreign Key"),
    ("social_post_analysis.csv", "incident_relevance_score"): ("Synthetic relevance score [0.0 - 1.0]", "Model Feature / Pseudo-Label"),
    ("social_post_analysis.csv", "predicted_event_type"): ("Model-predicted incident event category", "Model Pseudo-Label"),
    ("social_post_analysis.csv", "extracted_location"): ("NER extracted geographic location text", "Model Feature (NER)"),
    ("social_post_analysis.csv", "event_time_estimate"): ("Estimated event occurrence time", "Temporal Estimate"),
    ("social_post_analysis.csv", "duplicate_score"): ("Duplicate post likelihood score", "Model Feature"),
    ("social_post_analysis.csv", "spam_or_repost_flag"): ("Boolean flag for spam or repost", "Quality Filter"),
    ("social_post_analysis.csv", "evidence_quality"): ("Heuristic evidence quality rating", "Quality Rating"),
    ("social_post_analysis.csv", "model_version"): ("Identifier of simulated extraction model", "Model Metadata"),
    ("social_post_analysis.csv", "review_status"): ("Verification review state", "Workflow Status"),
    ("social_post_analysis.csv", "is_synthetic"): ("Boolean synthetic flag", "Provenance Metadata"),

    # flood_incident_observations.csv
    ("flood_incident_observations.csv", "flood_observation_id"): ("Unique flood observation identifier", "Primary Key"),
    ("flood_incident_observations.csv", "event_type"): ("Flood incident event type (e.g. WATERLOGGING, MANHOLE_OVERFLOW)", "Event Type Feature"),
    ("flood_incident_observations.csv", "observed_time"): ("Field observation timestamp", "Temporal Feature"),
    ("flood_incident_observations.csv", "reported_time"): ("Reporting timestamp", "Temporal Metadata"),
    ("flood_incident_observations.csv", "locality"): ("Suburban Mumbai locality name", "Spatial Feature"),
    ("flood_incident_observations.csv", "latitude"): ("Flood location latitude", "Spatial Coordinate"),
    ("flood_incident_observations.csv", "longitude"): ("Flood location longitude", "Spatial Coordinate"),
    ("flood_incident_observations.csv", "catchment_id"): ("Hydrological catchment identifier", "Hydrology Feature"),
    ("flood_incident_observations.csv", "asset_id"): ("Associated asset ID in infrastructure_assets", "Foreign Key"),
    ("flood_incident_observations.csv", "water_depth_cm"): ("Estimated inundation depth (cm)", "Severity Feature"),
    ("flood_incident_observations.csv", "affected_length_m"): ("Length of roadway inundated in meters", "Impact Feature"),
    ("flood_incident_observations.csv", "road_passability"): ("Road status (RESTRICTED, SLOW, CLOSED, OPEN, UNKNOWN)", "Disruption Target / Label"),
    ("flood_incident_observations.csv", "rainfall_15min_mm"): ("Local 15-minute rainfall (mm)", "Environmental Feature"),
    ("flood_incident_observations.csv", "water_level_cm"): ("Associated drainage canal water level (cm)", "Environmental Feature"),
    ("flood_incident_observations.csv", "flow_rate_lps"): ("Stormwater discharge rate (lps)", "Environmental Feature"),
    ("flood_incident_observations.csv", "blockage_status"): ("Drainage blockage state (CONFIRMED, SUSPECTED, CLEAR, UNKNOWN)", "Diagnostic State"),
    ("flood_incident_observations.csv", "verification_status"): ("Verification state", "Verification Status"),
    ("flood_incident_observations.csv", "measurement_method"): ("Measurement mechanism (GAUGE, VISUAL, SENSOR)", "Methodology"),
    ("flood_incident_observations.csv", "is_synthetic"): ("Boolean synthetic flag", "Provenance Metadata"),
    ("flood_incident_observations.csv", "provenance"): ("Provenance label: SYNTHETIC_GENERATED", "Provenance Metadata"),

    # pairwise_relationship_training.csv
    ("pairwise_relationship_training.csv", "pair_id"): ("Unique observation pair identifier", "Primary Key"),
    ("pairwise_relationship_training.csv", "observation_id_a"): ("Observation A ID matching observations.csv", "Foreign Key"),
    ("pairwise_relationship_training.csv", "observation_id_b"): ("Observation B ID matching observations.csv", "Foreign Key"),
    ("pairwise_relationship_training.csv", "time_diff_minutes"): ("Temporal offset between observations in minutes", "Pairwise Feature"),
    ("pairwise_relationship_training.csv", "geo_distance_km"): ("Haversine geographic distance in kilometers", "Pairwise Feature"),
    ("pairwise_relationship_training.csv", "text_similarity"): ("Semantic text embedding cosine similarity [0.0 - 1.0]", "Pairwise Feature"),
    ("pairwise_relationship_training.csv", "same_locality"): ("Binary indicator if both observations share locality (1/0)", "Pairwise Feature"),
    ("pairwise_relationship_training.csv", "same_source"): ("Binary indicator if both observations share source channel (1/0)", "Pairwise Feature"),
    ("pairwise_relationship_training.csv", "reliability_product"): ("Product of source reliability scores [0.0 - 1.0]", "Pairwise Feature"),
    ("pairwise_relationship_training.csv", "relation_label"): ("Ground-truth pair relationship class (5 classes)", "Training Target / Label"),
    ("pairwise_relationship_training.csv", "is_synthetic"): ("Boolean synthetic flag", "Provenance Metadata"),
    ("pairwise_relationship_training.csv", "provenance"): ("Provenance label: SYNTHETIC_GENERATED_LABELS", "Provenance Metadata"),

    # relationship_evidence_graph_edges.csv
    ("relationship_evidence_graph_edges.csv", "edge_id"): ("Unique knowledge graph edge identifier", "Primary Key"),
    ("relationship_evidence_graph_edges.csv", "source_entity_type"): ("Entity category of source node (INCIDENT, OBSERVATION, ASSET, SOCIAL_POST)", "Graph Meta-Type"),
    ("relationship_evidence_graph_edges.csv", "source_entity_id"): ("Identifier of source entity node", "Graph Node FK"),
    ("relationship_evidence_graph_edges.csv", "target_entity_type"): ("Entity category of target node (INCIDENT, OBSERVATION, ASSET, ROUTE, LOCATION)", "Graph Meta-Type"),
    ("relationship_evidence_graph_edges.csv", "target_entity_id"): ("Identifier of target entity node", "Graph Node FK"),
    ("relationship_evidence_graph_edges.csv", "relationship_type"): ("Edge relationship predicate (e.g. IMPACTS_ROUTE, LOCATED_AT, DUPLICATE_OF)", "Graph Relation Type"),
    ("relationship_evidence_graph_edges.csv", "confidence_score"): ("Edge connection confidence score [0.0 - 1.0]", "Graph Edge Weight"),
    ("relationship_evidence_graph_edges.csv", "rule_id"): ("Inference rule triggering edge creation", "Rule Reference"),
    ("relationship_evidence_graph_edges.csv", "evidence_ids"): ("Semicolon-delimited list of supporting observation IDs", "Graph Evidence Link"),
    ("relationship_evidence_graph_edges.csv", "verification_status"): ("Verification state (HYPOTHESIS, REVIEWED_SIMULATED, UNVERIFIED)", "Graph Status"),
    ("relationship_evidence_graph_edges.csv", "explanation"): ("Natural language explanation of edge relationship", "Graph Explanation"),
    ("relationship_evidence_graph_edges.csv", "is_synthetic"): ("Boolean synthetic flag", "Provenance Metadata"),

    # routes.csv
    ("routes.csv", "route_id"): ("Unique transit/commute route identifier", "Primary Key"),
    ("routes.csv", "origin_name"): ("Commute origin point name in Western Suburbs", "Spatial Feature"),
    ("routes.csv", "origin_lat"): ("Origin latitude", "Spatial Coordinate"),
    ("routes.csv", "origin_lon"): ("Origin longitude", "Spatial Coordinate"),
    ("routes.csv", "destination_name"): ("Commute destination point name in Western Suburbs", "Spatial Feature"),
    ("routes.csv", "destination_lat"): ("Destination latitude", "Spatial Coordinate"),
    ("routes.csv", "destination_lon"): ("Destination longitude", "Spatial Coordinate"),
    ("routes.csv", "travel_mode"): ("Transport mode (DRIVE, TRANSIT, TWO_WHEELER)", "Mode Feature"),
    ("routes.csv", "routing_provider"): ("Routing engine tag (e.g. OSRM_SIMULATED)", "Provider Metadata"),
    ("routes.csv", "geometry_status"): ("Geometry generation status flag", "Status Metadata"),
    ("routes.csv", "is_synthetic"): ("Boolean synthetic flag", "Provenance Metadata"),

    # route_impacts.csv
    ("route_impacts.csv", "route_impact_id"): ("Unique route impact event identifier", "Primary Key"),
    ("route_impacts.csv", "route_id"): ("Referenced route ID matching routes.csv", "Foreign Key"),
    ("route_impacts.csv", "incident_id"): ("Referenced incident ID matching incident_clusters.csv", "Foreign Key"),
    ("route_impacts.csv", "segment_id"): ("Synthetic road/route segment ID", "Segment Identifier"),
    ("route_impacts.csv", "impact_type"): ("Disruption category (ON_ROUTE, CLOSURE, POSSIBLE_DELAY, NEAR_ROUTE_CANDIDATE, NO_CONFIRMED_IMPACT)", "Target / Impact Label"),
    ("route_impacts.csv", "distance_from_route_m"): ("Distance between incident centroid and route in meters", "Spatial Feature"),
    ("route_impacts.csv", "valid_from"): ("Impact window start timestamp", "Temporal Window"),
    ("route_impacts.csv", "valid_to"): ("Impact window end timestamp", "Temporal Window"),
    ("route_impacts.csv", "impact_confidence"): ("Confidence of impact estimation [0.0 - 1.0]", "Model Feature / Score"),
    ("route_impacts.csv", "verification_status"): ("Impact verification state", "Status Metadata"),
    ("route_impacts.csv", "is_synthetic"): ("Boolean synthetic flag", "Provenance Metadata"),
}

def generate_dictionary():
    csv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))
    rows = []

    for fp in csv_files:
        fname = os.path.basename(fp)
        df = pd.read_csv(fp, low_memory=False)
        total_rows = len(df)

        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isnull().sum())
            null_pct = round((null_count / total_rows) * 100, 2)
            unique_vals = int(df[col].nunique())
            
            desc, role = COLUMN_DESCRIPTIONS.get((fname, col), ("Dataset field", "Feature / Attribute"))
            
            sample_val = str(df[col].dropna().iloc[0]) if unique_vals > 0 else "NULL"
            if len(sample_val) > 40:
                sample_val = sample_val[:37] + "..."

            rows.append({
                "dataset_file": fname,
                "column_name": col,
                "data_type": dtype,
                "null_count": null_count,
                "null_percentage": null_pct,
                "unique_values_count": unique_vals,
                "structural_role": role,
                "description": desc,
                "sample_value": sample_val
            })

    dict_df = pd.DataFrame(rows)
    out_csv = os.path.join(DATASET_DIR, "data_dictionary.csv")
    dict_df.to_csv(out_csv, index=False)
    print(f"Generated data_dictionary.csv with {len(dict_df)} columns documented across {len(csv_files)} datasets.")
    return dict_df

if __name__ == "__main__":
    generate_dictionary()
