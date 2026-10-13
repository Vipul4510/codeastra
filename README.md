# UrbanPulse Expanded Multimodal Incident Dataset

## Important provenance notice
This is a **synthetic demonstration dataset** generated for schema design, prototyping, and pipeline testing. It is NOT a real-world incident feed, NOT verified sensor telemetry, and NOT actual social-media content. Locality names and approximate geographic anchors are used to make examples relevant to the Mumbai Western Suburbs corridor. All incident records, sensor readings, posts, labels, infrastructure relationships, and route-impact rows are synthetic. Do not present these rows as real incidents or use them as evidence of real events.

## Size
The package contains more than 25,000 rows across 13 CSV files. The primary `observations.csv` contains exactly 25,000 observations. Additional files contain linked synthetic incidents, flood readings, infrastructure, social post examples, training pairs, graph edges, and route impacts.

## Files
- `observations.csv`: Unified multimodal observation stream (25,000 rows).
- `incident_clusters.csv`: Candidate canonical incident records.
- `observation_incident_links.csv`: Synthetic candidate membership links from observations to incidents. These are generated associations, not ground truth.
- `infrastructure_assets.csv`: Synthetic roads, drains, manholes, pumps, sensors, stations and other assets.
- `asset_dependencies.csv`: Synthetic infrastructure dependency edges. Validate before operational use.
- `sensor_readings.csv`: Synthetic rainfall, water level, flow, drainage capacity and sensor-health readings.
- `social_media_posts.csv`: Synthetic example posts. These are not copied from real platforms and have no real permalinks.
- `social_post_analysis.csv`: Synthetic analysis features for social posts; not model-generated or validated predictions.
- `flood_incident_observations.csv`: Detailed synthetic flood/drainage observations.
- `pairwise_relationship_training.csv`: Synthetic pairwise relationship labels for prototyping only.
- `relationship_evidence_graph_edges.csv`: Synthetic example graph edges and rationale fields. Some entity references are intentionally illustrative; validate referential integrity before loading into a graph.
- `routes.csv`: Synthetic example route requests; not guaranteed OSRM-generated route geometry.
- `route_impacts.csv`: Synthetic route-impact examples, not confirmed road disruptions.

## Suggested usage
1. Load `observations.csv` as the unified observation table.
2. Use spatial and temporal filtering to generate candidate pairs.
3. Use `pairwise_relationship_training.csv` to prototype pairwise classification, but do not claim real-world accuracy from synthetic labels.
4. Use `flood_incident_observations.csv` and `sensor_readings.csv` to test anomaly detection and flood-related rules.
5. Use `infrastructure_assets.csv` and `asset_dependencies.csv` to prototype a NetworkX graph. Keep inferred and unverified edges distinct from verified relationships.
6. Use `relationship_evidence_graph_edges.csv` to test graph visualization only after validating entity references and edge semantics.
7. Replace synthetic routes with actual OSRM responses and use verified road geometries for route matching.
8. For social-media ingestion, connect only through permitted APIs/feeds and retain source provenance, timestamps, and access/retention controls.

## Important ML caution
Synthetic training labels may encode artificial patterns. Do not use the generated scores or labels to claim performance on real-world incidents. Before production use, collect independently reviewed real labels, use group-aware train/test splits by incident or scenario, and evaluate false merges, false splits, precision, recall, calibration, and route false-alert rate.

## Time and coordinates
Timestamps are ISO 8601 strings. Geographic columns use latitude and longitude in decimal degrees. OSRM GeoJSON uses coordinate order [longitude, latitude], so convert carefully when integrating route geometry.
