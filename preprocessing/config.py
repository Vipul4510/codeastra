"""
UrbanPulse Preprocessing Configuration & Domain Rules
Mumbai Western Suburbs Corridor Specifications & Taxonomies
"""

from typing import Dict, Tuple

# Timezone standard: Indian Standard Time (IST, UTC+05:30)
DEFAULT_TIMEZONE_STR = "+05:30"
DEFAULT_TIMEZONE_NAME = "Asia/Kolkata"

# Mumbai Western Suburbs Geographic Bounding Box
# Extends from Bandra (south) to Dahisar / Mira Road boundary (north)
# Coastal line (west) to Sanjay Gandhi National Park / Western Express Highway (east)
MUMBAI_WESTERN_SUBURBS_BBOX = {
    "min_lat": 18.9800,  # Covers southern approaches (Mahim/Bandra)
    "max_lat": 19.2800,  # Covers northern boundary (Dahisar/Mira border)
    "min_lon": 72.7800,  # Arabian sea coast (Versova, Juhu, Bandra West)
    "max_lon": 72.9000,  # Foothills / Eastern border of Western Suburbs corridor
}

# Regional Location Zone Anchors (Matching Knowledge Graph LOC-01 to LOC-08)
LOCALITY_TO_ZONE_MAP: Dict[str, str] = {
    "BANDRA": "LOC-01",
    "KHAR": "LOC-01",
    "SANTACRUZ": "LOC-01",
    "VILE PARLE": "LOC-02",
    "JUHU": "LOC-02",
    "VERSOVA": "LOC-02",
    "ANDHERI": "LOC-03",
    "ANDHERI EAST": "LOC-03",
    "ANDHERI WEST": "LOC-03",
    "JOGESHWARI": "LOC-04",
    "OSHIWARA": "LOC-04",
    "GOREGAON": "LOC-05",
    "GOREGAON EAST": "LOC-05",
    "GOREGAON WEST": "LOC-05",
    "MALAD": "LOC-06",
    "MALAD EAST": "LOC-06",
    "MALAD WEST": "LOC-06",
    "KANDIVALI": "LOC-07",
    "KANDIVALI EAST": "LOC-07",
    "KANDIVALI WEST": "LOC-07",
    "BORIVALI": "LOC-08",
    "BORIVALI EAST": "LOC-08",
    "BORIVALI WEST": "LOC-08",
    "DAHISAR": "LOC-08",
    "DAHISAR EAST": "LOC-08",
    "DAHISAR WEST": "LOC-08",
}

# Approximate Centroids for Mumbai Western Suburbs Locality Anchors (lat, lon)
LOCALITY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "LOC-01": (19.0596, 72.8295),  # Bandra
    "LOC-02": (19.1000, 72.8350),  # Vile Parle / Versova / Juhu
    "LOC-03": (19.1197, 72.8464),  # Andheri
    "LOC-04": (19.1360, 72.8490),  # Jogeshwari
    "LOC-05": (19.1663, 72.8526),  # Goregaon
    "LOC-06": (19.1860, 72.8485),  # Malad
    "LOC-07": (19.2060, 72.8510),  # Kandivali
    "LOC-08": (19.2307, 72.8567),  # Borivali / Dahisar
}

# Canonical Event Taxonomy
CANONICAL_EVENT_TYPE_MAP: Dict[str, str] = {
    # Flooding & Hydrology
    "ROAD_WATERLOGGING": "ROAD_WATERLOGGING",
    "WATERLOGGING": "ROAD_WATERLOGGING",
    "FLOOD": "ROAD_WATERLOGGING",
    "FLOODING": "ROAD_WATERLOGGING",
    "WATER_LOGGING": "ROAD_WATERLOGGING",
    "WATER_LEVEL_RISE": "WATER_LEVEL_RISE",
    "WATER_RISE": "WATER_LEVEL_RISE",
    "HIGH_WATER_LEVEL": "WATER_LEVEL_RISE",
    "SUBWAY_INUNDATION": "SUBWAY_INUNDATION",
    "UNDERPASS_FLOODING": "SUBWAY_INUNDATION",
    "UNDERPASS_FLOODED": "SUBWAY_INUNDATION",
    "SUBWAY_FLOODED": "SUBWAY_INUNDATION",
    "MANHOLE_OVERFLOW": "MANHOLE_OVERFLOW",
    "MANHOLE_OPEN": "MANHOLE_OVERFLOW",
    "DRAIN_BLOCKAGE": "DRAIN_BLOCKAGE",
    "DRAINAGE_OVERFLOW": "DRAIN_BLOCKAGE",
    "CLOGGED_DRAIN": "DRAIN_BLOCKAGE",
    "PUMP_FAILURE": "PUMP_FAILURE",
    "PUMP_OFFLINE": "PUMP_FAILURE",
    "PUMPING_STATION_FAILURE": "PUMP_FAILURE",
    "HEAVY_RAINFALL": "HEAVY_RAINFALL",
    "TORRENTIAL_RAIN": "HEAVY_RAINFALL",
    "RAIN_STORM": "HEAVY_RAINFALL",

    # Traffic & Mobility
    "TRAFFIC_SLOWDOWN": "TRAFFIC_SLOWDOWN",
    "TRAFFIC_JAM": "TRAFFIC_SLOWDOWN",
    "CONGESTION": "TRAFFIC_SLOWDOWN",
    "ROAD_CLOSURE": "ROAD_CLOSURE",
    "ROAD_BLOCKED": "ROAD_CLOSURE",
    "ROUTE_DIVERTED": "ROAD_CLOSURE",
    "ROAD_ACCIDENT": "ROAD_ACCIDENT",
    "ACCIDENT": "ROAD_ACCIDENT",
    "VEHICLE_COLLISION": "ROAD_ACCIDENT",
    "TRANSIT_DELAY": "TRANSIT_DELAY",
    "BUS_DELAY": "TRANSIT_DELAY",
    "TRAIN_DELAY": "TRANSIT_DELAY",

    # Rail & Power Infrastructure
    "RAIL_SIGNAL_FAILURE": "RAIL_SIGNAL_FAILURE",
    "SIGNAL_FAULT": "RAIL_SIGNAL_FAILURE",
    "OHE_WIRE_FAILURE": "OHE_WIRE_FAILURE",
    "OVERHEAD_WIRE_BREAK": "OHE_WIRE_FAILURE",
    "OHE_BREAKDOWN": "OHE_WIRE_FAILURE",
    "POWER_FAILURE": "POWER_FAILURE",
    "POWER_OUTAGE": "POWER_FAILURE",
    "BLACKOUT": "POWER_FAILURE",

    # Environmental & Civil Incidents
    "TREE_FALL": "TREE_FALL",
    "FALLEN_TREE": "TREE_FALL",
    "TREE_COLLAPSE": "TREE_FALL",
    "FIRE_REPORT": "FIRE_REPORT",
    "FIRE_ALARM": "FIRE_REPORT",
    "SENSOR_ANOMALY": "SENSOR_ANOMALY",
    "TELEMETRY_ANOMALY": "SENSOR_ANOMALY",
    "GENERAL_INCIDENT": "GENERAL_INCIDENT",
    "CIVIC_INCIDENT": "GENERAL_INCIDENT",
}

# Canonical Source Modality Taxonomy
CANONICAL_SOURCE_MAP: Dict[str, str] = {
    "SENSOR": "SENSOR",
    "SRC-SENSOR": "SENSOR",
    "SOCIAL_MEDIA": "SOCIAL_MEDIA",
    "SRC-SOCIAL_MEDIA": "SOCIAL_MEDIA",
    "CITIZEN_REPORT": "CITIZEN_REPORT",
    "SRC-CITIZEN_REPORT": "CITIZEN_REPORT",
    "OFFICIAL_FEED": "OFFICIAL_FEED",
    "SRC-OFFICIAL_FEED": "OFFICIAL_FEED",
    "WEATHER_FEED": "WEATHER_FEED",
    "SRC-WEATHER_FEED": "WEATHER_FEED",
    "TRAFFIC_FEED": "TRAFFIC_FEED",
    "SRC-TRAFFIC_FEED": "TRAFFIC_FEED",
    "TRANSIT_FEED": "TRANSIT_FEED",
    "SRC-TRANSIT_FEED": "TRANSIT_FEED",
    "SIMULATED_TEST": "SIMULATED_TEST",
    "SRC-SIMULATED_TEST": "SIMULATED_TEST",
}

# Prior Reliability Defaults by Source Type (when raw value is missing)
DEFAULT_SOURCE_RELIABILITY: Dict[str, float] = {
    "OFFICIAL_FEED": 0.95,
    "SENSOR": 0.88,
    "WEATHER_FEED": 0.85,
    "TRANSIT_FEED": 0.85,
    "TRAFFIC_FEED": 0.80,
    "CITIZEN_REPORT": 0.65,
    "SOCIAL_MEDIA": 0.55,
    "SIMULATED_TEST": 0.70,
    "UNKNOWN": 0.50,
}

# Repost & Copy Detection Parameters
REPOST_SIMILARITY_THRESHOLD = 0.82
REPOST_TIME_WINDOW_HOURS = 36.0
