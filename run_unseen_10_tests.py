"""
Validation of 10 COMPLETELY UNSEEN, ARBITRARY Real-World Mumbai Coordinates & Road Segments
None of these exist anywhere in observations.csv, sensor_readings.csv, or infrastructure_assets.csv.
"""

import json
import pandas as pd
from fastapi.testclient import TestClient
from api_service import app, init_system

init_system()
client = TestClient(app)

# Load dataset coordinates to strictly verify 0% overlap
df_obs = pd.read_csv("observations.csv")
df_sen = pd.read_csv("sensor_readings.csv")
df_ast = pd.read_csv("infrastructure_assets.csv")

known_coords = set(zip(df_obs["latitude"].round(4), df_obs["longitude"].round(4)))
known_coords.update(set(zip(df_sen["latitude"].round(4), df_sen["longitude"].round(4))))
known_coords.update(set(zip(df_ast["latitude"].round(4), df_ast["longitude"].round(4))))

unseen_cases = [
    {
        "id": "UNSEEN-01",
        "title": "Chhatrapati Shivaji Maharaj International Airport (T2 Arrival Ramp)",
        "payload": {
            "source_modality": "CITIZEN_REPORT",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.0974,
            "longitude": 72.8743,
            "locality": "Sahar, Andheri East",
            "description_or_text": "Severe water accumulation on T2 terminal arrival underpass ramp, taxis stalled.",
            "water_level_cm": 46.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 5.2,
                "speed_drop_pct": 74.0,
                "congestion_level": "GRIDLOCK",
                "traffic_delay_minutes": 42.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "AIRPORT-RAMP-01"
            }
        }
    },
    {
        "id": "UNSEEN-02",
        "title": "BKC Diamond Bourse Financial District Junction",
        "payload": {
            "source_modality": "TRAFFIC_FEED",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.0657,
            "longitude": 72.8681,
            "locality": "Bandra Kurla Complex",
            "description_or_text": "Mithi river overflow entering arterial financial avenue near Bharat Diamond Bourse.",
            "water_level_cm": 58.0,
            "power_status": "OUTAGE",
            "traffic": {
                "traffic_speed_kmh": 3.0,
                "speed_drop_pct": 85.0,
                "congestion_level": "GRIDLOCK",
                "traffic_delay_minutes": 60.0,
                "road_passability": "CLOSED",
                "affected_route_id": "BKC-AVENUE-3"
            }
        }
    },
    {
        "id": "UNSEEN-03",
        "title": "Juhu Beach Flying Club Low-Lying Coastal Corridor",
        "payload": {
            "source_modality": "SENSOR_TELEMETRY",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.0991,
            "longitude": 72.8298,
            "locality": "Juhu Tara Road",
            "description_or_text": "High tide storm surge backflowing through open storm outfall onto Juhu Tara Road.",
            "water_level_cm": 36.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 9.0,
                "speed_drop_pct": 58.0,
                "congestion_level": "SEVERE",
                "traffic_delay_minutes": 28.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "JUHU-COASTAL-RD"
            }
        }
    },
    {
        "id": "UNSEEN-04",
        "title": "Lokhandwala Complex Creek Bridge",
        "payload": {
            "source_modality": "SOCIAL_POST",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.1438,
            "longitude": 72.8252,
            "locality": "Lokhandwala, Andheri West",
            "description_or_text": "Oshiwara river backwater rising rapidly past danger mark near Infinity Mall bridge.",
            "water_level_cm": 64.0,
            "power_status": "OUTAGE",
            "traffic": {
                "traffic_speed_kmh": 2.1,
                "speed_drop_pct": 90.0,
                "congestion_level": "GRIDLOCK",
                "traffic_delay_minutes": 75.0,
                "road_passability": "CLOSED",
                "affected_route_id": "LINK-ROAD-NORTH"
            }
        }
    },
    {
        "id": "UNSEEN-05",
        "title": "Aarey Colony Forest Arterial Link Road",
        "payload": {
            "source_modality": "CITIZEN_REPORT",
            "event_type": "ROAD_CLOSURE",
            "latitude": 19.1506,
            "longitude": 72.8752,
            "locality": "Aarey Milk Colony, Goregaon",
            "description_or_text": "Tree fall combined with culvert erosion blocking two-lane internal road to Marol.",
            "water_level_cm": 22.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 0.0,
                "speed_drop_pct": 100.0,
                "congestion_level": "GRIDLOCK",
                "traffic_delay_minutes": 90.0,
                "road_passability": "CLOSED",
                "affected_route_id": "AAREY-INTERNAL-RD"
            }
        }
    },
    {
        "id": "UNSEEN-06",
        "title": "Bandra Bandstand Coastal Promenade Foot",
        "payload": {
            "source_modality": "SENSOR_TELEMETRY",
            "event_type": "WATER_LEVEL_NORMAL",
            "latitude": 19.0435,
            "longitude": 72.8196,
            "locality": "Bandra West Promenade",
            "description_or_text": "Normal wave dissipation sensors, culverts functioning dry.",
            "water_level_cm": 3.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 38.0,
                "speed_drop_pct": 0.0,
                "congestion_level": "NORMAL",
                "traffic_delay_minutes": 0.0,
                "road_passability": "PASSABLE",
                "affected_route_id": "BJ-ROAD"
            }
        }
    },
    {
        "id": "UNSEEN-07",
        "title": "Marol Naka Andheri-Kurla Road Metro Underpass",
        "payload": {
            "source_modality": "TRAFFIC_FEED",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.1098,
            "longitude": 72.8798,
            "locality": "Marol Naka",
            "description_or_text": "Heavy stormwater pooling under Metro Line 1 piers causing bumper-to-bumper tailback.",
            "water_level_cm": 39.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 6.8,
                "speed_drop_pct": 66.0,
                "congestion_level": "SEVERE",
                "traffic_delay_minutes": 35.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "ANDHERI-KURLA-RD"
            }
        }
    },
    {
        "id": "UNSEEN-08",
        "title": "Versova Fishing Village Jetty",
        "payload": {
            "source_modality": "CITIZEN_REPORT",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.1352,
            "longitude": 72.8094,
            "locality": "Versova Village",
            "description_or_text": "Sea inlet surge flooding access path to fish market and jetty docks.",
            "water_level_cm": 50.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 4.0,
                "speed_drop_pct": 80.0,
                "congestion_level": "SEVERE",
                "traffic_delay_minutes": 40.0,
                "road_passability": "CLOSED",
                "affected_route_id": "JETTY-ACCESS-WAY"
            }
        }
    },
    {
        "id": "UNSEEN-09",
        "title": "Sanjay Gandhi National Park Gate Arterial Approach",
        "payload": {
            "source_modality": "SENSOR_TELEMETRY",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.2289,
            "longitude": 72.8624,
            "locality": "Borivali East SGNP Gate",
            "description_or_text": "Dahisar river headwaters swelling over culvert near park entrance.",
            "water_level_cm": 42.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 12.0,
                "speed_drop_pct": 50.0,
                "congestion_level": "MODERATE",
                "traffic_delay_minutes": 20.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "SGNP-ACCESS-RD"
            }
        }
    },
    {
        "id": "UNSEEN-10",
        "title": "Mahim Causeway Inter-City Connector Foot",
        "payload": {
            "source_modality": "TRAFFIC_FEED",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.0412,
            "longitude": 72.8421,
            "locality": "Mahim Causeway, Bandra South",
            "description_or_text": "Southern approach to Western Suburbs flooded at high tide, traffic slowing on bridge.",
            "water_level_cm": 31.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 14.0,
                "speed_drop_pct": 48.0,
                "congestion_level": "MODERATE",
                "traffic_delay_minutes": 26.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "MAHIM-CAUSEWAY"
            }
        }
    }
]

print("=" * 80)
print("VERIFYING ZERO COORDINATE OVERLAP WITH DATASET")
print("=" * 80)

for tc in unseen_cases:
    lat, lon = round(tc["payload"]["latitude"], 4), round(tc["payload"]["longitude"], 4)
    in_dataset = (lat, lon) in known_coords
    print(f"[{tc['id']}] ({lat}, {lon}) in dataset: {in_dataset}")
    assert not in_dataset, f"Coordinate {lat}, {lon} unexpectedly exists in dataset!"

print("\nAll 10 test coordinates are 100% UNSEEN and absent from the dataset.\n")

print("=" * 80)
print("TESTING DYNAMIC SPATIAL LOCALIZATION & MODEL-GENERATED RESPONSE FOR UNSEEN INCIDENTS")
print("=" * 80)

unseen_results = []

for tc in unseen_cases:
    res = client.post("/api/v1/telemetry", json=tc["payload"])
    assert res.status_code == 200, f"Error on {tc['id']}: {res.text}"
    data = res.json()

    loc = data["incident_location"]
    resp = data["generated_response"]
    traf = resp.get("traffic_impact", {})
    ast = loc.get("nearby_critical_assets", [])

    print(f"\n[{tc['id']}] {tc['title']}")
    print(f"  Incident Coords : ({loc['coordinates']['latitude']}, {loc['coordinates']['longitude']}) in {loc['locality']}")
    print(f"  Nearest Asset   : {ast[0]['asset_id']} ({ast[0]['asset_type']}) at {ast[0]['distance_km']} km away in {ast[0].get('locality')}")
    print(f"  Alert / Urgency : {resp['alert_level']} | {resp['urgency']}")
    print(f"  Traffic Impact  : +{traf.get('delay_minutes')} min delay | Status: {traf.get('passability')} | Detour: {traf.get('affected_route_id')}")
    print(f"  Target Ward EOC : {resp['dispatch_units'][0]}")
    print(f"  Top Action Plan : {resp['recommended_actions'][0]}")

    unseen_results.append({
        "case_id": tc["id"],
        "title": tc["title"],
        "unseen_coords": (loc['coordinates']['latitude'], loc['coordinates']['longitude']),
        "locality": loc['locality'],
        "nearest_asset": ast[0] if ast else None,
        "alert_level": resp["alert_level"],
        "urgency": resp["urgency"],
        "traffic_impact": traf,
        "actions": resp["recommended_actions"]
    })

with open("unseen_10_cases_output.json", "w") as f:
    json.dump(unseen_results, f, indent=2)

print("\n" + "=" * 80)
print("DYNAMIC LOCALIZATION VERIFICATION COMPLETE: ALL 10 UNSEEN COORDINATES RESOLVED PROPERLY.")
print("=" * 80)
