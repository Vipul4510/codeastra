"""
Comprehensive 10-Case Test Suite for UrbanPulse Real-Time Telemetry & Traffic Ingestion API
"""

import json
from fastapi.testclient import TestClient
from api_service import app, init_system

init_system()
client = TestClient(app)

test_cases = [
    {
        "id": "TC-01",
        "title": "Critical Underpass Submergence with Power Failure (Milan Subway, Santacruz)",
        "payload": {
            "source_modality": "SENSOR_TELEMETRY",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.0825,
            "longitude": 72.8412,
            "locality": "Santacruz West",
            "description_or_text": "Severe waterlogging at underpass, water level rising rapidly. Milan subway completely flooded.",
            "water_level_cm": 58.0,
            "rainfall_rate_mm": 35.0,
            "power_status": "OUTAGE",
            "pump_status": "STOPPED",
            "traffic": {
                "traffic_speed_kmh": 4.5,
                "speed_drop_pct": 78.0,
                "congestion_level": "GRIDLOCK",
                "traffic_delay_minutes": 50.0,
                "road_passability": "CLOSED",
                "affected_route_id": "ROUTE-004"
            }
        }
    },
    {
        "id": "TC-02",
        "title": "Flash Corroborating Citizen Report at Same Location (Same Incident Clustering Test)",
        "payload": {
            "source_modality": "CITIZEN_REPORT",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.0830,
            "longitude": 72.8415,
            "locality": "Santacruz West",
            "description_or_text": "Water entered bus floor at Milan subway, cars stuck in gridlock.",
            "water_level_cm": 52.0,
            "traffic": {
                "traffic_speed_kmh": 3.0,
                "speed_drop_pct": 82.0,
                "congestion_level": "GRIDLOCK",
                "traffic_delay_minutes": 55.0,
                "road_passability": "CLOSED",
                "affected_route_id": "ROUTE-004"
            }
        }
    },
    {
        "id": "TC-03",
        "title": "Moderate Highway Waterlogging on Western Express Highway (Andheri East)",
        "payload": {
            "source_modality": "SENSOR_TELEMETRY",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.1198,
            "longitude": 72.8520,
            "locality": "Andheri East",
            "description_or_text": "Curb-height waterlogging along southbound WEH carriageway, slow moving traffic.",
            "water_level_cm": 28.0,
            "rainfall_rate_mm": 18.0,
            "power_status": "NORMAL",
            "pump_status": "RUNNING",
            "traffic": {
                "traffic_speed_kmh": 16.0,
                "speed_drop_pct": 45.0,
                "congestion_level": "MODERATE",
                "traffic_delay_minutes": 22.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "ROUTE-003"
            }
        }
    },
    {
        "id": "TC-04",
        "title": "Dry Weather Traffic Jam (Unrelated to Water / Runoff - Andheri West)",
        "payload": {
            "source_modality": "TRAFFIC_FEED",
            "event_type": "TRAFFIC_SLOWDOWN",
            "latitude": 19.1364,
            "longitude": 72.8285,
            "locality": "Andheri West",
            "description_or_text": "Heavy peak hour crawl near SV Road intersection due to broken down heavy commercial vehicle.",
            "water_level_cm": 2.0,
            "rainfall_rate_mm": 0.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 11.0,
                "speed_drop_pct": 52.0,
                "congestion_level": "SEVERE",
                "traffic_delay_minutes": 25.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "ROUTE-007"
            }
        }
    },
    {
        "id": "TC-05",
        "title": "Critical Coastal Pump Station Trip during High Tide (Versova)",
        "payload": {
            "source_modality": "SENSOR_TELEMETRY",
            "event_type": "PUMP_FAILURE",
            "latitude": 19.1412,
            "longitude": 72.8170,
            "locality": "Versova",
            "description_or_text": "Primary effluent drainage pump AST-00006 tripped. Surcharge level in discharge canal.",
            "water_level_cm": 65.0,
            "rainfall_rate_mm": 22.0,
            "power_status": "OUTAGE",
            "pump_status": "FAILED",
            "traffic": {
                "traffic_speed_kmh": 22.0,
                "speed_drop_pct": 15.0,
                "congestion_level": "NORMAL",
                "traffic_delay_minutes": 5.0,
                "road_passability": "PASSABLE",
                "affected_route_id": "ROUTE-001"
            }
        }
    },
    {
        "id": "TC-06",
        "title": "Stormwater Culvert Blockage with Surcharge (Bandra West - S.V. Road)",
        "payload": {
            "source_modality": "CITIZEN_REPORT",
            "event_type": "DRAIN_BLOCKAGE",
            "latitude": 19.0550,
            "longitude": 72.8360,
            "locality": "Bandra West",
            "description_or_text": "Plastic waste clogging main stormwater culvert, water overflowing onto Linking Road.",
            "water_level_cm": 38.0,
            "rainfall_rate_mm": 12.0,
            "power_status": "NORMAL",
            "traffic": {
                "traffic_speed_kmh": 9.5,
                "speed_drop_pct": 60.0,
                "congestion_level": "SEVERE",
                "traffic_delay_minutes": 32.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "ROUTE-008"
            }
        }
    },
    {
        "id": "TC-07",
        "title": "Clear Weather Normal Sensor Baseline Check (Borivali West)",
        "payload": {
            "source_modality": "SENSOR_TELEMETRY",
            "event_type": "WATER_LEVEL_NORMAL",
            "latitude": 19.2294,
            "longitude": 72.8485,
            "locality": "Borivali West",
            "description_or_text": "All culvert telemetry levels operating normally within nominal retention threshold.",
            "water_level_cm": 4.5,
            "rainfall_rate_mm": 0.0,
            "power_status": "NORMAL",
            "pump_status": "STANDBY",
            "traffic": {
                "traffic_speed_kmh": 42.0,
                "speed_drop_pct": 0.0,
                "congestion_level": "NORMAL",
                "traffic_delay_minutes": 0.0,
                "road_passability": "PASSABLE",
                "affected_route_id": "ROUTE-009"
            }
        }
    },
    {
        "id": "TC-08",
        "title": "Sudden Cloudburst & Flash Water Accumulation (Malad Subway)",
        "payload": {
            "source_modality": "SENSOR_TELEMETRY",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.1872,
            "longitude": 72.8481,
            "locality": "Malad West",
            "description_or_text": "Intense torrential burst over Malad Subway, rapid inundation crossing yellow gauge mark.",
            "water_level_cm": 48.0,
            "rainfall_rate_mm": 52.0,
            "power_status": "NORMAL",
            "pump_status": "RUNNING",
            "traffic": {
                "traffic_speed_kmh": 5.0,
                "speed_drop_pct": 75.0,
                "congestion_level": "GRIDLOCK",
                "traffic_delay_minutes": 40.0,
                "road_passability": "CLOSED",
                "affected_route_id": "ROUTE-007"
            }
        }
    },
    {
        "id": "TC-09",
        "title": "Social Media Incident Report with High Traffic Impact (Goregaon West)",
        "payload": {
            "source_modality": "SOCIAL_POST",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.1663,
            "longitude": 72.8491,
            "locality": "Goregaon West",
            "description_or_text": "Traffic completely halted near station road due to knee high water. Avoid this route!",
            "water_level_cm": 35.0,
            "traffic": {
                "traffic_speed_kmh": 7.0,
                "speed_drop_pct": 68.0,
                "congestion_level": "SEVERE",
                "traffic_delay_minutes": 38.0,
                "road_passability": "RESTRICTED",
                "affected_route_id": "ROUTE-002"
            }
        }
    },
    {
        "id": "TC-10",
        "title": "Upstream Surcharge Propagation (Dahisar Check Naka Corridor)",
        "payload": {
            "source_modality": "SENSOR_TELEMETRY",
            "event_type": "ROAD_WATERLOGGING",
            "latitude": 19.2480,
            "longitude": 72.8589,
            "locality": "Dahisar East",
            "description_or_text": "Upstream canal bank overflow spilling onto arterial service road.",
            "water_level_cm": 62.0,
            "rainfall_rate_mm": 40.0,
            "power_status": "OUTAGE",
            "pump_status": "STOPPED",
            "traffic": {
                "traffic_speed_kmh": 2.5,
                "speed_drop_pct": 88.0,
                "congestion_level": "GRIDLOCK",
                "traffic_delay_minutes": 65.0,
                "road_passability": "CLOSED",
                "affected_route_id": "ROUTE-006"
            }
        }
    }
]

print("=" * 80)
print("RUNNING 10 TEST CASES AGAINST POST /api/v1/telemetry WITH TRAFFIC & LOCATION REASONING")
print("=" * 80)

results = []

for idx, tc in enumerate(test_cases, 1):
    res = client.post("/api/v1/telemetry", json=tc["payload"])
    assert res.status_code == 200, f"Failed on {tc['id']}: {res.text}"
    body = res.json()
    
    loc = body["incident_location"]
    resp = body["generated_response"]
    traf = resp.get("traffic_impact", {})
    ml = body.get("model_evaluation", {})

    print(f"\n[{tc['id']}] {tc['title']}")
    print(f"  Coordinates   : ({loc['coordinates']['latitude']}, {loc['coordinates']['longitude']}) in {loc['locality']}")
    print(f"  Alert Level   : {resp['alert_level']} | Urgency: {resp['urgency']}")
    print(f"  ML Class/Conf : {ml.get('predicted_relationship')} ({ml.get('confidence')})")
    print(f"  Traffic Delay : +{traf.get('delay_minutes')} min | Speed Drop: {traf.get('speed_drop_pct')}% | Passability: {traf.get('passability')}")
    print(f"  Route Detour  : {traf.get('affected_route_id')}")
    print(f"  Top Action    : {resp['recommended_actions'][0] if resp['recommended_actions'] else 'None'}")

    results.append({
        "case_id": tc["id"],
        "title": tc["title"],
        "status_code": res.status_code,
        "location": loc,
        "alert_level": resp["alert_level"],
        "urgency": resp["urgency"],
        "model_relationship": ml.get("predicted_relationship"),
        "model_confidence": ml.get("confidence"),
        "traffic_impact": traf,
        "actions_count": len(resp["recommended_actions"]),
        "sample_actions": resp["recommended_actions"][:3]
    })

with open("test_10_cases_output.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("ALL 10 TEST CASES PASSED SUCCESSFULLY. Detailed output written to test_10_cases_output.json")
print("=" * 80)
