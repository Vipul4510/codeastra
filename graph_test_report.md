# UrbanPulse Evidence Graph Test Report (Phase 5)

## 1. Executive Summary

This report documents the verification, validation, and domain querying of the **UrbanPulse Evidence Graph** constructed for the Mumbai Western Suburbs corridor. 

Per Phase 5 requirements, graph construction, graph validation, and graph querying have been implemented as three distinct, modular Python packages (`graph_builder.py`, `graph_validation.py`, and `graph_queries.py`). The pipeline was rigorously evaluated using an end-to-end flooding scenario centered on the **Andheri Subway Monsoon Inundation** and subjected to a comprehensive automated unit test suite.

---

## 2. Unit Test Suite Execution Results

All 10 Phase 5 graph unit tests passed with 100% success rate, alongside all 22 previously established tests in Phase 2 and Phase 3 (total 32 passing tests):

```
test_contradiction_isolation_and_query (test_graph.TestUrbanPulseEvidenceGraph) ... OK
test_duplicate_edge_detection (test_graph.TestUrbanPulseEvidenceGraph) ... OK
test_evidence_repost_chain_compression (test_graph.TestUrbanPulseEvidenceGraph) ... OK
test_graph_updates_with_validation (test_graph.TestUrbanPulseEvidenceGraph) ... OK
test_invalid_endpoints_and_dangling_references (test_graph.TestUrbanPulseEvidenceGraph) ... OK
test_invalid_node_identifiers_and_missing_types (test_graph.TestUrbanPulseEvidenceGraph) ... OK
test_isolated_nodes_detection (test_graph.TestUrbanPulseEvidenceGraph) ... OK
test_json_roundtrip_serialization_and_audit (test_graph.TestUrbanPulseEvidenceGraph) ... OK
test_safe_alternative_routing_with_causation_disclaimer (test_graph.TestUrbanPulseEvidenceGraph) ... OK
test_semantic_edge_type_mismatch_prevention (test_graph.TestUrbanPulseEvidenceGraph) ... OK

----------------------------------------------------------------------
Ran 10 tests in 0.033s | Total System Suite: Ran 32 tests in 0.021s | Status: OK
```

### Invariant Test Verification Matrix

| Test Case | Invariant Tested | Result | Verification Details |
| :--- | :--- | :--- | :--- |
| `test_invalid_node_identifiers_and_missing_types` | Reqt 1: Node Identifier Validity | **PASS** | Rejects empty strings, non-strings, and unknown node types with `GraphValidationError`. |
| `test_invalid_endpoints_and_dangling_references` | Reqt 1 & 9: Endpoint Integrity | **PASS** | Prevents dangling source/target node insertions and forbids self-loops. |
| `test_semantic_edge_type_mismatch_prevention` | Reqt 2 & 4: Semantic Typing | **PASS** | Blocks nonsense connections (e.g., `IMPACTS_ROUTE` between two observations). |
| `test_duplicate_edge_detection` | Reqt 10: Duplicate Edge Audit | **PASS** | Audits edge registries and flags duplicate directed edges. |
| `test_isolated_nodes_detection` | Reqt 10: Isolated Node Audit | **PASS** | Flags degree-0 nodes in the incident evidence graph. |
| `test_contradiction_isolation_and_query` | Reqt 3 & 10: Contradiction Isolation | **PASS** | Preserves conflicting assertions without false merges; surfaces both claims in queries. |
| `test_graph_updates_with_validation` | Reqt 10: Graph Updates | **PASS** | Enforces confidence score bounds `[0.0, 1.0]` and maintains audit timestamps on edge mutation. |
| `test_evidence_repost_chain_compression` | Reqt 6: Single Evidence Chain | **PASS** | Collapses retweet cascades (`DUPLICATE_OF`) to count as 1 independent evidence chain. |
| `test_safe_alternative_routing_with_causation_disclaimer` | Reqt 7: Shortest Path Guard | **PASS** | Computes safe alternative path and verifies presence of mandatory causation disclaimer. |
| `test_json_roundtrip_serialization_and_audit` | Reqt 1 & 10: Serialization | **PASS** | Serializes graph to JSON, reloads, and passes full schema audit with 0 issues. |

---

## 3. Andheri Subway Flooding Scenario Simulation

The test scenario was executed via `run_graph_scenario.py` with 12 nodes and 13 edges.

### 3.1 Graph Topology & Schema Audit
- **Schema Valid**: `True`
- **Total Nodes**: 12 (`INCIDENT`: 1, `OBSERVATION`: 2, `SENSOR`: 1, `SOCIAL_POST`: 2, `INFRASTRUCTURE_ASSET`: 3, `ROUTE`: 2, `TRANSIT_SERVICE`: 1)
- **Total Edges**: 13 (`SUPPORTS`: 3, `CONTRADICTS`: 2, `DUPLICATE_OF`: 1, `AFFECTS_ASSET`: 1, `POSSIBLE_DEPENDENCY`: 2, `IMPACTS_ROUTE`: 3, `OBSERVED_BY`: 1)
- **Isolated Nodes**: 0
- **Validation Issues**: 0

### 3.2 Query Results

#### Query 1: Incident Corroborating Evidence Chain
- **Target Incident**: `INC-ANDHERI-FLOOD-01`
- **Total Supporting Evidence Nodes**: 3
- **Independent Evidence Chains**: 3 (Viral repost `@trafficbot` successfully consolidated under root post)
- **Mean Evidence Confidence**: 0.873
  - `[OBSERVATION] OBS-CITIZEN-01`: Confidence = 0.88 (`SUPPORTS`, Eyewitness citizen report of 55cm water depth)
  - `[SENSOR] SNS-TELEMETRY-01`: Confidence = 0.96 (`SUPPORTS`, Ultrasonic sensor confirms 58.5cm water level)
  - `[SOCIAL_POST] POST-TWITTER-01`: Confidence = 0.78 (`SUPPORTS`, Crowd signal confirms traffic paralysis)

#### Query 2: Contradiction Isolation & Review
- **Has Contradictions**: `True`
- **Conflicting Claims Count**: 1
- **Conflict Summary**: Citizen report `OBS-CONFLICT-01` ("Civic Notice: Subway Passable Claim") contradicts eyewitness report `OBS-CITIZEN-01`:
  > *"Factual conflict: Normal passability claim directly contradicts 55cm submersion reports."*
- **Outcome**: The conflicting assertion is isolated on a `CONTRADICTS` edge and not merged into the confirmed incident cluster.

#### Query 3: Connected Infrastructure & Hydraulic Dependencies
- **Directly Affected Assets**: 1
  - `[Hop 1] AST-00026` (`SUBWAY_UNDERPASS`) - Andheri Subway Underpass Arterial Culvert (`DIRECT_INCIDENT_IMPACT`)
- **Multi-Hop Connected Assets**: 2
  - `[Hop 2] AST-00052` (`RAIL_STATION`) - Andheri Western Railway Suburban Station (`DEPENDENCY: AST-00026 -> AST-00052`: Subway culvert flooding backs up into railway station pedestrian subway access)

#### Query 4: Reachable Downstream Route & Transit Disruptions
- **Total Impacted Corridors Discovered**: 3
  1. `[ROUTE] ROUTE-ANDHERI-SUBWAY`: Impact = `TOTAL_CLOSURE` (Confidence = 0.96)
  2. `[ROUTE] ROUTE-SV-ROAD`: Impact = `SEVERE_CONGESTION` (Confidence = 0.88, 2.5km gridlock diversion)
  3. `[TRANSIT_SERVICE] TRANSIT-WESTERN-RAILWAY`: Impact = `SPEED_RESTRICTION_DELAY` (Confidence = 0.74, 15-minute commuter rail speed limits)

#### Query 5: Safe Alternative Route Bypassing Flood Blockage
- **Start**: `Bandra_Junction`
- **Destination**: `Goregaon_Hub`
- **Avoided Blocked Segment**: `ROUTE-ANDHERI-SUBWAY`
- **Computed Path**: `Bandra_Junction -> WEH_Flyover_South -> ROUTE-WEH-ALTERNATIVE -> Goregaon_Hub`
- **Causation Notice**:
  > *"NOTICE ON CAUSATION: Shortest path indicates spatial topological reachability avoiding blocked nodes. It does NOT assert or establish physical causal chains."*

---

## 4. Deliverables Summary

1. `graph_validation.py`: Strict typing, endpoint checks, duplicate prevention, and full-graph auditing.
2. `graph_builder.py`: Heterogeneous NetworkX evidence graph constructor with metadata serialization.
3. `graph_queries.py`: BFS evidence traversal, repost chain consolidation, contradiction isolation, and safe routing.
4. `run_graph_scenario.py`: Andheri Subway scenario runner and query evaluator.
5. `tests/test_graph.py`: 10 comprehensive unit tests covering all Phase 5 requirements.
6. `sample_scenario_graph.json`: Serialized scenario graph nodes and edges.
7. `graph_scenario_query_results.json`: Output of all 5 domain queries in structured JSON.
8. `graph_schema.md`: Complete graph schema and query algorithm documentation.
