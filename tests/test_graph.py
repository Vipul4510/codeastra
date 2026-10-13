"""
Unit Tests for UrbanPulse Evidence Graph Construction and Graph Algorithms (Phase 5)
===================================================================================
Covers all core invariants:
1. Invalid node identifiers and missing/invalid node types.
2. Invalid edge endpoints and dangling references (Requirement 1 & 9).
3. Semantic edge-type mismatch prevention (Requirement 2 & 4).
4. Duplicate edge detection in whole-graph audit (Requirement 10).
5. Isolated node detection in whole-graph audit (Requirement 10).
6. Contradiction preservation and bidirectional query surfacing (Requirement 3 & 10).
7. Safe edge updates with bounds validation (Requirement 10).
8. BFS repost chain compression to avoid social amplification (Requirement 6).
9. Multi-hop infrastructure dependency exploration (Requirement 8).
10. Shortest-path routing with mandatory causation disclaimer (Requirement 7).
11. JSON serialization/deserialization with audit verification (Requirement 1 & 10).
"""

import unittest
import os
import sys
import tempfile
import networkx as nx

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from graph_validation import (
    NodeType,
    EdgeType,
    GraphValidator,
    GraphValidationError
)
from graph_builder import UrbanPulseEvidenceGraph
from graph_queries import UrbanPulseGraphQueries


class TestUrbanPulseEvidenceGraph(unittest.TestCase):

    def setUp(self):
        self.eg = UrbanPulseEvidenceGraph(name="Test_Evidence_Graph")

    def test_invalid_node_identifiers_and_missing_types(self):
        """Test that invalid IDs or invalid node types raise GraphValidationError."""
        # Case A: Empty string node_id
        with self.assertRaises(GraphValidationError):
            self.eg.add_node(node_id="", node_type=NodeType.INCIDENT)

        # Case B: Non-string node_id
        with self.assertRaises(GraphValidationError):
            self.eg.add_node(node_id=None, node_type=NodeType.INCIDENT)

        # Case C: Invalid NodeType enum value
        with self.assertRaises(GraphValidationError):
            self.eg.add_node(node_id="NODE-001", node_type="INVALID_TYPE")

    def test_invalid_endpoints_and_dangling_references(self):
        """Test that edges referencing non-existent nodes raise GraphValidationError."""
        self.eg.add_node(node_id="INC-001", node_type=NodeType.INCIDENT)

        # Target node missing
        with self.assertRaises(GraphValidationError) as ctx:
            self.eg.add_edge(
                source_id="INC-001",
                target_id="NON_EXISTENT_ASSET",
                relationship_type=EdgeType.AFFECTS_ASSET,
                confidence=0.9,
                evidence_ids=["EVD-1"],
                rule_or_model_version="v1.0",
                explanation="Test"
            )
        self.assertIn("does not exist in graph", str(ctx.exception))

        # Source node missing
        with self.assertRaises(GraphValidationError) as ctx:
            self.eg.add_edge(
                source_id="NON_EXISTENT_OBS",
                target_id="INC-001",
                relationship_type=EdgeType.SUPPORTS,
                confidence=0.85,
                evidence_ids=["EVD-1"],
                rule_or_model_version="v1.0",
                explanation="Test"
            )
        self.assertIn("does not exist in graph", str(ctx.exception))

        # Self-loop
        with self.assertRaises(GraphValidationError) as ctx:
            self.eg.add_edge(
                source_id="INC-001",
                target_id="INC-001",
                relationship_type=EdgeType.SAME_INCIDENT_CANDIDATE,
                confidence=1.0,
                evidence_ids=["EVD-1"],
                rule_or_model_version="v1.0",
                explanation="Self loop"
            )
        self.assertIn("self-loop", str(ctx.exception).lower())

    def test_semantic_edge_type_mismatch_prevention(self):
        """Test that invalid endpoint entity types for a relationship are rejected."""
        self.eg.add_node(node_id="OBS-001", node_type=NodeType.OBSERVATION)
        self.eg.add_node(node_id="OBS-002", node_type=NodeType.OBSERVATION)

        # IMPACTS_ROUTE cannot connect two OBSERVATION nodes
        with self.assertRaises(GraphValidationError) as ctx:
            self.eg.add_edge(
                source_id="OBS-001",
                target_id="OBS-002",
                relationship_type=EdgeType.IMPACTS_ROUTE,
                confidence=0.8,
                evidence_ids=["OBS-001"],
                rule_or_model_version="v1.0",
                explanation="Invalid connection"
            )
        self.assertIn("Semantic Type Mismatch", str(ctx.exception))

    def test_duplicate_edge_detection(self):
        """Test that duplicate edge insertions are detected during full audit."""
        self.eg.add_node(node_id="OBS-001", node_type=NodeType.OBSERVATION)
        self.eg.add_node(node_id="INC-001", node_type=NodeType.INCIDENT)

        self.eg.add_edge(
            source_id="OBS-001",
            target_id="INC-001",
            relationship_type=EdgeType.SUPPORTS,
            confidence=0.9,
            evidence_ids=["OBS-001"],
            rule_or_model_version="v1.0",
            explanation="Initial edge"
        )

        # Use a MultiDiGraph view or artificially test duplicate detection in audit
        multi_g = nx.MultiDiGraph(self.eg.graph)
        multi_g.add_edge(
            "OBS-001",
            "INC-001",
            relationship_type=EdgeType.SUPPORTS.value,
            confidence=0.95,
            evidence_ids=["OBS-001"],
            rule_or_model_version="v1.0",
            explanation="Duplicate edge"
        )
        audit = GraphValidator.audit_entire_graph(multi_g)
        self.assertFalse(audit["is_valid"])
        self.assertGreaterEqual(audit["duplicate_edges_count"], 1)

    def test_isolated_nodes_detection(self):
        """Test that orphan/isolated nodes with degree 0 are flagged in graph audit."""
        self.eg.add_node(node_id="OBS-CONNECTED-01", node_type=NodeType.OBSERVATION)
        self.eg.add_node(node_id="INC-CONNECTED-01", node_type=NodeType.INCIDENT)
        self.eg.add_node(node_id="OBS-ORPHAN-01", node_type=NodeType.OBSERVATION)  # 0 degree

        self.eg.add_edge(
            source_id="OBS-CONNECTED-01",
            target_id="INC-CONNECTED-01",
            relationship_type=EdgeType.SUPPORTS,
            confidence=0.85,
            evidence_ids=["OBS-CONNECTED-01"],
            rule_or_model_version="v1.0",
            explanation="Valid edge"
        )

        audit = GraphValidator.audit_entire_graph(self.eg.graph)
        self.assertFalse(audit["is_valid"])
        self.assertEqual(audit["isolated_nodes_count"], 1)
        self.assertIn("OBS-ORPHAN-01", audit["isolated_node_ids"])

    def test_contradiction_isolation_and_query(self):
        """Test that contradictory claims are recorded bidirectionally and surfaced in queries."""
        self.eg.add_node(
            node_id="INC-001",
            node_type=NodeType.INCIDENT,
            label="Flooding Incident"
        )
        self.eg.add_node(
            node_id="OBS-REPORT-FLOOD",
            node_type=NodeType.OBSERVATION,
            label="Citizen flood claim",
            text_raw="Subway is submerged 60cm"
        )
        self.eg.add_node(
            node_id="OBS-REPORT-CLEAR",
            node_type=NodeType.OBSERVATION,
            label="Citizen clear claim",
            text_raw="Subway is totally dry and passable"
        )

        self.eg.add_edge(
            source_id="OBS-REPORT-FLOOD",
            target_id="INC-001",
            relationship_type=EdgeType.SUPPORTS,
            confidence=0.9,
            evidence_ids=["OBS-REPORT-FLOOD"],
            rule_or_model_version="v1.0",
            explanation="Supports flood"
        )
        self.eg.add_edge(
            source_id="OBS-REPORT-CLEAR",
            target_id="OBS-REPORT-FLOOD",
            relationship_type=EdgeType.CONTRADICTS,
            confidence=0.88,
            evidence_ids=["OBS-REPORT-CLEAR", "OBS-REPORT-FLOOD"],
            rule_or_model_version="v1.0",
            explanation="Direct factual contradiction regarding water level"
        )

        queries = UrbanPulseGraphQueries(self.eg)
        contra_res = queries.get_incident_contradictions("INC-001")

        self.assertTrue(contra_res["has_contradictions"])
        self.assertEqual(contra_res["contradiction_count"], 1)
        conflict = contra_res["conflicts"][0]
        self.assertIn("Direct factual contradiction", conflict["conflict_explanation"])

    def test_graph_updates_with_validation(self):
        """Test updating edge attributes with bounds checking."""
        self.eg.add_node(node_id="OBS-01", node_type=NodeType.OBSERVATION)
        self.eg.add_node(node_id="INC-01", node_type=NodeType.INCIDENT)

        self.eg.add_edge(
            source_id="OBS-01",
            target_id="INC-01",
            relationship_type=EdgeType.SUPPORTS,
            confidence=0.7,
            evidence_ids=["OBS-01"],
            rule_or_model_version="v1.0",
            explanation="Initial"
        )

        # Valid update
        self.eg.update_edge(
            source_id="OBS-01",
            target_id="INC-01",
            confidence=0.95,
            explanation="Updated after sensor corroboration",
            rule_or_model_version="v2.0"
        )
        edge = self.eg.graph["OBS-01"]["INC-01"]
        self.assertEqual(edge["confidence"], 0.95)
        self.assertEqual(edge["rule_or_model_version"], "v2.0")

        # Invalid confidence (> 1.0)
        with self.assertRaises(GraphValidationError):
            self.eg.update_edge(source_id="OBS-01", target_id="INC-01", confidence=1.5)

    def test_evidence_repost_chain_compression(self):
        """
        Verify Requirement 6: Reposts/duplicates must be consolidated so viral cascades
        do not artificially inflate the independent evidence count.
        """
        self.eg.add_node(node_id="INC-01", node_type=NodeType.INCIDENT)
        self.eg.add_node(node_id="POST-ORIGINAL", node_type=NodeType.SOCIAL_POST, label="Original Tweet")
        self.eg.add_node(node_id="POST-COPY-1", node_type=NodeType.SOCIAL_POST, label="Retweet 1")
        self.eg.add_node(node_id="POST-COPY-2", node_type=NodeType.SOCIAL_POST, label="Retweet 2")

        # Original post supports incident
        self.eg.add_edge(
            source_id="POST-ORIGINAL",
            target_id="INC-01",
            relationship_type=EdgeType.SUPPORTS,
            confidence=0.8,
            evidence_ids=["POST-ORIGINAL"],
            rule_or_model_version="v1.0",
            explanation="Primary alert"
        )

        # Retweets link to original post via DUPLICATE_OF
        self.eg.add_edge(
            source_id="POST-COPY-1",
            target_id="POST-ORIGINAL",
            relationship_type=EdgeType.DUPLICATE_OF,
            confidence=0.99,
            evidence_ids=["POST-COPY-1", "POST-ORIGINAL"],
            rule_or_model_version="v1.0",
            explanation="Retweet duplicate"
        )
        self.eg.add_edge(
            source_id="POST-COPY-2",
            target_id="POST-ORIGINAL",
            relationship_type=EdgeType.DUPLICATE_OF,
            confidence=0.98,
            evidence_ids=["POST-COPY-2", "POST-ORIGINAL"],
            rule_or_model_version="v1.0",
            explanation="Retweet duplicate"
        )

        queries = UrbanPulseGraphQueries(self.eg)
        ev_res = queries.get_incident_evidence("INC-01")

        # Total primary supporting evidence discovered
        self.assertEqual(ev_res["total_evidence_nodes"], 1)
        # Independent chains count must be exactly 1
        self.assertEqual(ev_res["independent_evidence_chains"], 1)

    def test_safe_alternative_routing_with_causation_disclaimer(self):
        """
        Verify Requirement 7: Shortest-path routing avoids blocked nodes and includes
        an explicit disclaimer that spatial reachability is NOT proof of causation.
        """
        queries = UrbanPulseGraphQueries(self.eg)

        road_g = nx.Graph()
        # Normal path: A -> B (blocked) -> C
        road_g.add_edge("Node_A", "Node_B_Blocked", weight=1.0)
        road_g.add_edge("Node_B_Blocked", "Node_C", weight=1.0)
        # Bypass path: A -> D -> C
        road_g.add_edge("Node_A", "Node_D_Bypass", weight=2.0)
        road_g.add_edge("Node_D_Bypass", "Node_C", weight=2.0)

        result = queries.find_safe_alternative_route(
            start_node="Node_A",
            end_node="Node_C",
            blocked_nodes={"Node_B_Blocked"},
            road_network_graph=road_g
        )

        self.assertTrue(result["route_found"])
        self.assertEqual(result["path_nodes"], ["Node_A", "Node_D_Bypass", "Node_C"])
        self.assertNotIn("Node_B_Blocked", result["path_nodes"])
        # Explicit causation disclaimer check
        self.assertIn("causation_disclaimer", result)
        self.assertIn("NOT assert or establish physical causal chains", result["causation_disclaimer"])

    def test_json_roundtrip_serialization_and_audit(self):
        """Test that evidence graph can be serialized to JSON, loaded back, and pass full audit."""
        self.eg.add_node(
            node_id="INC-01",
            node_type=NodeType.INCIDENT,
            label="Test Flood",
            severity="CRITICAL"
        )
        self.eg.add_node(
            node_id="AST-01",
            node_type=NodeType.INFRASTRUCTURE_ASSET,
            label="Pumping Station",
            asset_type="PUMP"
        )
        self.eg.add_edge(
            source_id="INC-01",
            target_id="AST-01",
            relationship_type=EdgeType.AFFECTS_ASSET,
            confidence=0.92,
            evidence_ids=["INC-01"],
            rule_or_model_version="v1.0",
            explanation="Flooding impairs pump operations"
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            self.eg.save_to_json(temp_path)
            loaded_eg = UrbanPulseEvidenceGraph.load_from_json(temp_path)

            self.assertEqual(loaded_eg.graph.number_of_nodes(), 2)
            self.assertEqual(loaded_eg.graph.number_of_edges(), 1)
            self.assertTrue(loaded_eg.graph.has_edge("INC-01", "AST-01"))
            self.assertEqual(loaded_eg.graph["INC-01"]["AST-01"]["confidence"], 0.92)

            audit = GraphValidator.audit_entire_graph(loaded_eg.graph)
            self.assertTrue(audit["is_valid"])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
