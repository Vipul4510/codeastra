"""
UrbanPulse Graph Validation Module
===================================
Enforces strict schema compliance, endpoint validation, duplicate prevention,
and semantic integrity for the UrbanPulse multimodal evidence graph.

Validation Invariants:
1. All node IDs must be unique and have a valid NodeType.
2. All edge endpoints (source, target) must exist in the node registry.
3. Edge types must conform to EdgeType taxonomy.
4. Edge attributes must contain: relationship_type, evidence_ids, rule_or_model_version,
   timestamp, confidence, and human-readable explanation.
5. Identity relationships (SAME_INCIDENT, DUPLICATE) must not be confused with
   impact relationships (AFFECTS_ASSET, IMPACTS_ROUTE, POSSIBLE_DEPENDENCY).
6. Detect and isolate contradictions; prevent unsupported dependencies.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx


class NodeType(str, Enum):
    OBSERVATION = "OBSERVATION"
    INCIDENT = "INCIDENT"
    INFRASTRUCTURE_ASSET = "INFRASTRUCTURE_ASSET"
    SENSOR = "SENSOR"
    SOCIAL_POST = "SOCIAL_POST"
    ROUTE = "ROUTE"
    TRANSIT_SERVICE = "TRANSIT_SERVICE"


class EdgeType(str, Enum):
    # Evidence & Identity Edges
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    DUPLICATE_OF = "DUPLICATE_OF"
    SAME_INCIDENT_CANDIDATE = "SAME_INCIDENT_CANDIDATE"
    # Physical & Mobility Edges
    AFFECTS_ASSET = "AFFECTS_ASSET"
    POSSIBLE_DEPENDENCY = "POSSIBLE_DEPENDENCY"
    IMPACTS_ROUTE = "IMPACTS_ROUTE"
    OBSERVED_BY = "OBSERVED_BY"


# Permitted (SourceType, TargetType) for each EdgeType to prevent semantic nonsense
PERMITTED_EDGE_TYPING: Dict[EdgeType, Set[Tuple[NodeType, NodeType]]] = {
    EdgeType.SUPPORTS: {
        (NodeType.OBSERVATION, NodeType.INCIDENT),
        (NodeType.SENSOR, NodeType.INCIDENT),
        (NodeType.SOCIAL_POST, NodeType.INCIDENT),
        (NodeType.OBSERVATION, NodeType.OBSERVATION),
    },
    EdgeType.CONTRADICTS: {
        (NodeType.OBSERVATION, NodeType.OBSERVATION),
        (NodeType.OBSERVATION, NodeType.INCIDENT),
        (NodeType.SOCIAL_POST, NodeType.INCIDENT),
        (NodeType.SENSOR, NodeType.OBSERVATION),
    },
    EdgeType.DUPLICATE_OF: {
        (NodeType.SOCIAL_POST, NodeType.SOCIAL_POST),
        (NodeType.OBSERVATION, NodeType.OBSERVATION),
    },
    EdgeType.SAME_INCIDENT_CANDIDATE: {
        (NodeType.OBSERVATION, NodeType.OBSERVATION),
        (NodeType.SOCIAL_POST, NodeType.OBSERVATION),
        (NodeType.SENSOR, NodeType.OBSERVATION),
        (NodeType.INCIDENT, NodeType.INCIDENT),
    },
    EdgeType.AFFECTS_ASSET: {
        (NodeType.INCIDENT, NodeType.INFRASTRUCTURE_ASSET),
        (NodeType.OBSERVATION, NodeType.INFRASTRUCTURE_ASSET),
    },
    EdgeType.POSSIBLE_DEPENDENCY: {
        (NodeType.INFRASTRUCTURE_ASSET, NodeType.INFRASTRUCTURE_ASSET),
    },
    EdgeType.IMPACTS_ROUTE: {
        (NodeType.INCIDENT, NodeType.ROUTE),
        (NodeType.INCIDENT, NodeType.TRANSIT_SERVICE),
        (NodeType.INFRASTRUCTURE_ASSET, NodeType.ROUTE),
        (NodeType.INFRASTRUCTURE_ASSET, NodeType.TRANSIT_SERVICE),
    },
    EdgeType.OBSERVED_BY: {
        (NodeType.INFRASTRUCTURE_ASSET, NodeType.SENSOR),
        (NodeType.ROUTE, NodeType.SENSOR),
    },
}

REQUIRED_EDGE_ATTRS = [
    "relationship_type",
    "evidence_ids",
    "rule_or_model_version",
    "timestamp",
    "confidence",
    "explanation",
]


class GraphValidationError(Exception):
    """Raised when graph data violates semantic integrity rules."""
    pass


class GraphValidator:
    """
    Validates nodes, edges, topologies, and updates before modifying the evidence graph.
    """

    @staticmethod
    def validate_node(node_id: str, attributes: Dict[str, Any]) -> None:
        """Validates a single node's identifier and attributes."""
        if not node_id or not isinstance(node_id, str):
            raise GraphValidationError(f"Invalid node_id: '{node_id}'. Must be a non-empty string.")

        node_type = attributes.get("node_type")
        if not node_type:
            raise GraphValidationError(f"Node '{node_id}' is missing required attribute 'node_type'.")

        try:
            NodeType(node_type)
        except ValueError:
            valid_types = [t.value for t in NodeType]
            raise GraphValidationError(f"Node '{node_id}' has invalid node_type '{node_type}'. Valid: {valid_types}")

    @staticmethod
    def validate_edge(
        source_id: str,
        target_id: str,
        edge_attrs: Dict[str, Any],
        graph: nx.Graph
    ) -> None:
        """
        Validates edge endpoints existence, edge type semantics, and required metadata.
        """
        # 1. Endpoint existence
        if not graph.has_node(source_id):
            raise GraphValidationError(f"Edge source endpoint '{source_id}' does not exist in graph.")
        if not graph.has_node(target_id):
            raise GraphValidationError(f"Edge target endpoint '{target_id}' does not exist in graph.")

        # 2. Self-loop check (except where explicitly allowed)
        if source_id == target_id:
            raise GraphValidationError(f"Invalid self-loop edge: '{source_id}' -> '{target_id}'.")

        # 3. Relationship Type validation
        rel_type_str = edge_attrs.get("relationship_type")
        if not rel_type_str:
            raise GraphValidationError(f"Edge ({source_id}, {target_id}) is missing 'relationship_type'.")

        try:
            rel_type = EdgeType(rel_type_str)
        except ValueError:
            valid_rels = [r.value for r in EdgeType]
            raise GraphValidationError(f"Invalid relationship_type '{rel_type_str}'. Valid: {valid_rels}")

        # 4. Semantic Endpoint Compatibility
        src_type = NodeType(graph.nodes[source_id].get("node_type"))
        tgt_type = NodeType(graph.nodes[target_id].get("node_type"))
        allowed_pairs = PERMITTED_EDGE_TYPING.get(rel_type, set())

        # Check forward or reverse (for undirected/symmetric relationships)
        if (src_type, tgt_type) not in allowed_pairs and (tgt_type, src_type) not in allowed_pairs:
            raise GraphValidationError(
                f"Semantic Type Mismatch: Relationship '{rel_type.value}' cannot connect "
                f"source '{src_type.value}' ({source_id}) to target '{tgt_type.value}' ({target_id})."
            )

        # 5. Required Attribute Verification
        for req_attr in REQUIRED_EDGE_ATTRS:
            if req_attr not in edge_attrs:
                raise GraphValidationError(
                    f"Edge ({source_id}, {target_id}, {rel_type_str}) is missing required attribute: '{req_attr}'"
                )

        # 6. Confidence Score Bounds
        conf = edge_attrs.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            raise GraphValidationError(
                f"Edge ({source_id}, {target_id}) has invalid confidence '{conf}'. Must be in range [0.0, 1.0]."
            )

    @staticmethod
    def audit_entire_graph(graph: nx.Graph) -> Dict[str, Any]:
        """
        Audits all nodes, edges, isolated nodes, contradictions, and components.
        Returns a structured validation summary report.
        """
        issues: List[str] = []
        node_counts_by_type: Dict[str, int] = {}
        edge_counts_by_type: Dict[str, int] = {}
        contradictions_count = 0
        duplicate_edges_detected = 0

        # Audit Nodes
        for node_id, attrs in graph.nodes(data=True):
            try:
                GraphValidator.validate_node(node_id, attrs)
                ntype = attrs.get("node_type")
                node_counts_by_type[ntype] = node_counts_by_type.get(ntype, 0) + 1
            except GraphValidationError as e:
                issues.append(f"Node Error: {str(e)}")

        # Audit Edges
        seen_edges: Set[Tuple[str, str, str]] = set()
        for u, v, attrs in graph.edges(data=True):
            rel_type = attrs.get("relationship_type", "UNKNOWN")
            # In directed graphs (DiGraph), (u, v) is directional; reciprocal edges (u->v and v->u) are valid symmetries
            edge_key = (u, v, rel_type) if graph.is_directed() else (min(u, v), max(u, v), rel_type)

            if edge_key in seen_edges:
                duplicate_edges_detected += 1
                issues.append(f"Duplicate Edge: Multiple '{rel_type}' edges between {u} and {v}.")
            seen_edges.add(edge_key)

            try:
                GraphValidator.validate_edge(u, v, attrs, graph)
                edge_counts_by_type[rel_type] = edge_counts_by_type.get(rel_type, 0) + 1
                if rel_type == EdgeType.CONTRADICTS.value:
                    contradictions_count += 1
            except GraphValidationError as e:
                issues.append(f"Edge Error: {str(e)}")

        # Isolated Nodes
        isolated = list(nx.isolates(graph))
        if isolated:
            issues.append(f"Isolated Nodes Detected: {len(isolated)} nodes have 0 degree.")

        is_valid = len(issues) == 0

        return {
            "is_valid": is_valid,
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "node_counts_by_type": node_counts_by_type,
            "edge_counts_by_type": edge_counts_by_type,
            "isolated_nodes_count": len(isolated),
            "isolated_node_ids": isolated,
            "active_contradictions_count": contradictions_count,
            "duplicate_edges_count": duplicate_edges_detected,
            "validation_issues": issues,
        }
