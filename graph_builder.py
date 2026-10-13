"""
UrbanPulse Graph Builder Module
================================
Constructs and maintains the heterogeneous multimodal evidence graph using NetworkX.

Key Design Principles:
1. Validates all node identifiers and edge endpoints prior to graph insertion.
2. Distinctly separates Identity Relations from Physical Impact and Dependency Relations.
3. Attaches rich auditing metadata to every edge (rule/model version, evidence IDs, confidence, rationale).
4. Strictly disallows dangling nodes, invalid references, and ungrounded dependencies.
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx

try:
    from .graph_validation import (
        NodeType,
        EdgeType,
        GraphValidator,
        GraphValidationError
    )
except (ImportError, ValueError):
    from graph_validation import (
        NodeType,
        EdgeType,
        GraphValidator,
        GraphValidationError
    )


class UrbanPulseEvidenceGraph:
    """
    Core heterogeneous evidence graph storing multimodal observations,
    incident identities, infrastructure assets, sensors, and route impacts.
    """

    def __init__(self, name: str = "UrbanPulse_Evidence_Graph"):
        self.name = name
        # Base DiGraph storing all directional dependencies and undirected identity edges (modeled bidirectionally)
        self.graph = nx.DiGraph()
        self.metadata: Dict[str, Any] = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            "version": "1.0.0",
            "corridor": "Mumbai Western Suburbs (Bandra to Dahisar)"
        }

    def add_node(
        self,
        node_id: str,
        node_type: NodeType | str,
        label: Optional[str] = None,
        locality: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        timestamp: Optional[str] = None,
        **extra_attrs
    ) -> None:
        """
        Adds a validated node to the evidence graph.
        """
        try:
            val_node_type = NodeType(node_type).value
        except (ValueError, TypeError):
            valid_types = [t.value for t in NodeType]
            raise GraphValidationError(f"Invalid node_type: '{node_type}'. Valid types: {valid_types}")

        attrs = {
            "node_type": val_node_type,
            "label": label or node_id,
            "locality": locality or "Western Suburbs",
            "latitude": float(latitude) if latitude is not None else None,
            "longitude": float(longitude) if longitude is not None else None,
            "timestamp": timestamp,
            **extra_attrs
        }
        # Validate schema before insertion
        GraphValidator.validate_node(node_id, attrs)
        self.graph.add_node(node_id, **attrs)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: EdgeType | str,
        confidence: float,
        evidence_ids: List[str],
        rule_or_model_version: str,
        explanation: str,
        timestamp: Optional[str] = None,
        **extra_attrs
    ) -> None:
        """
        Adds a validated relationship edge between two existing nodes.
        """
        try:
            rel_type = EdgeType(relationship_type).value
        except (ValueError, TypeError):
            valid_rels = [r.value for r in EdgeType]
            raise GraphValidationError(f"Invalid relationship_type: '{relationship_type}'. Valid types: {valid_rels}")

        ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%S+05:30")

        # Distinguish identity from impact relationships
        is_identity = rel_type in (
            EdgeType.SAME_INCIDENT_CANDIDATE.value,
            EdgeType.DUPLICATE_OF.value,
            EdgeType.CONTRADICTS.value
        )
        is_physical_impact = rel_type in (
            EdgeType.AFFECTS_ASSET.value,
            EdgeType.POSSIBLE_DEPENDENCY.value,
            EdgeType.IMPACTS_ROUTE.value
        )

        edge_attrs = {
            "relationship_type": rel_type,
            "confidence": float(confidence),
            "evidence_ids": list(evidence_ids),
            "rule_or_model_version": rule_or_model_version,
            "explanation": explanation,
            "timestamp": ts,
            "is_identity_relation": is_identity,
            "is_physical_impact": is_physical_impact,
            **extra_attrs
        }

        # Validate endpoints and semantics
        GraphValidator.validate_edge(source_id, target_id, edge_attrs, self.graph)

        self.graph.add_edge(source_id, target_id, **edge_attrs)

        # For symmetric identity relations (e.g., SAME_INCIDENT_CANDIDATE, CONTRADICTS), add reciprocal edge
        if rel_type in (EdgeType.SAME_INCIDENT_CANDIDATE.value, EdgeType.CONTRADICTS.value):
            if not self.graph.has_edge(target_id, source_id):
                rev_attrs = dict(edge_attrs)
                self.graph.add_edge(target_id, source_id, **rev_attrs)

    def update_edge(
        self,
        source_id: str,
        target_id: str,
        confidence: Optional[float] = None,
        explanation: Optional[str] = None,
        rule_or_model_version: Optional[str] = None
    ) -> None:
        """
        Updates an existing edge's confidence or explanation with validation.
        """
        if not self.graph.has_edge(source_id, target_id):
            raise GraphValidationError(f"Edge ({source_id}, {target_id}) does not exist for update.")

        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise GraphValidationError(f"Invalid confidence: {confidence}. Must be in [0.0, 1.0].")
            self.graph[source_id][target_id]["confidence"] = float(confidence)

        if explanation is not None:
            self.graph[source_id][target_id]["explanation"] = explanation

        if rule_or_model_version is not None:
            self.graph[source_id][target_id]["rule_or_model_version"] = rule_or_model_version

        self.graph[source_id][target_id]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S+05:30")

    def get_identity_subgraph(self) -> nx.Graph:
        """
        Extracts undirected subgraph of identity relationships (SAME_INCIDENT_CANDIDATE)
        specifically suitable for connected components analysis.
        """
        id_graph = nx.Graph()
        for u, v, d in self.graph.edges(data=True):
            if d.get("relationship_type") == EdgeType.SAME_INCIDENT_CANDIDATE.value:
                id_graph.add_node(u, **self.graph.nodes[u])
                id_graph.add_node(v, **self.graph.nodes[v])
                id_graph.add_edge(u, v, **d)
        return id_graph

    def get_impact_digraph(self) -> nx.DiGraph:
        """
        Extracts directed acyclic/cyclic subgraph of physical impact and routing dependencies.
        """
        imp_graph = nx.DiGraph()
        for u, v, d in self.graph.edges(data=True):
            if d.get("is_physical_impact"):
                imp_graph.add_node(u, **self.graph.nodes[u])
                imp_graph.add_node(v, **self.graph.nodes[v])
                imp_graph.add_edge(u, v, **d)
        return imp_graph

    def to_dict(self) -> Dict[str, Any]:
        """Serializes graph nodes, edges, and metadata to JSON-compatible dictionary."""
        nodes_list = []
        for n_id, attrs in self.graph.nodes(data=True):
            nodes_list.append({"node_id": n_id, **attrs})

        edges_list = []
        for u, v, attrs in self.graph.edges(data=True):
            edges_list.append({
                "source": u,
                "target": v,
                **attrs
            })

        return {
            "metadata": self.metadata,
            "nodes": nodes_list,
            "edges": edges_list
        }

    def save_to_json(self, filepath: str) -> None:
        """Saves serialized graph dictionary to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "UrbanPulseEvidenceGraph":
        """Loads and re-validates a graph from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        eg = cls()
        eg.metadata = data.get("metadata", {})

        for n in data.get("nodes", []):
            n_id = n["node_id"]
            attrs = {k: v for k, v in n.items() if k != "node_id"}
            eg.add_node(n_id, **attrs)

        for e in data.get("edges", []):
            src = e["source"]
            tgt = e["target"]
            attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
            if eg.graph.has_edge(src, tgt):
                continue
            eg.add_edge(src, tgt, **attrs)

        # Audit loaded graph
        audit = GraphValidator.audit_entire_graph(eg.graph)
        if not audit["is_valid"]:
            raise GraphValidationError(f"Loaded graph failed audit: {audit['validation_issues']}")

        return eg
