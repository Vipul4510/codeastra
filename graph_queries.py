"""
UrbanPulse Graph Queries & Algorithms Module
=============================================
Implements BFS/DFS evidence traversal, contradiction isolation, reachable impact
exploration, and routing queries.

Core Invariants:
1. Shortest path algorithms are used strictly for spatial reachability or road alternative
   routing; graph paths are NEVER interpreted as physical causation.
2. BFS is used for layered evidence aggregation and multi-hop dependency exploration.
3. Repost/duplicate chains are traversed to consolidate redundant evidence into single chains.
4. Contradictions are explicitly surfaced for human operator review.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx

try:
    from .graph_validation import NodeType, EdgeType
    from .graph_builder import UrbanPulseEvidenceGraph
except (ImportError, ValueError):
    from graph_validation import NodeType, EdgeType
    from graph_builder import UrbanPulseEvidenceGraph


class UrbanPulseGraphQueries:
    """
    Query and analysis engine for the UrbanPulse heterogeneous evidence graph.
    """

    def __init__(self, evidence_graph: UrbanPulseEvidenceGraph):
        self.eg = evidence_graph
        self.g = evidence_graph.graph

    def get_incident_evidence(self, incident_id: str) -> Dict[str, Any]:
        """
        Retrieves all corroborating evidence supporting an incident using BFS traversal.
        Consolidates viral repost/duplicate chains so copies only count as 1 evidence chain.
        """
        if not self.g.has_node(incident_id):
            return {"error": f"Incident '{incident_id}' not found in graph."}

        supporting_evidence: List[Dict[str, Any]] = []
        visited_nodes: Set[str] = {incident_id}
        queue: List[str] = [incident_id]
        total_confidence_sum = 0.0
        chain_parent_map: Dict[str, str] = {}

        # 1. BFS traversal to discover directly or indirectly supporting evidence nodes
        while queue:
            curr = queue.pop(0)
            # Find incoming edges that support curr
            for pred in self.g.predecessors(curr):
                edge_data = self.g.get_edge_data(pred, curr)
                rel = edge_data.get("relationship_type")

                if rel in (EdgeType.SUPPORTS.value, EdgeType.SAME_INCIDENT_CANDIDATE.value):
                    if pred not in visited_nodes:
                        visited_nodes.add(pred)
                        queue.append(pred)
                        node_attrs = self.g.nodes[pred]
                        conf = edge_data.get("confidence", 0.5)
                        total_confidence_sum += conf

                        supporting_evidence.append({
                            "evidence_id": pred,
                            "node_type": node_attrs.get("node_type"),
                            "label": node_attrs.get("label"),
                            "locality": node_attrs.get("locality"),
                            "relationship_to_incident": rel,
                            "confidence": conf,
                            "explanation": edge_data.get("explanation"),
                            "timestamp": edge_data.get("timestamp")
                        })

        # 2. Trace DUPLICATE_OF chains among discovered evidence
        independent_chain_count = 0
        for ev in supporting_evidence:
            ev_id = ev["evidence_id"]
            # Check if this evidence is a duplicate of another discovered node
            is_copy = False
            for succ in self.g.successors(ev_id):
                if succ in visited_nodes and self.g[ev_id][succ].get("relationship_type") == EdgeType.DUPLICATE_OF.value:
                    is_copy = True
                    chain_parent_map[ev_id] = succ
                    break
            ev["is_repost_or_duplicate"] = is_copy
            if not is_copy:
                independent_chain_count += 1

        avg_conf = (total_confidence_sum / len(supporting_evidence)) if supporting_evidence else 0.0

        return {
            "incident_id": incident_id,
            "incident_attributes": self.g.nodes[incident_id],
            "total_evidence_nodes": len(supporting_evidence),
            "independent_evidence_chains": max(1, independent_chain_count) if supporting_evidence else 0,
            "mean_evidence_confidence": round(avg_conf, 3),
            "evidence_items": supporting_evidence
        }

    def get_incident_contradictions(self, incident_id: str) -> Dict[str, Any]:
        """
        Retrieves all active contradictory claims that conflict with the incident
        or any of its supporting evidence nodes.
        """
        if not self.g.has_node(incident_id):
            return {"error": f"Incident '{incident_id}' not found in graph."}

        # First collect incident node + its primary supporting evidence nodes
        ev_res = self.get_incident_evidence(incident_id)
        evidence_ids = {incident_id}
        for item in ev_res.get("evidence_items", []):
            evidence_ids.add(item["evidence_id"])

        contradictions: List[Dict[str, Any]] = []
        seen_contra_pairs: Set[Tuple[str, str]] = set()

        for node_id in evidence_ids:
            # Check all neighbors with CONTRADICTS relationship
            for neighbor in list(self.g.successors(node_id)) + list(self.g.predecessors(node_id)):
                edge_data = self.g.get_edge_data(node_id, neighbor) or self.g.get_edge_data(neighbor, node_id)
                if edge_data and edge_data.get("relationship_type") == EdgeType.CONTRADICTS.value:
                    pair_key = tuple(sorted([node_id, neighbor]))
                    if pair_key not in seen_contra_pairs:
                        seen_contra_pairs.add(pair_key)
                        contradictions.append({
                            "conflicting_node_a": node_id,
                            "conflicting_node_b": neighbor,
                            "node_b_type": self.g.nodes[neighbor].get("node_type"),
                            "node_b_label": self.g.nodes[neighbor].get("label"),
                            "node_b_text": self.g.nodes[neighbor].get("text_raw", ""),
                            "conflict_explanation": edge_data.get("explanation"),
                            "confidence": edge_data.get("confidence"),
                            "timestamp": edge_data.get("timestamp")
                        })

        return {
            "incident_id": incident_id,
            "has_contradictions": len(contradictions) > 0,
            "contradiction_count": len(contradictions),
            "conflicts": contradictions
        }

    def get_connected_infrastructure(self, incident_id: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Retrieves directly affected infrastructure assets and explores upstream/downstream
        dependencies via BFS up to max_hops along POSSIBLE_DEPENDENCY edges.
        """
        if not self.g.has_node(incident_id):
            return {"error": f"Incident '{incident_id}' not found in graph."}

        # 1. Identify directly affected assets (AFFECTS_ASSET)
        direct_assets: Set[str] = set()
        for succ in self.g.successors(incident_id):
            edge_data = self.g[incident_id][succ]
            if edge_data.get("relationship_type") == EdgeType.AFFECTS_ASSET.value:
                direct_assets.add(succ)

        # 2. BFS along POSSIBLE_DEPENDENCY edges
        all_infrastructure: List[Dict[str, Any]] = []
        visited_assets: Set[str] = set()
        queue: List[Tuple[str, int, str]] = [(ast, 1, "DIRECT_INCIDENT_IMPACT") for ast in direct_assets]

        for ast in direct_assets:
            visited_assets.add(ast)

        while queue:
            curr_asset, hop, reason = queue.pop(0)
            ast_attrs = self.g.nodes[curr_asset]

            all_infrastructure.append({
                "asset_id": curr_asset,
                "asset_type": ast_attrs.get("asset_type", "ASSET"),
                "label": ast_attrs.get("label"),
                "locality": ast_attrs.get("locality"),
                "hop_distance": hop,
                "connection_reason": reason
            })

            if hop < max_hops:
                # Explore outgoing dependency edges
                for next_asset in self.g.successors(curr_asset):
                    edge_data = self.g[curr_asset][next_asset]
                    if edge_data.get("relationship_type") == EdgeType.POSSIBLE_DEPENDENCY.value:
                        if next_asset not in visited_assets:
                            visited_assets.add(next_asset)
                            queue.append((
                                next_asset,
                                hop + 1,
                                f"DEPENDENCY: {curr_asset} -> {next_asset} ({edge_data.get('explanation')})"
                            ))

        return {
            "incident_id": incident_id,
            "direct_assets_count": len(direct_assets),
            "total_connected_assets": len(all_infrastructure),
            "max_hops_explored": max_hops,
            "connected_assets": all_infrastructure
        }

    def get_downstream_route_impacts(self, incident_id: str, max_hops: int = 3) -> Dict[str, Any]:
        """
        Explores reachable downstream mobility disruptions (routes, transit corridors)
        using directed BFS traversal.
        """
        if not self.g.has_node(incident_id):
            return {"error": f"Incident '{incident_id}' not found in graph."}

        impacted_corridors: List[Dict[str, Any]] = []
        visited_nodes: Set[str] = {incident_id}
        queue: List[Tuple[str, int, float, str]] = [(incident_id, 0, 1.0, "INCIDENT_ORIGIN")]

        while queue:
            curr_node, hop, curr_conf, path_summary = queue.pop(0)

            # Check outgoing edges
            for succ in self.g.successors(curr_node):
                edge_data = self.g[curr_node][succ]
                rel = edge_data.get("relationship_type")
                edge_conf = edge_data.get("confidence", 0.5)
                succ_type = self.g.nodes[succ].get("node_type")

                # If edge is an impact on route or transit service
                if rel == EdgeType.IMPACTS_ROUTE.value or succ_type in (NodeType.ROUTE.value, NodeType.TRANSIT_SERVICE.value):
                    cum_conf = round(curr_conf * edge_conf, 3)
                    node_attrs = self.g.nodes[succ]
                    impacted_corridors.append({
                        "route_id": succ,
                        "node_type": succ_type,
                        "label": node_attrs.get("label"),
                        "origin_destination": f"{node_attrs.get('origin', '')} -> {node_attrs.get('destination', '')}",
                        "impact_type": edge_data.get("impact_type", "BLOCKAGE"),
                        "path_distance_hops": hop + 1,
                        "pathway": f"{path_summary} -> [{rel}] -> {succ}",
                        "cumulative_impact_confidence": cum_conf,
                        "explanation": edge_data.get("explanation")
                    })

                # If intermediate node is an asset, continue traversal
                elif rel in (EdgeType.AFFECTS_ASSET.value, EdgeType.POSSIBLE_DEPENDENCY.value):
                    if succ not in visited_nodes and hop < max_hops:
                        visited_nodes.add(succ)
                        queue.append((
                            succ,
                            hop + 1,
                            curr_conf * edge_conf,
                            f"{path_summary} -> {succ}"
                        ))

        return {
            "incident_id": incident_id,
            "total_impacted_routes": len(impacted_corridors),
            "impacted_corridors": impacted_corridors
        }

    def find_safe_alternative_route(
        self,
        start_node: str,
        end_node: str,
        blocked_nodes: Set[str],
        road_network_graph: Optional[nx.Graph] = None
    ) -> Dict[str, Any]:
        """
        Demonstrates valid shortest-path querying on a spatial routing graph avoiding blocked nodes.
        Explicitly enforces Requirement 7:
        "Use shortest-path algorithms only for valid graph questions; do not interpret a graph path as proof of causation."
        """
        routing_g = road_network_graph or self.eg.get_impact_digraph().to_undirected()

        # Remove blocked nodes from evaluation view
        sub_graph = routing_g.copy()
        for b_node in blocked_nodes:
            if sub_graph.has_node(b_node):
                sub_graph.remove_node(b_node)

        causation_disclaimer = (
            "NOTICE ON CAUSATION: Shortest path indicates spatial topological reachability avoiding "
            "blocked nodes. It does NOT assert or establish physical causal chains."
        )

        if not sub_graph.has_node(start_node) or not sub_graph.has_node(end_node):
            return {
                "route_found": False,
                "error": f"Start '{start_node}' or Destination '{end_node}' is blocked or not in graph.",
                "causation_disclaimer": causation_disclaimer
            }

        try:
            path = nx.shortest_path(sub_graph, source=start_node, target=end_node, weight="weight")
            length = len(path) - 1
            return {
                "route_found": True,
                "start": start_node,
                "destination": end_node,
                "avoided_nodes": list(blocked_nodes),
                "path_nodes": path,
                "path_hops": length,
                "causation_disclaimer": causation_disclaimer
            }
        except nx.NetworkXNoPath:
            return {
                "route_found": False,
                "error": f"No available alternative route between '{start_node}' and '{end_node}'.",
                "causation_disclaimer": causation_disclaimer
            }
