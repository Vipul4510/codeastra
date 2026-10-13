"""
UrbanPulse Multimodal Deduplication Engine
Implements exact record deduplication and social media repost/copy detection.
Architecturally enforces strict separation between duplicate detection and same-incident correlation.
"""

import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any

try:
    from .config import REPOST_SIMILARITY_THRESHOLD, REPOST_TIME_WINDOW_HOURS
    from .normalizers import TextNormalizer
except (ImportError, ValueError):
    from config import REPOST_SIMILARITY_THRESHOLD, REPOST_TIME_WINDOW_HOURS
    from normalizers import TextNormalizer


def compute_token_jaccard(text_a: str, text_b: str) -> float:
    """
    Computes Jaccard word-token similarity between two normalized strings.
    """
    if not text_a or not text_b:
        return 0.0
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return round(intersection / union, 4) if union > 0 else 0.0


def compute_char_ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """
    Computes character n-gram Jaccard similarity to catch subtle typos or minor edits.
    """
    if not text_a or not text_b:
        return 0.0
    if len(text_a) < n or len(text_b) < n:
        return 1.0 if text_a == text_b else 0.0
    
    ngrams_a = {text_a[i:i+n] for i in range(len(text_a) - n + 1)}
    ngrams_b = {text_b[i:i+n] for i in range(len(text_b) - n + 1)}
    
    intersection = len(ngrams_a.intersection(ngrams_b))
    union = len(ngrams_a.union(ngrams_b))
    return round(intersection / union, 4) if union > 0 else 0.0


class ExactDuplicateDetector:
    """
    Identifies exact duplicates using:
    1. Unique source record identifiers (e.g. platform_post_id, external record_id).
    2. Deterministic content fingerprints (source_type, normalized_text, locality, timestamp_minute).
    """

    def __init__(self):
        self.seen_source_ids: Dict[str, str] = {}  # source_id -> first_event_id
        self.seen_content_hashes: Dict[str, str] = {}  # hash -> first_event_id

    def check_and_register(
        self,
        event_id: str,
        source_record_id: Optional[str],
        source_type: str,
        normalized_text: str,
        locality: str,
        timestamp_iso: Optional[str],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Returns:
            (is_exact_duplicate, duplicate_of_id, duplicate_type)
        """
        # 1. Check exact source identifier if provided
        if source_record_id and str(source_record_id).strip():
            src_key = f"{source_type}::{str(source_record_id).strip()}"
            if src_key in self.seen_source_ids:
                return True, self.seen_source_ids[src_key], "EXACT_SOURCE_ID"
            self.seen_source_ids[src_key] = event_id

        # 2. Check exact content fingerprint
        if normalized_text and len(normalized_text) > 5 and timestamp_iso:
            # Bucket timestamp to 5-minute intervals to catch duplicate telemetry or identical bursts
            ts_bucket = timestamp_iso[:16]  # e.g. "2026-10-12T14:30"
            raw_fingerprint = f"{source_type}|{locality.lower()}|{ts_bucket}|{normalized_text}"
            fp_hash = hashlib.md5(raw_fingerprint.encode("utf-8")).hexdigest()

            if fp_hash in self.seen_content_hashes:
                return True, self.seen_content_hashes[fp_hash], "EXACT_CONTENT_FINGERPRINT"
            self.seen_content_hashes[fp_hash] = event_id

        return False, None, None


class SocialRepostDetector:
    """
    Detects potential copied or reposted social media content.
    Uses normalized text similarity (token & character n-gram Jaccard) within a temporal window.
    Preserves original post provenance, publication time, and platform metadata.
    """

    def __init__(
        self,
        similarity_threshold: float = REPOST_SIMILARITY_THRESHOLD,
        time_window_hours: float = REPOST_TIME_WINDOW_HOURS
    ):
        self.threshold = similarity_threshold
        self.time_window_ms = int(time_window_hours * 3600 * 1000)
        # Inverted index of recent posts by locality or global
        # Structure: list of dicts: {"post_id", "epoch_ms", "normalized_text"}
        self.post_history: List[Dict[str, Any]] = []

    def check_and_register(
        self,
        post_id: str,
        raw_text: str,
        epoch_ms: Optional[int],
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Returns:
            (is_repost_or_copy, repost_parent_id, similarity_score)
        """
        norm_text = TextNormalizer.normalize_for_matching(raw_text)
        if not norm_text or len(norm_text) < 10 or epoch_ms is None:
            # Retain in history if has text and timestamp
            if norm_text and epoch_ms:
                self.post_history.append({
                    "post_id": post_id,
                    "epoch_ms": epoch_ms,
                    "normalized_text": norm_text
                })
            return False, None, None

        # Compare against historical posts within temporal window
        best_parent_id = None
        best_score = 0.0

        # Scan recent posts (scanned backwards for recency)
        for prior_post in reversed(self.post_history):
            time_delta = epoch_ms - prior_post["epoch_ms"]
            # Check if prior post is older and within window
            if 0 < time_delta <= self.time_window_ms:
                prior_text = prior_post["normalized_text"]
                
                # Fast equality check
                if norm_text == prior_text:
                    best_parent_id = prior_post["post_id"]
                    best_score = 1.0
                    break
                
                # Token Jaccard check
                sim = compute_token_jaccard(norm_text, prior_text)
                if sim >= self.threshold:
                    if sim > best_score:
                        best_score = sim
                        best_parent_id = prior_post["post_id"]
                elif sim >= 0.65:
                    # Near miss: check character n-grams to catch minor character edits / additions
                    char_sim = compute_char_ngram_jaccard(norm_text, prior_text)
                    if char_sim >= self.threshold and char_sim > best_score:
                        best_score = char_sim
                        best_parent_id = prior_post["post_id"]

        # Register current post into history
        self.post_history.append({
            "post_id": post_id,
            "epoch_ms": epoch_ms,
            "normalized_text": norm_text
        })

        if best_parent_id and best_score >= self.threshold:
            return True, best_parent_id, best_score

        return False, None, None


class IndependenceGuard:
    """
    Enforces architectural separation:
    Asserts that duplicate detection is strictly distinct from same-incident correlation.
    """

    @staticmethod
    def verify_separation(
        event_a_id: str,
        event_b_id: str,
        is_duplicate: bool,
        is_same_incident: bool,
        rationale: str = ""
    ) -> Dict[str, Any]:
        """
        Validates relationship semantics:
        - (is_duplicate=True, is_same_incident=True): Redundant copies of the same report.
        - (is_duplicate=False, is_same_incident=True): Two DIFFERENT reports describing the SAME INCIDENT.
        - (is_duplicate=False, is_same_incident=False): Independent reports describing DIFFERENT incidents.
        - (is_duplicate=True, is_same_incident=False): Semantic contradiction / invalid state.
        """
        is_valid = True
        violation_reason = None

        if is_duplicate and not is_same_incident:
            is_valid = False
            violation_reason = (
                f"Contradiction: Event {event_a_id} and {event_b_id} flagged as duplicate, "
                "but not assigned to same incident."
            )

        return {
            "event_a_id": event_a_id,
            "event_b_id": event_b_id,
            "is_duplicate": is_duplicate,
            "is_same_incident": is_same_incident,
            "is_semantically_valid": is_valid,
            "violation_reason": violation_reason,
            "separation_mode": (
                "COPIED_SIGNAL" if is_duplicate
                else ("CORROBORATING_MULTIMODAL_EVIDENCE" if is_same_incident else "INDEPENDENT_EVENTS")
            )
        }
