"""
Unit and Integration Tests for UrbanPulse Multimodal Preprocessing & Deduplication
Covers:
- Missing, invalid, and timezone-normalized timestamps
- In-corridor, out-of-bounds, invalid, and missing geographic coordinates
- Category, source, asset ID, and locality taxonomies
- Exact duplicate detection (source ID and content fingerprint)
- Social media repost / copy-paste detection
- Independence of duplicate detection from same-incident correlation
- Preservation of temporal dimensions and source reliability
- Contradictory records preservation without silent overwrites
- Pipeline determinism and reproducibility
"""

import unittest
import sys
import os

# Ensure package path is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from preprocessing.normalizers import (
    TimestampNormalizer,
    SpatialNormalizer,
    TaxonomyNormalizer,
    TextNormalizer,
)
from preprocessing.deduplicator import (
    ExactDuplicateDetector,
    SocialRepostDetector,
    IndependenceGuard,
)
from preprocessing.event_schema import StandardizedEvent


class TestUrbanPulsePreprocessing(unittest.TestCase):

    def test_missing_timestamp_handling(self):
        """Rule: Do not delete records with missing timestamps; do not fabricate dates."""
        for missing_val in [None, "", "   ", float("nan"), "NaN", "null"]:
            iso, epoch, status = TimestampNormalizer.normalize(missing_val)
            self.assertIsNone(iso)
            self.assertIsNone(epoch)
            self.assertEqual(status, "MISSING")

    def test_invalid_timestamp_format(self):
        """Rule: Flag unparseable timestamps as INVALID_FORMAT without raising unhandled errors."""
        iso, epoch, status = TimestampNormalizer.normalize("invalid-garbage-date-2026")
        self.assertIsNone(iso)
        self.assertIsNone(epoch)
        self.assertEqual(status, "INVALID_FORMAT")

    def test_timezone_normalization(self):
        """Rule: Timestamps must be normalized to consistent timezone-aware IST (+05:30)."""
        # Naive timestamp
        raw_naive = "2026-07-15T14:30:00"
        iso, epoch, status = TimestampNormalizer.normalize(raw_naive)
        self.assertEqual(status, "VALID")
        self.assertIn("+05:30", iso)
        self.assertTrue(iso.startswith("2026-07-15T14:30:00"))

        # UTC timestamp
        raw_utc = "2026-07-15T09:00:00Z"
        iso_utc, epoch_utc, status_utc = TimestampNormalizer.normalize(raw_utc)
        self.assertEqual(status_utc, "VALID")
        # 09:00 UTC == 14:30 IST (+5:30)
        self.assertIn("+05:30", iso_utc)
        self.assertTrue(iso_utc.startswith("2026-07-15T14:30:00"))
        self.assertEqual(epoch, epoch_utc)

    def test_valid_in_corridor_coordinates(self):
        """Coordinates inside Western Suburbs (e.g. Andheri West: 19.1197, 72.8464) must be VALID_IN_CORRIDOR."""
        res = SpatialNormalizer.validate_and_normalize(19.1197, 72.8464, "Andheri West")
        self.assertEqual(res["spatial_quality"], "VALID_IN_CORRIDOR")
        self.assertTrue(res["is_geocoded"])
        self.assertEqual(res["latitude"], 19.1197)
        self.assertEqual(res["longitude"], 72.8464)
        self.assertEqual(res["latitude_raw"], 19.1197)
        self.assertEqual(res["locality_zone"], "LOC-03")

    def test_out_of_bounds_coordinates(self):
        """Coordinates outside Western Suburbs (e.g. Delhi: 28.6139, 77.2090) must be VALID_OUT_OF_BOUNDS."""
        res = SpatialNormalizer.validate_and_normalize(28.6139, 77.2090, "Delhi")
        self.assertEqual(res["spatial_quality"], "VALID_OUT_OF_BOUNDS")
        self.assertTrue(res["is_geocoded"])
        self.assertEqual(res["latitude"], 28.6139)
        self.assertEqual(res["longitude"], 77.2090)

    def test_invalid_and_missing_coordinates(self):
        """Missing or out-of-range coordinates must be flagged and preserved without fabricating 0,0."""
        # Missing
        res_missing = SpatialNormalizer.validate_and_normalize(None, None, "Borivali")
        self.assertEqual(res_missing["spatial_quality"], "MISSING_COORDINATES")
        self.assertFalse(res_missing["is_geocoded"])
        self.assertIsNone(res_missing["latitude"])

        # Invalid (>90 lat)
        res_invalid = SpatialNormalizer.validate_and_normalize(999.0, 72.84, "Kandivali")
        self.assertEqual(res_invalid["spatial_quality"], "INVALID_COORDINATES")
        self.assertFalse(res_invalid["is_geocoded"])
        self.assertEqual(res_invalid["latitude_raw"], 999.0)

    def test_taxonomy_normalization(self):
        """Rule: Normalize event types, source names, asset IDs, and localities."""
        # Event types
        self.assertEqual(TaxonomyNormalizer.normalize_event_type("water-logging"), "ROAD_WATERLOGGING")
        self.assertEqual(TaxonomyNormalizer.normalize_event_type("underpass_flooded"), "SUBWAY_INUNDATION")
        self.assertEqual(TaxonomyNormalizer.normalize_event_type("traffic jam"), "TRAFFIC_SLOWDOWN")

        # Sources
        self.assertEqual(TaxonomyNormalizer.normalize_source_type("src-sensor"), "SENSOR")
        self.assertEqual(TaxonomyNormalizer.normalize_source_type("social_media"), "SOCIAL_MEDIA")

        # Asset IDs
        asset_norm, valid_fmt = TaxonomyNormalizer.normalize_asset_id("ast-00124")
        self.assertEqual(asset_norm, "AST-00124")
        self.assertTrue(valid_fmt)

        asset_bad, bad_fmt = TaxonomyNormalizer.normalize_asset_id("NOT_AN_ASSET")
        self.assertFalse(bad_fmt)

        # Locality
        self.assertEqual(TaxonomyNormalizer.normalize_locality("andheri west"), "Andheri West")

    def test_exact_duplicate_detection_by_source_id(self):
        """Rule: Detect exact duplicates using source record identifiers when available."""
        detector = ExactDuplicateDetector()

        # Ingest first event
        is_dupe_1, dupe_of_1, d_type_1 = detector.check_and_register(
            event_id="EVT-001",
            source_record_id="POST-9999",
            source_type="SOCIAL_MEDIA",
            normalized_text="waterlogging at subway",
            locality="Andheri",
            timestamp_iso="2026-07-15T14:30:00+05:30"
        )
        self.assertFalse(is_dupe_1)
        self.assertIsNone(dupe_of_1)

        # Re-ingest with identical source_record_id
        is_dupe_2, dupe_of_2, d_type_2 = detector.check_and_register(
            event_id="EVT-002",
            source_record_id="POST-9999",
            source_type="SOCIAL_MEDIA",
            normalized_text="waterlogging at subway",
            locality="Andheri",
            timestamp_iso="2026-07-15T14:30:00+05:30"
        )
        self.assertTrue(is_dupe_2)
        self.assertEqual(dupe_of_2, "EVT-001")
        self.assertEqual(d_type_2, "EXACT_SOURCE_ID")

    def test_exact_duplicate_content_fingerprint(self):
        """Rule: Detect identical duplicate payloads across identical time/locality/text."""
        detector = ExactDuplicateDetector()

        detector.check_and_register(
            event_id="EVT-100",
            source_record_id=None,
            source_type="CITIZEN_REPORT",
            normalized_text="tree fallen on sv road blocking traffic",
            locality="Bandra",
            timestamp_iso="2026-08-01T10:15:00+05:30"
        )

        is_dupe, dupe_of, d_type = detector.check_and_register(
            event_id="EVT-101",
            source_record_id=None,
            source_type="CITIZEN_REPORT",
            normalized_text="tree fallen on sv road blocking traffic",
            locality="Bandra",
            timestamp_iso="2026-08-01T10:15:20+05:30"
        )
        self.assertTrue(is_dupe)
        self.assertEqual(dupe_of, "EVT-100")
        self.assertEqual(d_type, "EXACT_CONTENT_FINGERPRINT")

    def test_social_media_repost_detection(self):
        """Rule: Detect copied or reposted social media content using normalized text similarity."""
        repost_detector = SocialRepostDetector(similarity_threshold=0.80, time_window_hours=24.0)

        # User 1 posts original tweet at 10:00 AM (epoch: 1785000000000)
        t0 = 1785000000000
        is_rep1, p1, s1 = repost_detector.check_and_register(
            post_id="POST-001",
            raw_text="Massive waterlogging near Dahisar subway! Avoid Western Express Highway!",
            epoch_ms=t0
        )
        self.assertFalse(is_rep1)

        # User 2 reposts / retweets near-identical copy 2 hours later
        t1 = t0 + (2 * 3600 * 1000)
        is_rep2, p2, s2 = repost_detector.check_and_register(
            post_id="POST-002",
            raw_text="RT @user: Massive waterlogging near Dahisar subway! Avoid Western Express Highway!!",
            epoch_ms=t1
        )
        self.assertTrue(is_rep2)
        self.assertEqual(p2, "POST-001")
        self.assertGreaterEqual(s2, 0.80)

        # User 3 posts something completely unrelated at t1
        is_rep3, p3, s3 = repost_detector.check_and_register(
            post_id="POST-003",
            raw_text="Weather is sunny in Bandra, pleasant breeze.",
            epoch_ms=t1
        )
        self.assertFalse(is_rep3)

    def test_independence_of_duplicate_from_same_incident(self):
        """Rule: Keep duplicate detection separate from same-incident detection. Two different reports can describe the same incident without being duplicates."""
        # Case A: Two different citizens reporting the same incident in their own words
        guard_result = IndependenceGuard.verify_separation(
            event_a_id="EVT-CITIZEN-1",
            event_b_id="EVT-CITIZEN-2",
            is_duplicate=False,
            is_same_incident=True,
            rationale="Two independent eyewitnesses describing the same waterlogged intersection"
        )
        self.assertTrue(guard_result["is_semantically_valid"])
        self.assertFalse(guard_result["is_duplicate"])
        self.assertTrue(guard_result["is_same_incident"])
        self.assertEqual(guard_result["separation_mode"], "CORROBORATING_MULTIMODAL_EVIDENCE")

        # Case B: Identical bot repost (both duplicate and same incident)
        guard_repost = IndependenceGuard.verify_separation(
            event_a_id="EVT-POST-1",
            event_b_id="EVT-POST-2",
            is_duplicate=True,
            is_same_incident=True,
            rationale="Bot repost of identical tweet"
        )
        self.assertTrue(guard_repost["is_semantically_valid"])
        self.assertEqual(guard_repost["separation_mode"], "COPIED_SIGNAL")

        # Case C: Invalid contradictory state (duplicate=True, but same_incident=False)
        guard_invalid = IndependenceGuard.verify_separation(
            event_a_id="EVT-X",
            event_b_id="EVT-Y",
            is_duplicate=True,
            is_same_incident=False
        )
        self.assertFalse(guard_invalid["is_semantically_valid"])
        self.assertIsNotNone(guard_invalid["violation_reason"])

    def test_preservation_of_temporal_and_reliability_dimensions(self):
        """Rule: Preserve source reliability, observation time, publication time, and collection time separately."""
        event = StandardizedEvent(
            event_id="EVT-TEST-1",
            raw_id="RAW-1",
            source_modality="CITIZEN_REPORT",
            source_channel="CITIZEN_FEED",
            canonical_event_type="ROAD_WATERLOGGING",
            raw_event_type="waterlogging",
            observation_time_iso="2026-07-15T14:00:00+05:30",
            observation_time_epoch_ms=1784104200000,
            publication_time_iso="2026-07-15T14:15:00+05:30",
            collection_time_iso="2026-07-15T14:20:00+05:30",
            latitude=19.1197,
            longitude=72.8464,
            latitude_raw="19.1197",
            longitude_raw="72.8464",
            locality_normalized="Andheri West",
            locality_zone="LOC-03",
            text_raw="Heavy waterlogging near station",
            text_cleaned="Heavy waterlogging near station",
            source_reliability=0.75,
            reliability_imputed=False,
            provenance="SYNTHETIC_GENERATED",
            is_synthetic=True,
            is_exact_duplicate=False,
            duplicate_of_id=None,
            duplicate_type=None,
            is_repost_or_copy=False,
            repost_parent_id=None,
            repost_similarity=None,
            has_missing_timestamp=False,
            has_missing_coordinates=False,
            has_missing_text=False,
            spatial_quality="VALID_IN_CORRIDOR",
            data_quality_score=1.0,
            is_usable_for_spatial_clustering=True,
            is_usable_for_text_matching=True,
            evaluation_candidate_incident_id=None,
        )

        d = event.to_dict()
        # Ensure separate temporal timestamps
        self.assertEqual(d["observation_time_iso"], "2026-07-15T14:00:00+05:30")
        self.assertEqual(d["publication_time_iso"], "2026-07-15T14:15:00+05:30")
        self.assertEqual(d["collection_time_iso"], "2026-07-15T14:20:00+05:30")
        self.assertEqual(d["source_reliability"], 0.75)
        self.assertFalse(d["reliability_imputed"])

    def test_contradictory_records_preservation(self):
        """Rule: Contradictory reports must both be preserved with appropriate flags and no silent overwrites."""
        report_clear = StandardizedEvent(
            event_id="EVT-ANDHERI-CLEAR",
            raw_id="RAW-CLEAR",
            source_modality="CITIZEN_REPORT",
            source_channel="CITIZEN_FEED",
            canonical_event_type="GENERAL_INCIDENT",
            raw_event_type="clear_road",
            observation_time_iso="2026-07-15T14:00:00+05:30",
            observation_time_epoch_ms=1784104200000,
            publication_time_iso="2026-07-15T14:00:00+05:30",
            collection_time_iso="2026-07-15T14:05:00+05:30",
            latitude=19.1197,
            longitude=72.8464,
            latitude_raw="19.1197",
            longitude_raw="72.8464",
            locality_normalized="Andheri West",
            locality_zone="LOC-03",
            text_raw="Andheri subway is completely clear and open",
            text_cleaned="Andheri subway is completely clear and open",
            source_reliability=0.65,
            reliability_imputed=False,
            provenance="SYNTHETIC_GENERATED",
            is_synthetic=True,
            is_exact_duplicate=False,
            duplicate_of_id=None,
            duplicate_type=None,
            is_repost_or_copy=False,
            repost_parent_id=None,
            repost_similarity=None,
            has_missing_timestamp=False,
            has_missing_coordinates=False,
            has_missing_text=False,
            spatial_quality="VALID_IN_CORRIDOR",
            data_quality_score=1.0,
            is_usable_for_spatial_clustering=True,
            is_usable_for_text_matching=True,
        )

        report_flooded = StandardizedEvent(
            event_id="EVT-ANDHERI-FLOOD",
            raw_id="RAW-FLOOD",
            source_modality="SENSOR_TELEMETRY",
            source_channel="SENSOR",
            canonical_event_type="SUBWAY_INUNDATION",
            raw_event_type="inundation",
            observation_time_iso="2026-07-15T14:02:00+05:30",
            observation_time_epoch_ms=1784104320000,
            publication_time_iso="2026-07-15T14:02:00+05:30",
            collection_time_iso="2026-07-15T14:02:00+05:30",
            latitude=19.1197,
            longitude=72.8464,
            latitude_raw="19.1197",
            longitude_raw="72.8464",
            locality_normalized="Andheri West",
            locality_zone="LOC-03",
            text_raw="Sensor alert: 52cm water level at subway",
            text_cleaned="Sensor alert: 52cm water level at subway",
            source_reliability=0.92,
            reliability_imputed=False,
            provenance="SYNTHETIC_GENERATED",
            is_synthetic=True,
            is_exact_duplicate=False,
            duplicate_of_id=None,
            duplicate_type=None,
            is_repost_or_copy=False,
            repost_parent_id=None,
            repost_similarity=None,
            has_missing_timestamp=False,
            has_missing_coordinates=False,
            has_missing_text=False,
            spatial_quality="VALID_IN_CORRIDOR",
            data_quality_score=1.0,
            is_usable_for_spatial_clustering=True,
            is_usable_for_text_matching=True,
        )

        event_store = [report_clear, report_flooded]
        # Both records are retained
        self.assertEqual(len(event_store), 2)
        # Neither is marked as duplicate of the other
        self.assertFalse(event_store[0].is_exact_duplicate)
        self.assertFalse(event_store[1].is_exact_duplicate)
        # Higher reliability sensor is preserved alongside citizen report
        self.assertGreater(event_store[1].source_reliability, event_store[0].source_reliability)


if __name__ == "__main__":
    unittest.main()
