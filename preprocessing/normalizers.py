"""
UrbanPulse Multimodal Normalization Engine
Normalizes timestamps, coordinates, categories, asset IDs, localities, and text
without silently fabricating missing data.
"""

import re
import datetime
from typing import Optional, Tuple, Dict, Any

try:
    from .config import (
        DEFAULT_TIMEZONE_STR,
        MUMBAI_WESTERN_SUBURBS_BBOX,
        LOCALITY_TO_ZONE_MAP,
        CANONICAL_EVENT_TYPE_MAP,
        CANONICAL_SOURCE_MAP,
        DEFAULT_SOURCE_RELIABILITY,
    )
except (ImportError, ValueError):
    from config import (
        DEFAULT_TIMEZONE_STR,
        MUMBAI_WESTERN_SUBURBS_BBOX,
        LOCALITY_TO_ZONE_MAP,
        CANONICAL_EVENT_TYPE_MAP,
        CANONICAL_SOURCE_MAP,
        DEFAULT_SOURCE_RELIABILITY,
    )

# Common regex patterns
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_REGEX = re.compile(r"@\w+")
HASHTAG_REGEX = re.compile(r"#\w+")
WHITESPACE_REGEX = re.compile(r"\s+")
ASSET_ID_REGEX = re.compile(r"^AST-\d{3,6}$", re.IGNORECASE)

class TimestampNormalizer:
    """
    Normalizes timestamps into a consistent ISO 8601 timezone-aware format (IST / UTC+05:30).
    Preserves original timestamp strings and flags missing or unparseable values.
    """

    @staticmethod
    def normalize(raw_ts: Any) -> Tuple[Optional[str], Optional[int], str]:
        """
        Returns:
            (iso_string, epoch_ms, quality_status)
            quality_status: 'VALID', 'MISSING', 'INVALID_FORMAT'
        """
        if raw_ts is None or (isinstance(raw_ts, float) and (raw_ts != raw_ts or raw_ts == 0)):
            return None, None, "MISSING"

        ts_str = str(raw_ts).strip()
        if not ts_str or ts_str.lower() in ("nan", "none", "null", "nat", ""):
            return None, None, "MISSING"

        # Attempt parsing various common datetime formats
        dt: Optional[datetime.datetime] = None

        # 1. ISO format with/without Z or offset
        try:
            # Handle ISO string directly (Python 3.11+ fromisoformat handles most)
            cleaned_iso = ts_str.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(cleaned_iso)
        except Exception:
            pass

        # 2. Try common string patterns
        if dt is None:
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y/%m/%d %H:%M:%S",
                "%d-%m-%Y %H:%M:%S",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    dt = datetime.datetime.strptime(ts_str, fmt)
                    break
                except Exception:
                    continue

        if dt is None:
            # Could not parse format
            return None, None, "INVALID_FORMAT"

        # Ensure timezone awareness (Defaulting to IST / UTC+05:30 for Mumbai if naive)
        ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ist_tz)
        else:
            # Convert to IST for consistent corridor alignment
            dt = dt.astimezone(ist_tz)

        iso_str = dt.isoformat()
        epoch_ms = int(dt.timestamp() * 1000)
        return iso_str, epoch_ms, "VALID"


class SpatialNormalizer:
    """
    Validates geographic coordinates and verifies Mumbai Western Suburbs corridor boundaries.
    Preserves raw coordinate values and assigns spatial quality flags.
    """

    @staticmethod
    def validate_and_normalize(
        raw_lat: Any,
        raw_lon: Any,
        location_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates coordinates and assigns spatial quality flags.
        Returns dictionary with normalized fields and quality indicators.
        """
        lat_float: Optional[float] = None
        lon_float: Optional[float] = None
        spatial_quality = "MISSING_COORDINATES"

        # Parse raw latitude
        if raw_lat is not None and str(raw_lat).strip().lower() not in ("nan", "none", "null", ""):
            try:
                val = float(raw_lat)
                if -90.0 <= val <= 90.0:
                    lat_float = round(val, 6)
                else:
                    spatial_quality = "INVALID_COORDINATES"
            except (ValueError, TypeError):
                spatial_quality = "INVALID_COORDINATES"

        # Parse raw longitude
        if raw_lon is not None and str(raw_lon).strip().lower() not in ("nan", "none", "null", ""):
            try:
                val = float(raw_lon)
                if -180.0 <= val <= 180.0:
                    lon_float = round(val, 6)
                else:
                    spatial_quality = "INVALID_COORDINATES"
            except (ValueError, TypeError):
                spatial_quality = "INVALID_COORDINATES"

        # If both parsed successfully, test corridor bounds
        if lat_float is not None and lon_float is not None:
            bbox = MUMBAI_WESTERN_SUBURBS_BBOX
            in_corridor = (
                bbox["min_lat"] <= lat_float <= bbox["max_lat"] and
                bbox["min_lon"] <= lon_float <= bbox["max_lon"]
            )
            if in_corridor:
                spatial_quality = "VALID_IN_CORRIDOR"
            else:
                spatial_quality = "VALID_OUT_OF_BOUNDS"
        elif spatial_quality != "INVALID_COORDINATES":
            spatial_quality = "MISSING_COORDINATES"

        # Determine locality zone from location text
        locality_zone = "UNKNOWN"
        if location_text:
            cleaned_loc = str(location_text).strip().upper()
            locality_zone = LOCALITY_TO_ZONE_MAP.get(cleaned_loc, "UNKNOWN")

        return {
            "latitude": lat_float,
            "longitude": lon_float,
            "latitude_raw": raw_lat,
            "longitude_raw": raw_lon,
            "spatial_quality": spatial_quality,
            "is_geocoded": (lat_float is not None and lon_float is not None),
            "locality_zone": locality_zone,
        }


class TaxonomyNormalizer:
    """
    Normalizes incident categories, source names, asset IDs, and localities.
    """

    @staticmethod
    def normalize_event_type(raw_type: Any) -> str:
        if raw_type is None or str(raw_type).strip().lower() in ("nan", "none", "null", ""):
            return "GENERAL_INCIDENT"
        cleaned = str(raw_type).strip().upper().replace(" ", "_").replace("-", "_")
        return CANONICAL_EVENT_TYPE_MAP.get(cleaned, cleaned)

    @staticmethod
    def normalize_source_type(raw_src: Any) -> str:
        if raw_src is None or str(raw_src).strip().lower() in ("nan", "none", "null", ""):
            return "UNKNOWN"
        cleaned = str(raw_src).strip().upper()
        return CANONICAL_SOURCE_MAP.get(cleaned, cleaned)

    @staticmethod
    def normalize_asset_id(raw_asset: Any) -> Tuple[Optional[str], bool]:
        """
        Returns (normalized_asset_id, is_valid_format)
        """
        if raw_asset is None or str(raw_asset).strip().lower() in ("nan", "none", "null", ""):
            return None, False
        cleaned = str(raw_asset).strip().upper()
        is_valid = bool(ASSET_ID_REGEX.match(cleaned))
        return cleaned, is_valid

    @staticmethod
    def normalize_locality(raw_loc: Any) -> str:
        if raw_loc is None or str(raw_loc).strip().lower() in ("nan", "none", "null", ""):
            return "Unknown Locality"
        cleaned = str(raw_loc).strip()
        # Title case standard (e.g. Andheri West, Borivali East)
        return " ".join(word.capitalize() for word in cleaned.split())

    @staticmethod
    def normalize_reliability(raw_rel: Any, source_type: str) -> Tuple[float, bool]:
        """
        Returns (reliability_score, is_imputed)
        """
        if raw_rel is not None and str(raw_rel).strip().lower() not in ("nan", "none", "null", ""):
            try:
                val = float(raw_rel)
                if 0.0 <= val <= 1.0:
                    return round(val, 4), False
            except (ValueError, TypeError):
                pass
        
        # Fallback to source-type prior without silently faking precision
        fallback = DEFAULT_SOURCE_RELIABILITY.get(source_type, 0.50)
        return fallback, True


class TextNormalizer:
    """
    Cleans raw text descriptions and prepares normalized text representations for deduplication.
    """

    @staticmethod
    def clean_text(raw_text: Any) -> str:
        if raw_text is None or str(raw_text).strip().lower() in ("nan", "none", "null", ""):
            return ""
        text = str(raw_text)
        # Strip URLs
        text = URL_REGEX.sub("", text)
        # Strip mentions
        text = MENTION_REGEX.sub("", text)
        # Normalize whitespace
        text = WHITESPACE_REGEX.sub(" ", text)
        return text.strip()

    @staticmethod
    def normalize_for_matching(raw_text: Any) -> str:
        """
        Returns lowercased, stripped, alphanumeric-focused string for duplicate / repost detection.
        """
        cleaned = TextNormalizer.clean_text(raw_text)
        # Lowercase
        normalized = cleaned.lower()
        # Remove punctuation except spaces
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = WHITESPACE_REGEX.sub(" ", normalized).strip()
        return normalized
