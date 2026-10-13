"""
UrbanPulse Multimodal Event Preprocessing Package
"""

from .config import (
    MUMBAI_WESTERN_SUBURBS_BBOX,
    CANONICAL_EVENT_TYPE_MAP,
    CANONICAL_SOURCE_MAP,
    LOCALITY_TO_ZONE_MAP,
)
from .normalizers import (
    TimestampNormalizer,
    SpatialNormalizer,
    TaxonomyNormalizer,
    TextNormalizer,
)
from .deduplicator import (
    ExactDuplicateDetector,
    SocialRepostDetector,
    IndependenceGuard,
)
from .event_schema import StandardizedEvent
from .pipeline import MultimodalPreprocessor

__all__ = [
    "MUMBAI_WESTERN_SUBURBS_BBOX",
    "CANONICAL_EVENT_TYPE_MAP",
    "CANONICAL_SOURCE_MAP",
    "LOCALITY_TO_ZONE_MAP",
    "TimestampNormalizer",
    "SpatialNormalizer",
    "TaxonomyNormalizer",
    "TextNormalizer",
    "ExactDuplicateDetector",
    "SocialRepostDetector",
    "IndependenceGuard",
    "StandardizedEvent",
    "MultimodalPreprocessor",
]
