"""
UrbanPulse Route Service (OSRM Backend Routing Foundation)
==========================================================
Dedicated backend routing service using Open Source Routing Machine (OSRM).
Features:
- Validates geographic coordinates (lat [-90, 90], lon [-180, 180]).
- Ensures OSRM longitude,latitude coordinate ordering.
- Retrieves full geometry as GeoJSON LineString (overview=full&geometries=geojson).
- Calculates distance in metres and kilometres.
- Calculates baseline duration in seconds and minutes (without claiming live traffic).
- Configurable via environment variables (OSRM_BASE_URL, OSRM_TIMEOUT_SECONDS, OSRM_PROFILE).
- Clear error handling for invalid input, unavailable OSRM, timeouts, and missing routes.
"""

import os
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import httpx
from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------------------------
# Domain Exceptions
# ------------------------------------------------------------------------------
class RoutingError(Exception):
    """Base exception for routing operations."""
    pass


class CoordinateValidationError(RoutingError):
    """Raised when coordinates are invalid or out of acceptable bounds."""
    pass


class RouteNotFoundError(RoutingError):
    """Raised when no route can be found between origin and destination."""
    pass


class OSRMTimeoutError(RoutingError):
    """Raised when communication with OSRM times out."""
    pass


class OSRMServiceError(RoutingError):
    """Raised when OSRM service is unavailable, unreachable, or returns a 5xx error."""
    pass


class OSRMInvalidResponseError(RoutingError):
    """Raised when OSRM returns an unparseable or malformed response."""
    pass


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
class OSRMConfig:
    """Configuration for OSRM Route API queries."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        profile: Optional[str] = None,
    ):
        raw_base = base_url or os.environ.get("OSRM_BASE_URL", "http://router.project-osrm.org")
        self.base_url = raw_base.rstrip("/")

        env_timeout = os.environ.get("OSRM_TIMEOUT_SECONDS")
        if timeout_seconds is not None:
            self.timeout_seconds = float(timeout_seconds)
        elif env_timeout:
            try:
                self.timeout_seconds = float(env_timeout)
            except ValueError:
                self.timeout_seconds = 10.0
        else:
            self.timeout_seconds = 10.0

        self.profile = (profile or os.environ.get("OSRM_PROFILE", "driving")).strip()


# ------------------------------------------------------------------------------
# Coordinate Validation Helper
# ------------------------------------------------------------------------------
def validate_coordinate_pair(lat: float, lon: float, label: str = "Coordinate") -> None:
    """
    Validates latitude and longitude values.
    Raises CoordinateValidationError if invalid or non-finite.
    """
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise CoordinateValidationError(f"{label} values must be numeric; got lat={lat!r}, lon={lon!r}")

    if not math.isfinite(lat) or not math.isfinite(lon):
        raise CoordinateValidationError(f"{label} values must be finite real numbers; got lat={lat}, lon={lon}")

    if not (-90.0 <= lat <= 90.0):
        raise CoordinateValidationError(
            f"{label} latitude must be between -90.0 and 90.0 degrees; got {lat}"
        )

    if not (-180.0 <= lon <= 180.0):
        raise CoordinateValidationError(
            f"{label} longitude must be between -180.0 and 180.0 degrees; got {lon}"
        )


# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------
class Coordinate(BaseModel):
    latitude: float = Field(..., description="Latitude in decimal degrees (-90.0 to 90.0)", json_schema_extra={"example": 19.2307})
    longitude: float = Field(..., description="Longitude in decimal degrees (-180.0 to 180.0)", json_schema_extra={"example": 72.8567})

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Latitude must be a finite real number")
        if not (-90.0 <= v <= 90.0):
            raise ValueError(f"Latitude must be between -90.0 and 90.0; got {v}")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Longitude must be a finite real number")
        if not (-180.0 <= v <= 180.0):
            raise ValueError(f"Longitude must be between -180.0 and 180.0; got {v}")
        return v


class GeoJSONLineString(BaseModel):
    type: str = Field(default="LineString", description="GeoJSON geometry type")
    coordinates: List[List[float]] = Field(
        ...,
        description="List of [longitude, latitude] coordinate pairs along the route"
    )


class DistanceBreakdown(BaseModel):
    meters: float = Field(..., description="Total route distance in metres", json_schema_extra={"example": 15420.2})
    kilometers: float = Field(..., description="Total route distance in kilometres", json_schema_extra={"example": 15.42})


class DurationBreakdown(BaseModel):
    seconds: float = Field(..., description="Baseline free-flow duration in seconds", json_schema_extra={"example": 1823.4})
    minutes: float = Field(..., description="Baseline free-flow duration in minutes", json_schema_extra={"example": 30.39})
    traffic_aware: bool = Field(
        default=False,
        description="Whether duration reflects live traffic conditions (OSRM does not provide live traffic ETAs)"
    )
    note: str = Field(
        default="Baseline duration from OSRM routing engine (free-flow road network speeds; live traffic-aware ETA not included).",
        description="Clarification on duration calculation methodology"
    )


class RouteSourceInfo(BaseModel):
    provider: str = Field(default="OSRM", description="Routing engine provider")
    profile: str = Field(default="driving", description="OSRM routing profile used (e.g., driving)")
    base_url: str = Field(..., description="OSRM service base URL queried")
    retrieval_timestamp: str = Field(..., description="ISO 8601 UTC timestamp of route retrieval")


class RouteLeg(BaseModel):
    distance_meters: float = Field(..., description="Leg distance in metres")
    duration_seconds: float = Field(..., description="Leg baseline duration in seconds")
    summary: Optional[str] = Field(default="", description="Summary description of the leg corridor")
    steps: Optional[List[Dict[str, Any]]] = Field(default=None, description="Turn-by-turn navigation steps if available")


class RoutePlanRequest(BaseModel):
    origin: Coordinate = Field(..., description="Starting location coordinate")
    destination: Coordinate = Field(..., description="Destination location coordinate")


class RoutePlanResponse(BaseModel):
    origin: Coordinate = Field(..., description="Origin coordinate")
    destination: Coordinate = Field(..., description="Destination coordinate")
    geometry: GeoJSONLineString = Field(..., description="Actual route geometry as GeoJSON LineString")
    distance: DistanceBreakdown = Field(..., description="Route distance in metres and kilometres")
    duration: DurationBreakdown = Field(..., description="Route baseline duration in seconds and minutes")
    source: RouteSourceInfo = Field(..., description="Route engine provider and retrieval timestamp metadata")
    legs: List[RouteLeg] = Field(default_factory=list, description="Route legs and steps if available")


# ------------------------------------------------------------------------------
# Route Service Implementation
# ------------------------------------------------------------------------------
class OSRMRouteService:
    """Dedicated service for planning routes via OSRM Route API."""

    def __init__(self, config: Optional[OSRMConfig] = None, client: Optional[httpx.Client] = None):
        self.config = config or OSRMConfig()
        self._client = client

    def build_osrm_url(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        alternatives: Union[bool, int] = False
    ) -> str:
        """
        Builds the OSRM route URL.
        NOTE: OSRM coordinates are strictly ordered {longitude},{latitude}!
        Format: {base_url}/route/v1/{profile}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson&steps=true
        """
        coords_str = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        url = f"{self.config.base_url}/route/v1/{self.config.profile}/{coords_str}?overview=full&geometries=geojson&steps=true"
        if alternatives:
            alt_val = "true" if alternatives is True else str(alternatives)
            url = f"{url}&alternatives={alt_val}"
        return url

    def plan_route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float
    ) -> RoutePlanResponse:
        """
        Validates input coordinates, queries OSRM, and returns primary route details.
        """
        return self.plan_routes(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            alternatives=False
        )[0]

    def plan_routes(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        alternatives: Union[bool, int] = 3
    ) -> List[RoutePlanResponse]:
        """
        Validates input coordinates, queries OSRM for candidate routes (including native alternatives),
        and returns list of parsed RoutePlanResponse objects.
        """
        # 1. Validate inputs
        validate_coordinate_pair(origin_lat, origin_lon, label="Origin")
        validate_coordinate_pair(dest_lat, dest_lon, label="Destination")

        # 2. Build URL with coordinates in longitude,latitude order
        url = self.build_osrm_url(origin_lat, origin_lon, dest_lat, dest_lon, alternatives=alternatives)

        # 3. Execute HTTP request
        retrieval_timestamp = datetime.now(timezone.utc).isoformat()
        response_data = self._fetch_osrm_data(url)

        # 4. Parse all returned candidate routes
        return self._parse_osrm_routes(
            response_data=response_data,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            retrieval_timestamp=retrieval_timestamp
        )

    def _fetch_osrm_data(self, url: str) -> Dict[str, Any]:
        """Performs HTTP GET to OSRM API with timeout and error handling."""
        timeout = httpx.Timeout(self.config.timeout_seconds)
        try:
            if self._client is not None:
                resp = self._client.get(url, timeout=timeout)
            else:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.get(url)
        except httpx.TimeoutException as e:
            raise OSRMTimeoutError(
                f"OSRM route request timed out after {self.config.timeout_seconds} seconds: {str(e)}"
            ) from e
        except (httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError) as e:
            raise OSRMServiceError(
                f"OSRM routing service is unavailable or unreachable: {str(e)}"
            ) from e
        except Exception as e:
            raise OSRMServiceError(
                f"Unexpected error communicating with OSRM service: {str(e)}"
            ) from e

        # Attempt JSON decoding
        try:
            data = resp.json()
        except Exception as e:
            if resp.status_code >= 500:
                raise OSRMServiceError(
                    f"OSRM service returned HTTP {resp.status_code} with non-JSON response."
                ) from e
            raise OSRMInvalidResponseError(
                f"Failed to parse OSRM response as JSON (HTTP {resp.status_code}): {str(e)}"
            ) from e

        # Check HTTP status >= 500
        if resp.status_code >= 500:
            msg = data.get("message", f"HTTP {resp.status_code} Server Error") if isinstance(data, dict) else f"HTTP {resp.status_code}"
            raise OSRMServiceError(f"OSRM service returned server error: {msg}")

        return data

    def _parse_osrm_response(
        self,
        response_data: Dict[str, Any],
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        retrieval_timestamp: str
    ) -> RoutePlanResponse:
        """Parses verified OSRM JSON response payload into a single RoutePlanResponse (primary route)."""
        routes = self._parse_osrm_routes(
            response_data=response_data,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            dest_lat=dest_lat,
            dest_lon=dest_lon,
            retrieval_timestamp=retrieval_timestamp
        )
        return routes[0]

    def _parse_osrm_routes(
        self,
        response_data: Dict[str, Any],
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        retrieval_timestamp: str
    ) -> List[RoutePlanResponse]:
        """Parses verified OSRM JSON response payload into a list of candidate RoutePlanResponses."""
        if not isinstance(response_data, dict):
            raise OSRMInvalidResponseError("OSRM response payload must be a JSON object.")

        code = response_data.get("code")
        if code != "Ok":
            message = response_data.get("message", "No message provided")
            if code in ("NoRoute", "NoSegment"):
                raise RouteNotFoundError(
                    f"No route found between ({origin_lat}, {origin_lon}) and ({dest_lat}, {dest_lon}): {message}"
                )
            if code in ("InvalidQuery", "InvalidValue"):
                raise CoordinateValidationError(f"OSRM rejected query as invalid: {message}")
            raise RoutingError(f"OSRM returned error code '{code}': {message}")

        routes = response_data.get("routes")
        if not routes or not isinstance(routes, list) or len(routes) == 0:
            raise RouteNotFoundError(
                f"No route returned in OSRM response for points ({origin_lat}, {origin_lon}) to ({dest_lat}, {dest_lon})."
            )

        parsed_routes: List[RoutePlanResponse] = []
        for r_idx, r_dict in enumerate(routes):
            if not isinstance(r_dict, dict):
                continue
            raw_geom = r_dict.get("geometry")
            if not raw_geom or not isinstance(raw_geom, dict):
                raise OSRMInvalidResponseError(f"OSRM route at index {r_idx} is missing geometry LineString.")

            coords = raw_geom.get("coordinates")
            if not isinstance(coords, list):
                raise OSRMInvalidResponseError(f"OSRM route at index {r_idx} geometry coordinates must be a list.")

            geom_type = raw_geom.get("type", "LineString")

            # Distance
            dist_m = float(r_dict.get("distance", 0.0))
            dist_km = round(dist_m / 1000.0, 3)

            # Baseline Duration
            dur_s = float(r_dict.get("duration", 0.0))
            dur_min = round(dur_s / 60.0, 2)

            # Legs & Steps
            raw_legs = r_dict.get("legs", [])
            parsed_legs: List[RouteLeg] = []
            if isinstance(raw_legs, list):
                for leg in raw_legs:
                    if isinstance(leg, dict):
                        parsed_legs.append(
                            RouteLeg(
                                distance_meters=float(leg.get("distance", 0.0)),
                                duration_seconds=float(leg.get("duration", 0.0)),
                                summary=str(leg.get("summary", "")),
                                steps=leg.get("steps") if isinstance(leg.get("steps"), list) else None
                            )
                        )

            parsed_routes.append(
                RoutePlanResponse(
                    origin=Coordinate(latitude=origin_lat, longitude=origin_lon),
                    destination=Coordinate(latitude=dest_lat, longitude=dest_lon),
                    geometry=GeoJSONLineString(type=geom_type, coordinates=coords),
                    distance=DistanceBreakdown(meters=round(dist_m, 2), kilometers=dist_km),
                    duration=DurationBreakdown(seconds=round(dur_s, 2), minutes=dur_min),
                    source=RouteSourceInfo(
                        provider="OSRM",
                        profile=self.config.profile,
                        base_url=self.config.base_url,
                        retrieval_timestamp=retrieval_timestamp
                    ),
                    legs=parsed_legs
                )
            )

        return parsed_routes


# ------------------------------------------------------------------------------
# Module-level instance helper
# ------------------------------------------------------------------------------
_default_route_service: Optional[OSRMRouteService] = None


def get_route_service() -> OSRMRouteService:
    """Returns the default OSRM route service instance."""
    global _default_route_service
    if _default_route_service is None:
        _default_route_service = OSRMRouteService()
    return _default_route_service


def set_route_service(service: Optional[OSRMRouteService]) -> None:
    """Overrides the default OSRM route service instance (useful in tests)."""
    global _default_route_service
    _default_route_service = service
