"""Independent, detailed context map for the PDF parcel locator."""

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import from_bounds
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.strtree import STRtree


logger = logging.getLogger(__name__)
_WEB_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_FRAME_RADIUS_M = 65


@dataclass(frozen=True)
class ParcelLocator:
    photo: Image.Image | None
    extent: tuple[float, float, float, float]
    point: tuple[float, float]
    outline: BaseGeometry | None
    footprint_source: str | None


@lru_cache(maxsize=6)
def _footprint_index(path: Path, modified_ns: int) -> STRtree:
    """Index the WGS84 building polygons once per saved dataset revision."""
    features = json.loads(path.read_text(encoding="utf-8-sig"))["features"]
    polygons = []
    for feature in features:
        geometry = feature.get("geometry")
        if geometry and geometry["type"] in {"Polygon", "MultiPolygon"}:
            polygon = shape(geometry)
            if polygon.is_valid and not polygon.is_empty:
                polygons.append(polygon)
    return STRtree(polygons)


def _find_footprint(root, settlement, lon, lat):
    for suffix, source in (("osm_buildings", "OpenStreetMap contributors (ODbL)"),
                           ("open_buildings", "Google Open Buildings")):
        path = root / f"{settlement}_{suffix}.geojson"
        if not path.is_file():
            continue
        try:
            index = _footprint_index(path, path.stat().st_mtime_ns)
            hits = index.query(Point(lon, lat), predicate="covered_by")
            if len(hits) == 1:
                return transform(_WEB_MERCATOR.transform, index.geometries[hits[0]]), source
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Could not read the parcel locator footprint layer.", exc_info=True)
    return None, None


def _read_locator_photo(path: Path, bbox: tuple[float, float, float, float]) -> Image.Image | None:
    if not path.is_file():
        return None
    with rasterio.open(path) as source:
        if source.crs is None or source.crs.to_epsg() != 3857 or source.count < 3:
            return None
        x0, y0, x1, y1 = bbox
        bounds = source.bounds
        if x0 < bounds.left or x1 > bounds.right or y0 < bounds.bottom or y1 > bounds.top:
            return None
        window = from_bounds(*bbox, transform=source.transform)
        pixels = source.read((1, 2, 3), window=window, out_dtype="uint8")
    return Image.fromarray(np.moveaxis(pixels, 0, -1))


def load_parcel_locator(parcel: dict) -> ParcelLocator | None:
    """Match a unique footprint and crop the local satellite mosaic."""
    root = Path(os.environ.get("DASHBOARD_LOCATOR_DIR", "data/drive_dashboard/footprints"))
    settlement = parcel.get("settlement")
    lon, lat = parcel.get("lon"), parcel.get("lat")
    if (not root.is_dir() or settlement not in {"kibera", "mathare", "mukuru"}
            or lon is None or lat is None or not (isfinite(lon) and isfinite(lat))
            or not (-180 <= lon <= 180 and -85 <= lat <= 85)):
        return None
    outline, source = _find_footprint(root, settlement, lon, lat)
    x, y = _WEB_MERCATOR.transform(lon, lat)
    d = _FRAME_RADIUS_M
    bbox = (x - d, y - d, x + d, y + d)
    try:
        photo = _read_locator_photo(root / f"{settlement}_locator_imagery.tif", bbox)
    except (OSError, ValueError, rasterio.errors.RasterioError):
        logger.info("Detailed satellite imagery is unavailable for the locator.")
        photo = None
    return ParcelLocator(photo, (bbox[0], bbox[2], bbox[1], bbox[3]), (x, y), outline, source)
