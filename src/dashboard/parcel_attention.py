"""Generate a parcel-centred flood-model explanation from local artifacts."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError
from threading import Lock

import numpy as np
import rasterio
import torch
from pyproj import Geod
from rasterio.warp import transform
from rasterio.windows import Window

from .. import config
from ..data_pipeline import DataPipeline
from ..models.attention_unet_model import AttentionUNetModel
from ..models.dataset import DEFAULT_LAYOUT, EVENT_S1_LAYOUT


logger = logging.getLogger(__name__)
_INFERENCE = Lock()
_SETTLEMENTS = {"kibera", "mathare", "mukuru"}


@dataclass(frozen=True)
class ParcelAttention:
    rgb: np.ndarray
    heat: np.ndarray | None
    marker: tuple[int, int]
    image_width_m: float | None
    note: str


def load_parcel_attention(parcel: dict, root: Path | None = None) -> ParcelAttention | None:
    """Run tile-level Grad-CAM near the parcel using the trained flood head."""
    root = Path(root or os.environ.get("DASHBOARD_GRADCAM_DIR", "data/drive_dashboard/gradcam"))
    settlement = parcel.get("settlement")
    if settlement not in _SETTLEMENTS or not {"lon", "lat"} <= parcel.keys():
        return None
    path = root / f"{settlement}_raster_stack.tif"
    if not path.is_file():
        return None
    checkpoint = root / "attention_unet_best.pt"
    try:
        with rasterio.open(path) as src:
            layout = next((layout for layout in (DEFAULT_LAYOUT, EVENT_S1_LAYOUT)
                           if len(layout.band_names) == src.count), None)
            if src.crs is None or layout is None:
                return None
            xs, ys = transform("EPSG:4326", src.crs, [parcel["lon"]], [parcel["lat"]])
            row, col = src.index(xs[0], ys[0])
            if not (0 <= row < src.height and 0 <= col < src.width):
                return None
            tile_size = config.TILE_SIZE_PX
            half = tile_size // 2
            window = Window(col - half, row - half, tile_size, tile_size)
            tile = src.read(window=window, boundless=True, masked=True, out_dtype="float32").filled(np.nan)
            rgb_indices = [layout.band_names.index(name) for name in ("B4", "B3", "B2")]
            if not np.isfinite(tile[rgb_indices, half, half]).any():
                return None
            left = src.transform * (col - 31.5, row + .5)
            right = src.transform * (col + 32.5, row + .5)
            lon, lat = transform(src.crs, "EPSG:4326", [left[0], right[0]], [left[1], right[1]])
            width = Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])[2]
    except (rasterio.errors.RasterioError, ValueError, OverflowError):
        logger.warning("Could not read the parcel's satellite tile.", exc_info=True)
        return None

    sar = {i for i, name in enumerate(layout.band_names) if name.endswith("_vv")}
    scaled = np.nan_to_num(DataPipeline._normalise_bands(tile, sar_band_indices=sar), nan=0.0)
    # Use the training tile size for inference. Crop only the display to keep the parcel legible.
    crop = np.s_[half - 32:half + 32, half - 32:half + 32]
    rgb = np.moveaxis(scaled[rgb_indices], 0, -1)[crop]
    heat = None
    note = "The trained checkpoint is unavailable for this raster."
    if checkpoint.is_file():
        try:
            with _INFERENCE:
                weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model = AttentionUNetModel(layout=layout, device="cpu")
                model.network.load_state_dict(weights)
                overlay = model.explain(torch.from_numpy(scaled[layout.input_indices]))[crop]
            observed = np.isfinite(tile[rgb_indices]).all(axis=0)[crop]
            overlay = np.where(observed, overlay, 0)
            if np.isfinite(overlay).all() and np.any(overlay > 0):
                heat = overlay
                note = "Warmer areas contributed more to the flood model's output on this tile. The crosshair marks the selected parcel."
            else:
                note = "The model returned no positive attention in the area shown."
        except (OSError, RuntimeError, ValueError, UnpicklingError):
            logger.warning("Could not generate Grad-CAM for the parcel's raster layout.", exc_info=True)
            note = "A Grad-CAM overlay could not be generated for this model and raster."
    return ParcelAttention(rgb, heat, (32, 32), width, note)
