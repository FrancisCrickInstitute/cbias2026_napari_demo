"""
Segmentation shared by the workflow widget and a one-click menu command.

`segment_image` is a plain command, not a widget: napari runs it straight
from the Layers menu, injecting arguments based on their type annotations
and adding whatever it returns to the viewer. A menu command has no GUI for
parameters, so its parameters come from the plugin's settings, which the
manifest's `configurations` section adds to napari's Preferences dialog.
"""

from typing import TYPE_CHECKING

import numpy as np
from napari.settings import get_plugin_settings
from skimage import filters, measure, segmentation

if TYPE_CHECKING:
    import napari

PLUGIN_NAME = 'cbias2026-plugin'

# keys must match the `threshold_method` enum in napari.yaml
THRESHOLD_METHODS = {
    'otsu': filters.threshold_otsu,
    'li': filters.threshold_li,
    'triangle': filters.threshold_triangle,
}


def segment_channel(
    img: np.ndarray,
    sigma: float = 5,
    threshold_method: str = 'otsu',
    min_area: int = 0,
) -> np.ndarray:
    smoothed = filters.gaussian(img, sigma=sigma)
    mask = smoothed >= THRESHOLD_METHODS[threshold_method](smoothed)
    labels = measure.label(mask)
    if min_area > 0:
        areas = np.bincount(labels.ravel())
        too_small = areas < min_area
        too_small[0] = False  # background
        labels[too_small[labels]] = 0
        labels, _, _ = segmentation.relabel_sequential(labels)
    return labels


def segmentation_settings() -> dict:
    """Current values from Preferences > CBIAS2026 Demo Plugin.

    Returned as a plain dict so callers can take a snapshot on the main
    thread and hand it to a worker.
    """
    settings = get_plugin_settings(PLUGIN_NAME).segmentation
    return {
        'sigma': settings.sigma,
        'threshold_method': settings.threshold_method,
        'min_area': settings.min_area,
    }


def segment_image(
    image: 'napari.layers.Image',
) -> 'napari.types.LayerDataTuple':
    # napari injects the currently selected layer. Returning a
    # LayerDataTuple rather than LabelsData lets us name the new layer.
    labels = segment_channel(image.data, **segmentation_settings())
    return labels, {'name': f'{image.name}_segmented'}, 'labels'
