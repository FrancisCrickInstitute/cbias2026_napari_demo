"""
Segmentation shared by the workflow widget and a one-click menu command.

`segment_image` is a plain command, not a widget: napari runs it straight
from the Layers menu, injecting arguments based on their type annotations
and adding whatever it returns to the viewer.
"""

from typing import TYPE_CHECKING

import numpy as np
from skimage import filters, measure

if TYPE_CHECKING:
    import napari


def segment_channel(img: np.ndarray, sigma: float = 5) -> np.ndarray:
    smoothed = filters.gaussian(img, sigma=sigma)
    mask = smoothed >= filters.threshold_otsu(smoothed)
    return measure.label(mask)


def segment_image(
    image: 'napari.layers.Image',
) -> 'napari.types.LayerDataTuple':
    # napari injects the currently selected layer. Returning a
    # LayerDataTuple rather than LabelsData lets us name the new layer.
    labels = segment_channel(image.data)
    return labels, {'name': f'{image.name}_segmented'}, 'labels'
