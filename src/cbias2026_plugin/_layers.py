"""
Helpers for putting results into the viewer from code that adds layers
itself, such as a widget's button callback or a thread worker's `returned`
callback.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Only for type hints, so the threshold widgets don't import pandas
    import napari
    import pandas as pd


def show_labels(
    viewer: 'napari.viewer.Viewer',
    data: np.ndarray,
    name: str,
    features: 'pd.DataFrame | None' = None,
) -> 'napari.layers.Labels':
    """Add a labels layer, or update the existing one with this name.

    `viewer.add_labels` always adds a new layer, and napari makes the name
    unique ("name [1]"), so without this every run adds another copy.
    """
    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = data
        if features is not None:
            layer.features = features
    else:
        layer = viewer.add_labels(data, name=name, features=features)
    return layer


def place_above(
    viewer: 'napari.viewer.Viewer',
    layer: 'napari.layers.Layer',
    below_name: str,
):
    """Move `layer` to sit directly above the layer called `below_name`.

    New layers go on top of the list, so with several images each result
    would end up over the wrong image. Keeping a result directly above its
    image means hiding the layers above both shows the matching pair.
    """
    layers = viewer.layers
    if below_name not in layers:
        # e.g. removed or renamed while a worker was running
        return
    layers.move(layers.index(layer), layers.index(below_name) + 1)
