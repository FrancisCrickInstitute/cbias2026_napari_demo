"""
Writers appear in File > Save Selected Layers. napari picks a writer by the
layer type and the file extension the user chooses, then calls it with the
layer's data and a dict of the layer's other attributes.

See: https://napari.org/stable/plugins/building_a_plugin/guides.html#writers
"""

from typing import Any

import numpy as np
import pandas as pd


def write_single_image(path: str, data: Any, meta: dict) -> list[str]:
    """Save an Image layer's array as .npy.

    This plugin's reader only opens integer .npy files, so only integer
    images round-trip.
    """
    np.save(path, np.asarray(data))
    return [path]


def write_labels_features(path: str, data: Any, meta: dict) -> list[str]:
    """Save a Labels layer's per-object measurements (its `features`) as CSV.

    Returns no paths when the layer has no features, which napari reports
    as a failed save rather than writing an empty file.
    """
    features = pd.DataFrame(meta.get('features'))
    if features.empty:
        return []
    # 'index' only exists to map rows to label values for napari
    features.drop(columns='index', errors='ignore').to_csv(path, index=False)
    return [path]
