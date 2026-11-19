"""
Sample data shown under File > Open Sample > CBIAS2026 Demo Plugin, each
chosen to suit one of this plugin's demos.

A sample data function takes no arguments and returns a list of layer data
tuples, `(data, add_kwargs[, layer_type])`, so one sample can add several
layers of different types.

`human_mitosis` and `cells3d` are downloaded by scikit-image (via pooch) the
first time they're used and cached after that, so open them once before a
session without internet.

See: https://napari.org/stable/plugins/building_a_plugin/guides.html#sample-data
Based on napari's own skimage samples: https://github.com/napari/napari/blob/main/src/napari_builtins/_skimage_data.py
"""

import numpy as np
import skimage.data
from skimage import filters, measure


def mitotic_nuclei():
    """2D nuclei, small and densely packed: a target for the threshold
    widgets and the Segment image menu command."""
    return [(skimage.data.human_mitosis(), {'name': 'mitotic nuclei'})]


def cells3d_projection():
    """`cells3d` as one (channel, Y, X) layer, the layout the workflow widget
    expects. Channel 0 is membrane, channel 1 is nuclei.

    napari's built-in `cells3d` sample splits the channels into separate 3D
    layers instead, which the workflow widget can't use.
    """
    # (Z, C, Y, X) -> (C, Y, X)
    projection = skimage.data.cells3d().max(axis=0)
    return [
        (
            projection,
            {
                'name': 'cells3d max projection',
                'metadata': {'channels': ['membrane', 'nuclei']},
            },
        )
    ]


def synthetic_nuclei(seed: int = 0):
    """A noisy image of blob-shaped 'nuclei' plus their exact labels, so
    segmentation results can be compared against a known answer."""
    mask = skimage.data.binary_blobs(
        length=256, blob_size_fraction=0.06, volume_fraction=0.2, rng=seed
    )
    truth = measure.label(mask)
    rng = np.random.default_rng(seed)
    # Blur the edges and add noise so thresholding isn't trivially perfect
    image = filters.gaussian(mask.astype(np.float32), sigma=2)
    image = np.clip(image + rng.normal(0, 0.15, image.shape), 0, 1)
    return [
        (image.astype(np.float32), {'name': 'synthetic nuclei'}, 'image'),
        (truth, {'name': 'ground truth', 'opacity': 0.4}, 'labels'),
    ]
