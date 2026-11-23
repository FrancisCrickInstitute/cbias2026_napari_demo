"""
Sample data shown under File > Open Sample > CBIAS2026 Demo Plugin

A sample data function takes no arguments and returns a list of layer data
tuples, `(data, add_kwargs[, layer_type])`, so one sample can add several
layers of different types.

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


def cells3d_slice():
    """The middle z-slice of `cells3d` as one (channel, Y, X) layer, the
    layout the workflow widget expects. Channel 0 is membrane, channel 1 is
    nuclei.

    napari's built-in `cells3d` sample splits the channels into separate 3D
    layers instead, which the workflow widget can't use.
    """
    volume = skimage.data.cells3d()  # (Z, C, Y, X)
    middle = volume[volume.shape[0] // 2]
    return [
        (
            middle,
            {
                'name': 'cells3d middle slice',
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
    image = filters.gaussian(mask.astype(np.float32), sigma=1.5)
    image = np.clip(image + rng.normal(0, 0.1, image.shape), 0, 1)
    return [
        (image.astype(np.float32), {'name': 'synthetic nuclei'}, 'image'),
        (truth, {'name': 'ground truth', 'opacity': 0.4}, 'labels'),
    ]
