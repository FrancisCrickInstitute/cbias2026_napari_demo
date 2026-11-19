import numpy as np
import pytest
from napari.layers import Image, Labels

from cbias2026_plugin._sample_data import synthetic_nuclei
from cbias2026_plugin._segment import segment_channel
from cbias2026_plugin._workflow import ExampleWorkflowWidget

PLUGIN = 'cbias2026-plugin'


@pytest.mark.parametrize(
    ('key', 'expected'),
    [
        ('mitotic_nuclei', {'mitotic nuclei': Image}),
        ('cells3d_projection', {'cells3d max projection': Image}),
        (
            'synthetic_nuclei',
            {'synthetic nuclei': Image, 'ground truth': Labels},
        ),
    ],
)
def test_samples_open_from_manifest(make_napari_viewer, key, expected):
    viewer = make_napari_viewer()

    viewer.open_sample(PLUGIN, key)

    assert {layer.name: type(layer) for layer in viewer.layers} == expected


def test_synthetic_nuclei_is_reproducible_per_seed():
    (img_a, _, _), (truth_a, _, _) = synthetic_nuclei(seed=0)
    (img_b, _, _), (truth_b, _, _) = synthetic_nuclei(seed=0)
    (_, _, _), (truth_c, _, _) = synthetic_nuclei(seed=1)

    np.testing.assert_array_equal(img_a, img_b)
    np.testing.assert_array_equal(truth_a, truth_b)
    assert not np.array_equal(truth_a, truth_c)


def test_synthetic_nuclei_ground_truth_matches_image():
    (img, _, _), (truth, _, _) = synthetic_nuclei()

    # light smoothing should recover nearly every true object; if the image
    # and labels drifted apart this would fail
    found = segment_channel(img, sigma=1).max()

    assert truth.max() > 50
    assert abs(found - truth.max()) <= 0.1 * truth.max()


def test_cells3d_projection_runs_through_workflow(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    viewer.open_sample(PLUGIN, 'cells3d_projection')
    widget = ExampleWorkflowWidget(viewer)
    widget.channel_idx.setValue(1)  # nuclei

    widget.run_analysis()
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=10_000)

    labels = viewer.layers['cells3d max projection_labels_ch1']
    assert labels.data.shape == (256, 256)
    assert labels.data.max() > 5
