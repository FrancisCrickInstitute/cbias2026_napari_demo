import numpy as np
import pytest
import skimage.io
from napari.layers import Labels

from cbias2026_plugin._workflow import ExampleWorkflowWidget, segment_channel

SHAPE = (128, 128)
WAIT_MS = 10_000


def blobs(n: int) -> np.ndarray:
    """Dark image with `n` well-separated bright squares."""
    img = np.zeros(SHAPE, dtype=np.float32)
    for i in range(n):
        x = 10 + i * 40
        img[20:50, x : x + 25] = 1.0
    return img


def multichannel(blobs_per_channel: tuple[int, ...]) -> np.ndarray:
    """(C, Y, X) image, the layout napari layers hold after loading."""
    return np.stack([blobs(n) for n in blobs_per_channel])


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture
def widget(viewer):
    return ExampleWorkflowWidget(viewer)


def labels_layers(viewer):
    return [layer for layer in viewer.layers if isinstance(layer, Labels)]


def test_segment_channel_counts_objects():
    labels = segment_channel(blobs(3))
    assert labels.shape == SHAPE
    assert labels.max() == 3


def test_load_images_moves_channels_first(widget, viewer, qtbot, tmp_path):
    paths = []
    for name in ('a.tif', 'b.tif'):
        path = tmp_path / name
        # stored on disk as (Y, X, C)
        skimage.io.imsave(
            path,
            np.moveaxis(multichannel((1, 2, 3)), 0, -1),
            check_contrast=False,
        )
        paths.append(path)

    widget.load_images(paths)
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=WAIT_MS)

    assert [layer.name for layer in viewer.layers] == ['a.tif', 'b.tif']
    assert viewer.layers['a.tif'].data.shape == (3, *SHAPE)


def test_run_analysis_without_images_raises(widget):
    with pytest.raises(ValueError, match='No images'):
        widget.run_analysis()


def test_run_analysis_uses_selected_channel(widget, viewer, qtbot):
    viewer.add_image(multichannel((1, 3)), name='img1')
    viewer.add_image(multichannel((2, 2)), name='img2')
    widget.channel_idx.setValue(1)

    widget.run_analysis()
    qtbot.waitUntil(lambda: len(viewer.layers) == 4, timeout=WAIT_MS)

    assert viewer.layers['img1_labels_ch1'].data.max() == 3
    assert viewer.layers['img2_labels_ch1'].data.max() == 2


def test_run_analysis_table_has_every_object(widget, viewer, qtbot):
    viewer.add_image(multichannel((1,)), name='img1')
    viewer.add_image(multichannel((2,)), name='img2')

    widget.run_analysis()
    qtbot.waitUntil(lambda: len(viewer.layers) == 4, timeout=WAIT_MS)

    table = widget.output_box.toPlainText()
    assert 'eccentricity' in table
    # header + separator + one row per object
    assert len(table.splitlines()) == 2 + 3
    assert table.count('img1') == 1
    assert table.count('img2') == 2


def test_rerun_ignores_labels_layers(widget, viewer, qtbot):
    viewer.add_image(multichannel((1,)), name='img1')
    widget.run_analysis()
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=WAIT_MS)

    workers = widget.run_analysis()
    qtbot.waitUntil(lambda: len(viewer.layers) == 3, timeout=WAIT_MS)

    assert len(workers) == 1
    assert len(labels_layers(viewer)) == 2
