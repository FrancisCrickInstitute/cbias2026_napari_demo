import numpy as np
import pytest
from napari.settings import get_plugin_settings
from pydantic import ValidationError

from cbias2026_plugin._segment import (
    THRESHOLD_METHODS,
    segment_channel,
    segment_image,
)
from cbias2026_plugin._workflow import ExampleWorkflowWidget

PLUGIN = 'cbias2026-plugin'


@pytest.fixture
def settings():
    return get_plugin_settings(PLUGIN).segmentation


def close_blobs() -> np.ndarray:
    """Two squares 4 px apart: separate at low sigma, merged at high sigma."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[20:40, 10:28] = 1
    img[20:40, 32:50] = 1
    return img


def big_and_small_blob() -> np.ndarray:
    img = np.zeros((64, 64), dtype=np.float32)
    img[5:35, 5:35] = 1  # 900 px
    img[50:54, 50:54] = 1  # 16 px
    return img


def test_defaults_come_from_manifest(settings):
    assert settings.sigma == 5.0
    assert settings.threshold_method == 'otsu'
    assert settings.min_area == 0


def test_manifest_enum_matches_implemented_methods():
    from npe2 import PluginManager

    manifest = PluginManager.instance().get_manifest(PLUGIN)
    enum = (
        manifest.contributions.configurations['segmentation']
        .properties['threshold_method']
        .enum
    )
    assert set(enum) == set(THRESHOLD_METHODS)


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('sigma', 0.1),
        ('sigma', 100),
        ('min_area', -1),
        ('threshold_method', 'x'),
    ],
)
def test_invalid_values_rejected(settings, name, value):
    with pytest.raises(ValidationError):
        setattr(settings, name, value)


def test_min_area_removes_small_objects_and_relabels():
    labels = segment_channel(big_and_small_blob(), sigma=1, min_area=50)
    assert labels.max() == 1
    assert (labels == 1).sum() > 500


@pytest.mark.parametrize('method', sorted(THRESHOLD_METHODS))
def test_every_threshold_method_separates_distant_blobs(method):
    # distant blobs, so every method agrees; close ones depend on the method
    labels = segment_channel(
        big_and_small_blob(), sigma=1, threshold_method=method
    )
    assert labels.max() == 2


def test_menu_command_follows_settings(make_napari_viewer, settings):
    layer = make_napari_viewer().add_image(close_blobs(), name='blobs')

    settings.sigma = 1
    assert segment_image(layer)[0].max() == 2
    settings.sigma = 10
    assert segment_image(layer)[0].max() == 1


def test_workflow_widget_follows_settings(make_napari_viewer, settings, qtbot):
    viewer = make_napari_viewer()
    viewer.add_image(close_blobs()[np.newaxis], name='blobs')
    widget = ExampleWorkflowWidget(viewer)
    settings.sigma = 10

    widget.run_analysis()
    qtbot.waitUntil(lambda: len(viewer.layers) == 2, timeout=10_000)

    assert viewer.layers['blobs_labels_ch0'].data.max() == 1


def test_workflow_label_tracks_settings_changes(make_napari_viewer, settings):
    widget = ExampleWorkflowWidget(make_napari_viewer())
    assert widget.settings_label.text().startswith('sigma 5, otsu')

    settings.sigma = 2.5
    settings.threshold_method = 'li'

    assert widget.settings_label.text().startswith('sigma 2.5, li')
