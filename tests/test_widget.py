import numpy as np
import pytest
from napari.layers import Labels

from cbias2026_plugin._widget import (
    ImageThreshold,
    ThresholdQWidget,
    threshold_autogenerate_widget,
    threshold_factory_widget,
    threshold_image,
)

PLUGIN = 'cbias2026-plugin'
# 0, 0.25, 0.5, 0.75, 1.0 across the columns of a 4x5 image
RAMP = np.tile(np.linspace(0, 1, 5), (4, 1))


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


def labels_layers(viewer):
    return [layer for layer in viewer.layers if isinstance(layer, Labels)]


def test_threshold_image_above_and_below():
    above = threshold_image(RAMP, 0.5)
    below = threshold_image(RAMP, 0.5, keep_below=True)
    # pixels equal to the threshold count as above, so the two modes split
    # the image between them with no pixel lost or counted twice
    assert above[0].tolist() == [False, False, True, True, True]
    assert below[0].tolist() == [True, True, False, False, False]
    np.testing.assert_array_equal(above, ~below)


def test_threshold_image_rescales_integer_images():
    img = np.array([[0, 100, 200, 255]], dtype=np.uint8)
    assert threshold_image(img, 0.5).tolist() == [[False, False, True, True]]


@pytest.mark.parametrize(
    'display_name',
    [
        'Threshold (autogenerate)',
        'Threshold (magic_factory)',
        'Threshold (magic_factory, live)',
        'Threshold (Container)',
        'Threshold (QWidget)',
        'Example Workflow Widget',
    ],
)
def test_manifest_widgets_open_in_napari(viewer, display_name):
    dock_widget, _ = viewer.window.add_plugin_dock_widget(PLUGIN, display_name)
    assert dock_widget is not None


@pytest.mark.parametrize(
    ('display_name', 'slider_attr'),
    [
        ('Threshold (magic_factory)', 'threshold'),
        ('Threshold (magic_factory, live)', 'threshold'),
        ('Threshold (Container)', '_threshold_slider'),
    ],
)
def test_slider_value_readout_is_visible(
    make_napari_viewer, qtbot, display_name, slider_attr
):
    from qtpy.QtWidgets import QAbstractSpinBox

    viewer = make_napari_viewer(show=True)
    _, widget = viewer.window.add_plugin_dock_widget(PLUGIN, display_name)
    readout = getattr(widget, slider_attr).native.findChild(QAbstractSpinBox)

    # Qt 6.12 collapses the readout's text box to 1 px (pinned below it in
    # pyproject.toml until magicgui#751 is released)
    qtbot.waitUntil(lambda: readout.lineEdit().width() > 20)
    assert readout.text() == '0.50'


def test_autogenerate_function_returns_mask():
    assert threshold_autogenerate_widget(RAMP, 0.5).shape == RAMP.shape


def test_factory_widget_called_directly(viewer):
    layer = viewer.add_image(RAMP)
    widget = threshold_factory_widget()

    mask = widget(layer, 0.5)

    assert mask.sum() == 3 * RAMP.shape[0]


def test_live_widget_reuses_output_layer(viewer):
    viewer.add_image(RAMP, name='ramp')
    _, widget = viewer.window.add_plugin_dock_widget(
        PLUGIN, 'Threshold (magic_factory, live)'
    )

    widget.threshold.value = 0.1
    widget.threshold.value = 0.6

    [labels] = labels_layers(viewer)
    assert labels.data.sum() == 2 * RAMP.shape[0]


def test_container_updates_layer_in_place(viewer):
    viewer.add_image(RAMP, name='ramp')
    widget = ImageThreshold(viewer)

    widget._threshold_slider.value = 0.1
    widget._invert_checkbox.value = True

    [labels] = labels_layers(viewer)
    assert labels.name == 'ramp_thresholded'
    # keep_below with threshold 0.1 leaves only the zero column
    assert labels.data.sum() == RAMP.shape[0]


def test_qwidget_dropdown_tracks_image_layers(viewer):
    viewer.add_image(RAMP, name='a')
    widget = ThresholdQWidget(viewer)
    viewer.add_image(RAMP, name='b')
    viewer.add_labels(np.zeros((4, 5), dtype=int), name='not an image')
    widget.layer_combo.setCurrentText('b')

    viewer.layers.remove('a')

    items = [
        widget.layer_combo.itemText(i)
        for i in range(widget.layer_combo.count())
    ]
    assert items == ['b']
    assert widget.layer_combo.currentText() == 'b'


def test_qwidget_button_thresholds_selected_layer(viewer):
    viewer.add_image(RAMP, name='ramp')
    widget = ThresholdQWidget(viewer)
    widget.threshold_box.setValue(0.6)

    widget.btn.click()
    widget.btn.click()

    [labels] = labels_layers(viewer)
    assert labels.name == 'ramp_thresholded'
    assert labels.data.sum() == 2 * RAMP.shape[0]


def test_qwidget_button_without_images_does_nothing(viewer):
    widget = ThresholdQWidget(viewer)

    widget.btn.click()

    assert len(viewer.layers) == 0


def test_threshold_widgets_open_as_tabs(viewer):
    window = viewer.window._qt_window
    first, _ = viewer.window.add_plugin_dock_widget(
        PLUGIN, 'Threshold (Container)'
    )
    second, _ = viewer.window.add_plugin_dock_widget(
        PLUGIN, 'Threshold (QWidget)'
    )

    assert second in window.tabifiedDockWidgets(first)


def test_workflow_widget_docks_left_without_tabbing(viewer):
    from qtpy.QtCore import Qt

    window = viewer.window._qt_window
    dock, _ = viewer.window.add_plugin_dock_widget(
        PLUGIN, 'Example Workflow Widget'
    )

    assert window.dockWidgetArea(dock) == Qt.DockWidgetArea.LeftDockWidgetArea
    # napari's layer list is already on the left; this stacks below it
    assert window.tabifiedDockWidgets(dock) == []
