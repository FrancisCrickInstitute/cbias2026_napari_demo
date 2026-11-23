"""
The same thresholding operation/function exposed through each way napari
lets a plugin build a widget, from least to most code:

- a plain function flagged with `autogenerate: true` in the manifest.
    napari wraps it with magicgui, which builds the GUI from the type
    hints: an `ImageData` parameter becomes an image-layer dropdown and a
    `LabelsData` return value is added to the viewer as a Labels layer.
- a `magic_factory` decorated function, which lets us customise the
    generated widgets (here, a slider and the call button text).
- the same, with `auto_call=True`, so it re-runs as the slider moves.
- a `magicgui.widgets.Container` subclass, for composing magicgui widgets
    and wiring up our own callbacks.
- a `QWidget` subclass, for full control via Qt, at the cost of doing by
    hand what magicgui did for us, such as keeping the layer dropdown in
    sync with the viewer.

References:
- Widget specification: https://napari.org/stable/plugins/building_a_plugin/guides.html#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/
"""

from typing import TYPE_CHECKING

import numpy as np
from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from napari.layers import Image
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QWidget,
)
from skimage.util import img_as_float

if TYPE_CHECKING:
    import napari


def threshold_image(
    img: np.ndarray, threshold: float, keep_below: bool = False
) -> np.ndarray:
    """Shared func for every widget

    `img_as_float` rescales integer images to [0, 1], so one 0-1 threshold
    range works whatever the image's dtype.

    `keep_below` flag determines direction of threshold
    """
    img = img_as_float(img)
    return img < threshold if keep_below else img > threshold


def show_labels(viewer: 'napari.viewer.Viewer', data: np.ndarray, name: str):
    """Update the layer in place if it exists, so repeated runs don't pile up
    layers."""
    if name in viewer.layers:
        viewer.layers[name].data = data
    else:
        viewer.add_labels(data, name=name)


def threshold_autogenerate_widget(
    img: 'napari.types.ImageData', threshold: float = 0.5
) -> 'napari.types.LabelsData':
    return threshold_image(img, threshold)


@magic_factory(
    threshold={'widget_type': 'FloatSlider', 'max': 1},
    call_button='Threshold',
)
def threshold_factory_widget(
    img_layer: 'napari.layers.Image', threshold: float = 0.5
) -> 'napari.types.LabelsData':
    return threshold_image(img_layer.data, threshold)


# With auto_call=True there is no button: the function re-runs on every
# parameter change. napari reuses the output layer, so dragging the slider
# updates one Labels layer rather than adding one per value.
@magic_factory(
    threshold={'widget_type': 'FloatSlider', 'max': 1}, auto_call=True
)
def threshold_live_widget(
    img_layer: 'napari.layers.Image', threshold: float = 0.5
) -> 'napari.types.LabelsData':
    return threshold_image(img_layer.data, threshold)


class ImageThreshold(Container):
    def __init__(self, viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self._viewer = viewer
        # create_widget builds the same widgets magicgui would from a type
        # annotation, including a layer dropdown that tracks the viewer
        self._image_layer_combo = create_widget(
            label='Image', annotation='napari.layers.Image'
        )
        self._threshold_slider = create_widget(
            value=0.5,
            label='Threshold',
            annotation=float,
            widget_type='FloatSlider',
            options={'min': 0, 'max': 1},
        )
        self._invert_checkbox = CheckBox(text='Keep pixels below threshold')

        self._threshold_slider.changed.connect(self._threshold_im)
        self._invert_checkbox.changed.connect(self._threshold_im)

        self.extend(
            [
                self._image_layer_combo,
                self._threshold_slider,
                self._invert_checkbox,
            ]
        )

    def _threshold_im(self):
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return
        thresholded = threshold_image(
            image_layer.data,
            self._threshold_slider.value,
            keep_below=self._invert_checkbox.value,
        )
        show_labels(
            self._viewer, thresholded, f'{image_layer.name}_thresholded'
        )


class ThresholdQWidget(QWidget):
    # napari passes the viewer to any parameter annotated with
    # 'napari.viewer.Viewer'
    def __init__(self, viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self.viewer = viewer

        self.layer_combo = QComboBox()

        self.threshold_box = QDoubleSpinBox()
        self.threshold_box.setRange(0, 1)
        self.threshold_box.setSingleStep(0.05)
        self.threshold_box.setValue(0.5)

        self.invert_box = QCheckBox('Keep pixels below threshold')

        self.btn = QPushButton('Threshold')
        self.btn.clicked.connect(self._on_click)

        layout = QFormLayout()
        layout.addRow('Image', self.layer_combo)
        layout.addRow('Threshold', self.threshold_box)
        layout.addRow(self.invert_box)
        layout.addRow(self.btn)
        self.setLayout(layout)

        # magicgui's layer dropdowns auto-refresh dropdowns based on available layers
        # We need to do it ourselves here
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()

    def _refresh_layers(self, event=None):
        current = self.layer_combo.currentText()
        self.layer_combo.clear()
        self.layer_combo.addItems(
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
        )
        # keep the user's choice if that layer still exists
        if (idx := self.layer_combo.findText(current)) >= 0:
            self.layer_combo.setCurrentIndex(idx)

    def _on_click(self):
        name = self.layer_combo.currentText()
        if not name:
            return
        thresholded = threshold_image(
            self.viewer.layers[name].data,
            self.threshold_box.value(),
            keep_below=self.invert_box.isChecked(),
        )
        show_labels(self.viewer, thresholded, f'{name}_thresholded')
