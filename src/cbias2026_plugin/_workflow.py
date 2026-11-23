"""
A small end-to-end analysis widget: load a folder of TIFFs, segment one
channel of each image, and tabulate per-object measurements.

Loading and analysis both run in `thread_worker`s so the viewer stays
responsive. Workers must not touch Qt widgets or napari layers, so
everything they need is read on the main thread and passed in, and results
come back to the main thread through the `yielded`/`returned` callbacks.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import skimage.io
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.settings import get_plugin_settings
from qtpy import QtGui
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from skimage import measure

from ._layers import place_above, show_labels
from ._segment import PLUGIN_NAME, segment_channel, segmentation_settings

if TYPE_CHECKING:
    import napari

# Define the properties from regionprops we want to extract
# NOTE: In practice we could expose this as a configuration or UI element
MEASURED_PROPERTIES = ('label', 'area', 'eccentricity')


def measure_objects(labels: np.ndarray, img: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        measure.regionprops_table(labels, img, properties=MEASURED_PROPERTIES)
    )


class ExampleWorkflowWidget(QWidget):
    def __init__(self, viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self.viewer = viewer
        self._results: list[pd.DataFrame] = []

        self.dir_btn = QPushButton('Select folder of images')
        self.dir_btn.clicked.connect(self.select_images)

        self.channel_idx = QSpinBox()
        self.channel_idx.setPrefix('Channel index: ')
        self.channel_idx.setValue(0)

        # Parameters live in Preferences > CBIAS2026 Demo Plugin, shared with
        # the Segment image menu command. Listening for changes keeps this
        # label current even when they're edited while the widget is open.
        self.settings_label = QLabel()
        get_plugin_settings(PLUGIN_NAME).events.changed.connect(
            self._show_settings
        )
        self._show_settings()

        self.run_btn = QPushButton('Run analysis')
        self.run_btn.clicked.connect(self.run_analysis)

        self.output_box = QTextBrowser()
        self.output_box.setWordWrapMode(QtGui.QTextOption.NoWrap)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.dir_btn)
        self.layout().addWidget(self.channel_idx)
        self.layout().addWidget(self.settings_label)
        self.layout().addWidget(self.run_btn)
        self.layout().addWidget(self.output_box)

    def _show_settings(self, event=None):
        params = segmentation_settings()
        self.settings_label.setText(
            f'sigma {params["sigma"]:g}, {params["threshold_method"]} '
            f'threshold, min area {params["min_area"]} px'
        )

    def select_images(self):
        result = QFileDialog.getExistingDirectory(
            self,
            caption='Select directory of images',
            directory=str(Path.home()),
        )
        if result != '':
            self.load_images(sorted(Path(result).glob('*.tif')))

    def load_images(self, paths: list[Path]):
        @thread_worker(connect={'yielded': self._add_image})
        def load_worker(paths):
            for path in paths:
                image = skimage.io.imread(path)
                # TIFFs here are (Y, X, C); napari expects channels first
                # NOTE: This is not universally true, just for demo data
                # In reality, we would use a more sophisticated data loader!
                image = np.transpose(image, (2, 0, 1))
                yield {'img': image, 'name': path.name}

        return load_worker(paths)

    def _add_image(self, result: dict):
        self.viewer.add_image(result['img'], name=result['name'])

    def run_analysis(self):
        # Only Image layers: earlier runs add Labels layers to the viewer
        image_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Image)
        ]
        if not image_layers:
            raise ValueError('No images loaded in viewer!')
        self.output_box.clear()
        self._results = []
        channel = self.channel_idx.value()
        params = segmentation_settings()

        @thread_worker(connect={'returned': self._add_results})
        def analysis_worker(img, name, channel, params):
            img_channel = img[channel, ...]
            labels = segment_channel(img_channel, **params)
            props = measure_objects(labels, img_channel)
            props['Image'] = name
            return {
                'layer_name': name,
                'labels': labels,
                'props': props,
                'channel': channel,
            }

        return [
            analysis_worker(layer.data, layer.name, channel, params)
            for layer in image_layers
        ]

    def _add_results(self, result: dict):
        props = result['props']
        self._results.append(props)
        labels = show_labels(
            self.viewer,
            result['labels'],
            f'{result["layer_name"]}_labels_ch{result["channel"]}',
            # napari shows a Labels layer's features for the object under the
            # cursor. Without an 'index' column it matches row i to label i,
            # but regionprops rows start at label 1.
            features=props.assign(index=props['label']),
        )
        # Ensure that the labels layers sit next to image layer for easy viewing
        place_above(self.viewer, labels, result['layer_name'])
        # Workers finish in any order, so redraw the table from everything
        # collected so far rather than appending fragments
        self.output_box.setPlainText(
            pd.concat(self._results, ignore_index=True).to_markdown(
                index=True, tablefmt='simple', floatfmt='.4f'
            )
        )
