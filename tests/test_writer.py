import numpy as np
import pandas as pd
import pytest
from napari import save_layers
from napari.layers import Image, Labels

from cbias2026_plugin._reader import napari_get_reader
from cbias2026_plugin._writer import write_labels_features

PLUGIN = 'cbias2026-plugin'


@pytest.fixture
def measured_labels() -> Labels:
    data = np.zeros((10, 10), dtype=int)
    data[1:4, 1:4] = 1
    data[6:9, 6:9] = 2
    features = pd.DataFrame(
        {'index': [1, 2], 'label': [1, 2], 'area': [9.0, 9.0]}
    )
    return Labels(data, features=features)


def test_features_writer_saves_measurements(measured_labels, tmp_path):
    path = str(tmp_path / 'measurements.csv')
    data, meta, _ = measured_labels.as_layer_data_tuple()

    assert write_labels_features(path, data, meta) == [path]

    saved = pd.read_csv(path)
    # napari's lookup column is not part of the measurements
    assert list(saved.columns) == ['label', 'area']
    assert saved['label'].tolist() == [1, 2]


def test_features_writer_skips_layers_without_features(tmp_path):
    path = tmp_path / 'measurements.csv'
    data, meta, _ = Labels(np.zeros((5, 5), dtype=int)).as_layer_data_tuple()

    assert write_labels_features(str(path), data, meta) == []
    assert not path.exists()


def test_napari_picks_features_writer_for_csv(
    make_napari_viewer, measured_labels, tmp_path
):
    make_napari_viewer()
    path = str(tmp_path / 'measurements.csv')

    written = save_layers(path, [measured_labels], plugin=PLUGIN)

    assert written == [path]
    assert pd.read_csv(path)['area'].tolist() == [9.0, 9.0]


def test_integer_image_round_trips_through_reader(
    make_napari_viewer, tmp_path
):
    make_napari_viewer()
    original = np.random.default_rng(0).integers(0, 255, (8, 8))
    path = str(tmp_path / 'image.npy')

    save_layers(path, [Image(original)], plugin=PLUGIN)
    [(read_back, _, _)] = napari_get_reader(path)(path)

    np.testing.assert_array_equal(read_back, original)
