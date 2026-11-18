import numpy as np
import pytest
from napari._app_model import get_app_model
from napari.layers import Labels

from cbias2026_plugin._segment import segment_channel, segment_image

SEGMENT_CMD = 'cbias2026-plugin.segment_image'
DEMO_MENU = 'cbias2026-plugin.demo_menu'


def two_blobs() -> np.ndarray:
    img = np.zeros((64, 64), dtype=np.float32)
    img[10:30, 10:30] = 1
    img[40:60, 40:60] = 1
    return img


def menu_command_ids(menu_id: str) -> list[str]:
    return [
        item.command.id
        for item in get_app_model().menus.get_menu(menu_id)
        if hasattr(item, 'command')
    ]


def test_segment_channel_labels_each_blob():
    labels = segment_channel(two_blobs())
    assert labels.shape == (64, 64)
    assert labels.max() == 2


def test_segment_image_names_output_after_input(make_napari_viewer):
    layer = make_napari_viewer().add_image(two_blobs(), name='blobs')

    data, meta, layer_type = segment_image(layer)

    assert data.max() == 2
    assert meta['name'] == 'blobs_segmented'
    assert layer_type == 'labels'


def test_menu_command_segments_selected_layer(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((64, 64)), name='other')
    viewer.add_image(two_blobs(), name='blobs')  # newest layer is selected

    get_app_model().commands.execute_command(SEGMENT_CMD).result()

    labels = viewer.layers['blobs_segmented']
    assert isinstance(labels, Labels)
    assert labels.data.max() == 2


def test_menu_command_only_enabled_for_image_layers(make_napari_viewer):
    from napari._app_model.context import get_context

    viewer = make_napari_viewer()
    [action] = [
        item.command
        for item in get_app_model().menus.get_menu(DEMO_MENU)
        if getattr(item, 'command', None) and item.command.id == SEGMENT_CMD
    ]

    def enabled() -> bool:
        return action.enablement.eval(get_context(viewer.layers))

    assert not enabled()
    viewer.add_image(two_blobs())
    assert enabled()
    viewer.add_labels(np.zeros((64, 64), dtype=int))
    assert not enabled()


def test_demo_submenu_is_under_layers_segment(make_napari_viewer):
    make_napari_viewer()
    submenus = [
        item.submenu
        for item in get_app_model().menus.get_menu('napari/layers/segment')
        if hasattr(item, 'submenu')
    ]
    assert DEMO_MENU in submenus
    assert SEGMENT_CMD in menu_command_ids(DEMO_MENU)


@pytest.mark.parametrize(
    ('menu_id', 'command_id'),
    [
        (DEMO_MENU, 'cbias2026-plugin.make_live_widget'),
        ('napari/layers/measure', 'cbias2026-plugin.make_workflow_widget'),
    ],
)
def test_widgets_are_in_menus(make_napari_viewer, menu_id, command_id):
    make_napari_viewer()
    assert command_id in menu_command_ids(menu_id)
