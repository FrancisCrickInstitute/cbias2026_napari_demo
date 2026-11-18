import pytest
from npe2 import PluginManifest

PLUGIN = 'cbias2026-plugin'


@pytest.fixture(autouse=True)
def isolated_plugin(plugin_settings, npe2pm):
    """Register only this plugin, with settings stored under `tmp_path`.

    Without `plugin_settings`, anything that calls `get_plugin_settings`
    would read and write the real napari config directory, and the first
    test to do so would freeze the settings for the rest of the session.
    `plugin_settings` brings in npe2's `npe2pm`, which blocks plugin
    discovery, so this plugin has to be registered explicitly.
    """
    npe2pm.register(PluginManifest.from_distribution(PLUGIN))
    return npe2pm
