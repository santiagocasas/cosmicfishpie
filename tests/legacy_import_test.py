"""The deprecated ``cosmicfishpie`` import path must alias ``cosmicjellyfish``."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
import textwrap
import warnings

import pytest


def test_old_import_emits_a_deprecation_warning_and_returns_the_new_module():
    for name in list(sys.modules):
        if name == "cosmicfishpie" or name.startswith("cosmicfishpie."):
            del sys.modules[name]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = importlib.import_module("cosmicfishpie")

    new = importlib.import_module("cosmicjellyfish")
    assert old is new
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)


def test_old_and_new_submodules_are_the_same_object():
    old = importlib.import_module("cosmicfishpie.configs.config")
    new = importlib.import_module("cosmicjellyfish.configs.config")

    assert old is new
    assert old.__name__ == "cosmicjellyfish.configs.config"


def test_configuration_globals_are_shared():
    old = importlib.import_module("cosmicfishpie.configs.config")
    new = importlib.import_module("cosmicjellyfish.configs.config")

    old.LEGACY_ALIAS_PROBE = object()
    try:
        assert new.LEGACY_ALIAS_PROBE is old.LEGACY_ALIAS_PROBE
    finally:
        del old.LEGACY_ALIAS_PROBE


def test_from_import_resolves_through_the_alias():
    from cosmicfishpie.configs import config as old
    from cosmicjellyfish.configs import config as new

    assert old is new


@pytest.mark.parametrize("order", ["old-first", "new-first"])
def test_both_import_orders_work_in_a_fresh_interpreter(order):
    if order == "old-first":
        first, second = "cosmicfishpie", "cosmicjellyfish"
    else:
        first, second = "cosmicjellyfish", "cosmicfishpie"
    script = textwrap.dedent(f"""
        import importlib
        import warnings

        warnings.simplefilter("ignore", DeprecationWarning)

        a = importlib.import_module({first + ".cosmology.cosmology"!r})
        b = importlib.import_module({second + ".cosmology.cosmology"!r})
        assert a is b, "submodule identity"
        assert a.__name__ == "cosmicjellyfish.cosmology.cosmology", a.__name__
        print("ok")
        """)
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_package_data_is_reachable_through_either_name():
    from cosmicfishpie import configs as old_configs
    from cosmicjellyfish import configs as new_configs

    assert old_configs is new_configs
    assert (new_configs.__path__[0]).endswith("cosmicjellyfish/configs")


def test_legacy_pickle_using_the_old_module_path_still_loads():
    new_config = importlib.import_module("cosmicjellyfish.configs.config")
    payload = pickle.dumps(new_config.init, protocol=0)
    legacy_payload = payload.replace(
        b"cosmicjellyfish.configs.config", b"cosmicfishpie.configs.config"
    )

    assert legacy_payload != payload
    assert pickle.loads(legacy_payload) is new_config.init
