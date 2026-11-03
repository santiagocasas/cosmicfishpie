"""Deprecated import path for :mod:`cosmicjellyfish`.

``cosmicfishpie`` was renamed to ``cosmicjellyfish``. Importing this package is
kept working for one release cycle so that existing scripts, notebooks and
pickled objects keep loading, but it emits a :class:`DeprecationWarning`.

The alias does not copy anything: every ``cosmicfishpie.*`` module name resolves
to the *same* module object as the corresponding ``cosmicjellyfish.*`` module.
That keeps configuration globals, module-level locks, class identities and
pickling behaviour intact.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from types import ModuleType

_OLD = "cosmicfishpie"
_NEW = "cosmicjellyfish"
_PREFIX = _OLD + "."

_MESSAGE = (
    "The 'cosmicfishpie' package has been renamed to 'cosmicjellyfish'. "
    "This compatibility alias is deprecated and will be removed in a future "
    "release; update your imports from `import cosmicfishpie` to "
    "`import cosmicjellyfish`."
)


class _AliasLoader(importlib.abc.Loader):
    """Loader that hands back an already imported ``cosmicjellyfish`` module."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._saved: dict[str, object] = {}

    def create_module(self, spec):  # noqa: D102 - importlib protocol
        module = importlib.import_module(self._target)
        # importlib may stamp the legacy spec onto the shared module; remember
        # the real metadata so exec_module can restore it.
        for attribute in ("__name__", "__package__", "__loader__", "__spec__"):
            if attribute in module.__dict__:
                self._saved[attribute] = module.__dict__[attribute]
        return module

    def exec_module(self, module: ModuleType) -> None:
        for attribute, value in self._saved.items():
            module.__dict__[attribute] = value


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``cosmicfishpie.X`` to the existing ``cosmicjellyfish.X``."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102
        if not fullname.startswith(_PREFIX):
            return None
        target_name = _NEW + "." + fullname[len(_PREFIX) :]
        try:
            importlib.import_module(target_name)
        except ModuleNotFoundError as exc:
            if exc.name != target_name:
                raise
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(target_name))


def _install_finder() -> None:
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        # Insert before the path finders: the parent module's ``__path__`` points
        # at the real ``cosmicjellyfish`` directory, so otherwise ``PathFinder``
        # would load a second copy of every submodule under the legacy name.
        sys.meta_path.insert(0, _AliasFinder())


warnings.warn(_MESSAGE, DeprecationWarning, stacklevel=2)

_install_finder()

import cosmicjellyfish as _cosmicjellyfish  # noqa: E402

# Make ``import cosmicfishpie`` return the canonical module object.
sys.modules[__name__] = _cosmicjellyfish
