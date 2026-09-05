"""Run-local analysis configuration.

This module introduces :class:`AnalysisContext`, a snapshot of the configuration
state required by forecasts and likelihood-based data analysis.  During the
migration away from ``configs.config`` module globals, ``build_analysis_context``
uses the legacy initialiser and snapshots its resolved state.  New consumers can
therefore accept an ``AnalysisContext`` without depending on ``FisherMatrix``.

The legacy initialiser still mutates process-wide state in this milestone.  That
behaviour is intentionally preserved for compatibility and will be removed only
after probe implementations receive explicit contexts.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType, ModuleType
from typing import Any


def _freeze(value: Any) -> Any:
    """Recursively make resolved configuration values read-only.

    ``AnalysisContext`` is shared by data, theory, and sampler objects.  A
    mutable nested configuration value would reintroduce order-dependent run
    behaviour even though the dataclass itself is frozen.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    if (
        hasattr(value, "flags")
        and hasattr(value.flags, "writeable")
        and hasattr(value, "copy")
        and getattr(value, "ndim", 0) > 0
    ):
        frozen = value.copy()
        frozen.flags.writeable = False
        return frozen
    return value


@dataclass(frozen=True)
class AnalysisContext:
    """Resolved configuration and fiducial state for one analysis run.

    The field names intentionally mirror the existing configuration/FisherMatrix
    surface.  This keeps the object suitable as a compatibility bridge while
    separating run setup from Fisher-information computation.
    """

    settings: Mapping[str, Any]
    specs: Mapping[str, Any]
    observables: tuple[str, ...]
    freeparams: Mapping[str, Any]
    fiducialparams: Mapping[str, Any]
    fiducialcosmo: Any
    photoparams: Mapping[str, Any]
    photobiasparams: Mapping[str, Any]
    IAparams: Mapping[str, Any]
    Spectrobiasparams: Mapping[str, Any]
    Spectrononlinearparams: Mapping[str, Any]
    IMbiasparams: Mapping[str, Any]
    PShotparams: Mapping[str, Any]
    latex_names: Mapping[str, str]
    input_type: str
    backend_parameters: Mapping[str, Any]
    external: Mapping[str, Any] | None

    @classmethod
    def from_legacy_config(cls, legacy_config: ModuleType) -> AnalysisContext:
        """Snapshot resolved state published by :mod:`configs.config`.

        ``config.init`` must have completed successfully before this method is
        called.  Deep copies prevent a later legacy initialisation from changing
        the context's dictionaries.
        """

        input_type = legacy_config.input_type
        backend_attr = {
            "class": "boltzmann_classpars",
            "camb": "boltzmann_cambpars",
            "symbolic": "boltzmann_symbolicpars",
        }.get(input_type)
        backend_parameters = (
            deepcopy(getattr(legacy_config, backend_attr, {})) if backend_attr is not None else {}
        )
        external = getattr(legacy_config, "external", None)

        return cls(
            settings=_freeze(deepcopy(legacy_config.settings)),
            specs=_freeze(deepcopy(legacy_config.specs)),
            observables=tuple(deepcopy(legacy_config.obs)),
            freeparams=_freeze(deepcopy(legacy_config.freeparams)),
            fiducialparams=_freeze(deepcopy(legacy_config.fiducialparams)),
            # Cosmology instances are runtime services with backend-owned caches.
            # They are intentionally shared rather than deep-copied.
            fiducialcosmo=legacy_config.fiducialcosmo,
            photoparams=_freeze(deepcopy(legacy_config.photoparams)),
            photobiasparams=_freeze(deepcopy(legacy_config.Photobiasparams)),
            IAparams=_freeze(deepcopy(legacy_config.IAparams)),
            Spectrobiasparams=_freeze(deepcopy(legacy_config.Spectrobiasparams)),
            Spectrononlinearparams=_freeze(deepcopy(legacy_config.Spectrononlinearparams)),
            IMbiasparams=_freeze(deepcopy(legacy_config.IMbiasparams)),
            PShotparams=_freeze(deepcopy(legacy_config.PShotparams)),
            latex_names=_freeze(deepcopy(legacy_config.latex_names)),
            input_type=input_type,
            backend_parameters=_freeze(backend_parameters),
            external=_freeze(deepcopy(external)) if external is not None else None,
        )

    @property
    def obs(self) -> tuple[str, ...]:
        """Legacy spelling for ``observables``."""

        return self.observables

    @property
    def fiducialcosmopars(self) -> Mapping[str, Any]:
        """Legacy spelling for fiducial cosmological parameters."""

        return self.fiducialparams

    @property
    def photopars(self) -> Mapping[str, Any]:
        """Legacy spelling for photometric distribution parameters."""

        return self.photoparams

    @property
    def photobiaspars(self) -> Mapping[str, Any]:
        """Legacy spelling for photometric bias parameters."""

        return self.photobiasparams

    @property
    def IApars(self) -> Mapping[str, Any]:
        """Compatibility spelling for intrinsic-alignment parameters."""

        return self.IAparams

    @property
    def Spectrobiaspars(self) -> Mapping[str, Any]:
        """Compatibility spelling for spectroscopic bias parameters."""

        return self.Spectrobiasparams

    @property
    def PShotpars(self) -> Mapping[str, Any]:
        """Compatibility spelling for spectroscopic shot-noise parameters."""

        return self.PShotparams

    @property
    def Spectrononlinpars(self) -> Mapping[str, Any]:
        """Compatibility spelling used by the Fisher and likelihood modules."""

        return self.Spectrononlinearparams

    @property
    def allparams(self) -> dict[str, Any]:
        """Return the combined fiducial parameter mapping for this analysis."""

        return {
            **self.fiducialparams,
            **self.photoparams,
            **self.photobiasparams,
            **self.IAparams,
            **self.Spectrobiasparams,
            **self.Spectrononlinearparams,
            **self.IMbiasparams,
            **self.PShotparams,
        }

    @property
    def allparams_fidus(self) -> dict[str, Any]:
        """Compatibility alias for the Fisher configuration's fiducial mapping."""

        return self.allparams


def _load_legacy_config() -> ModuleType:
    """Load lazily so importing this module has no numerical-backend side effects."""

    return import_module("cosmicfishpie.configs.config")


def build_analysis_context(
    *,
    options: Mapping[str, Any] | None = None,
    specifications: Mapping[str, Any] | None = None,
    observables: list[str] | tuple[str, ...] | None = None,
    freepars: Mapping[str, Any] | None = None,
    extfiles: Mapping[str, Any] | None = None,
    fiducialpars: Mapping[str, Any] | None = None,
    photobiaspars: Mapping[str, Any] | str | None = None,
    photopars: Mapping[str, Any] | None = None,
    IApars: Mapping[str, Any] | None = None,
    PShotpars: Mapping[str, Any] | None = None,
    spectrobiaspars: Mapping[str, Any] | None = None,
    spectrononlinearpars: Mapping[str, Any] | None = None,
    IMbiaspars: Mapping[str, Any] | None = None,
    survey_name: str = "Euclid",
    cosmo_model: str = "w0waCDM",
    latexnames: Mapping[str, str] | None = None,
) -> AnalysisContext:
    """Build a run-local analysis context using the legacy resolver.

    Inputs are copied before calling ``config.init`` because its current
    implementation mutates ``options``.  This compatibility factory preserves
    legacy behaviour in milestone one; subsequent milestones will move the
    resolver itself out of the global configuration module.
    """

    legacy_config = _load_legacy_config()
    legacy_config.init(
        options=deepcopy(dict(options or {})),
        specifications=deepcopy(dict(specifications or {})),
        observables=deepcopy(observables),
        freepars=deepcopy(dict(freepars)) if freepars is not None else None,
        extfiles=deepcopy(dict(extfiles)) if extfiles is not None else None,
        fiducialpars=deepcopy(dict(fiducialpars)) if fiducialpars is not None else None,
        photobiaspars=deepcopy(photobiaspars),
        photopars=deepcopy(dict(photopars)) if photopars is not None else None,
        IApars=deepcopy(dict(IApars)) if IApars is not None else None,
        PShotpars=deepcopy(dict(PShotpars)) if PShotpars is not None else None,
        spectrobiaspars=deepcopy(dict(spectrobiaspars)) if spectrobiaspars is not None else None,
        spectrononlinearpars=(
            deepcopy(dict(spectrononlinearpars)) if spectrononlinearpars is not None else None
        ),
        IMbiaspars=deepcopy(dict(IMbiaspars)) if IMbiaspars is not None else None,
        surveyName=survey_name,
        cosmoModel=cosmo_model,
        latexnames=deepcopy(dict(latexnames)) if latexnames is not None else None,
    )
    return AnalysisContext.from_legacy_config(legacy_config)
