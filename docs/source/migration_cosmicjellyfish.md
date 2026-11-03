# Migrating from cosmicfishpie

The project was renamed from **`cosmicfishpie`** to **`cosmicjellyfish`**. The code is
unchanged apart from its import path, so a forecast should produce the same numbers after
the rename.

## What changes

| Before | After |
| --- | --- |
| `pip install cosmicfishpie` | `pip install cosmicjellyfish` |
| `import cosmicfishpie` | `import cosmicjellyfish` |
| `from cosmicfishpie.cosmology import cosmology` | `from cosmicjellyfish.cosmology import cosmology` |
| `cosmicfishpie/configs/...` in YAML paths | `cosmicjellyfish/configs/...` |

Update imports, `pyproject.toml`/`requirements.txt` pins, YAML files that point at
package paths, and any script paths in job submission templates.

## Backwards compatibility

For one release cycle the old import path keeps working:

```python
import cosmicfishpie  # DeprecationWarning
```

The alias is not a copy. `cosmicfishpie.X` and `cosmicjellyfish.X` are the **same module
object**, so global configuration state, module-level locks, class identities and pickled
objects are shared. Both import orders are supported, including loading an old pickle that
records a `cosmicfishpie.*` module path.

To silence the warning and be ready for the removal:

```bash
grep -rn "cosmicfishpie" .   # imports, paths, YAML
```

Two settings keep their old spelling on purpose:

- the sampler metadata key `cosmicfishpie_version` (written by new runs as well, so old
  analysis scripts keep reading it);
- the CLASS setting `neutrino_scheme: cosmicfishpie`, still accepted alongside
  `neutrino_scheme: cosmicjellyfish`.

Nothing else in the old name is preserved: `cosmicfishpie` will not be published to PyPI
under the old name, and the compatibility alias will be removed in a later major release.
