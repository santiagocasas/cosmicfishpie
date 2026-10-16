# GitHub Release Process

## Steps

1. Update the version in `cosmicfishpie/version.py`.

2. Run the release script:

    ```bash
    ./scripts/maintenance/release/release.sh
    ```

    This will commit the changes to the CHANGELOG and `version.py` files, create a new tag in git, and push
    both. The tag push triggers the `main.yml` release job (build, publish to PyPI, create the GitHub
    release) and the `docs.yml` workflow (rebuild and publish the docs site).

## Fixing a failed release

If for some reason the GitHub Actions release workflow failed with an error that needs to be fixed, you'll have to delete both the tag and corresponding release from GitHub. After you've pushed a fix, delete the tag from your local clone with

```bash
git tag -l | xargs git tag -d && git fetch -t
```

Then repeat the steps above.
