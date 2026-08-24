# Releasing to PyPI

Publishing is deliberately manual and tag-bound. The workflow rebuilds the notebooks,
runs the complete test suite and public calibration certificate, builds both artifacts,
and checks their metadata without any OIDC permission. It transfers only the verified
distributions to a separate, protected publish job. That minimal job does not check out
or execute project code; only it may request a short-lived PyPI token through Trusted
Publishing.

## One-time setup

1. On PyPI, configure a
   [pending or existing-project GitHub Trusted Publisher](https://docs.pypi.org/trusted-publishers/)
   for:
   repository `el-gabri/experimentation-framework`, workflow
   `publish-pypi.yml`, and environment `pypi`.
2. In GitHub, create the `pypi` environment and require reviewer approval. Restrict its
   deployment branch to protected `main`; the workflow independently checks that the
   exact release-tag input is an annotated tag with a GitHub-verified signature and
   points to a commit on `origin/main` before producing the upload artifact.
3. Protect `main` and require the CI checks in `ci.yml`.

No long-lived PyPI API token is required or expected in repository secrets.

## Release procedure

1. Update the version in `pyproject.toml` and
   `src/supply_experiments/_version.py`, and add the matching changelog entry. Search
   for the previous version and update the generated-notebook wheel paths and explicit
   package-metadata test expectation too. Then regenerate the notebooks. The metadata
   tests require all release-version references to agree.
2. Run locally:

   ```bash
   python -m pytest tests/ -q
   python calibration_certificate.py --output-dir calibration-artifacts
   python -m build
   python -m twine check dist/*
   ```

3. Commit and merge to `main`, then create and push an annotated, signed tag on that
   merged commit. Its name must be `v` plus the package version, for example
   `v2.0.0a1`. Confirm GitHub shows the tag signature as **Verified**.
4. In GitHub Actions, manually run **Publish to PyPI** from `main` and enter that exact
   tag. The unprivileged job checks out the tag, verifies its GitHub signature,
   `origin/main` ancestry, and package version, and builds with SHA-pinned actions.
   Only after that job succeeds does the publish-only job wait at the protected `pypi`
   environment and obtain OIDC permission for upload.
5. Verify the PyPI project page and install the exact version into a fresh environment.

PyPI files are immutable. If an artifact is wrong, increment the version; never reuse a
published version number.
