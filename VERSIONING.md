# Versioning and releases

Piper Kernels uses Python's PEP 440 version format and `v<version>` Git tags. The version in
`pyproject.toml`, the Git tag, the PyPI release, and the GitHub release must agree.

## Compatibility policy

The package is pre-1.0 while its operator interfaces are being established:

- Patch releases (`0.1.1`) preserve the documented public API and contain fixes or compatible
  performance improvements.
- Minor releases (`0.2.0`) may add operators or deliberately change a public pre-1.0 contract.
- Release candidates (`0.2.0rc1`) are used when a change needs integration testing before a
  stable release.

After 1.0, patch releases contain compatible fixes, minor releases add backward-compatible
features, and major releases contain breaking changes.

Public compatibility includes exported names, tensor constructors and methods, documented
storage attributes, accepted shapes and dtypes, mutation and dispatch behavior, and serialized
packed-storage conventions. Modules and names beginning with an underscore are internal.

## Release process

1. Start from a green `main` branch and choose the next version from the policy above.
2. Run `uv version <version>` in a release branch and move the `Unreleased` changelog entries
   under a dated heading for that version.
3. Open and merge the release pull request after CPU and GPU validation passes.
4. Create and push an annotated `v<version>` tag on the merge commit.
5. Review and approve the protected `pypi` GitHub environment deployment.
6. Verify the PyPI installation and the automatically created GitHub release.

The release workflow builds each distribution once, validates the wheel in an isolated
environment, publishes those exact artifacts to PyPI through Trusted Publishing, and attaches
the same files to the GitHub release. Published versions and artifacts are immutable; fixes use
a new version.
