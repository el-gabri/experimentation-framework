"""Single source of truth for the executable statistical implementation."""

IMPLEMENTATION_VERSION = "2.0.0a2"


def require_runtime_version(bound_version: str) -> None:
    """Fail closed when a stored design targets different estimator code."""
    if str(bound_version) != IMPLEMENTATION_VERSION:
        raise ValueError(
            "implementation_version diverge do runtime: "
            f"design={bound_version!r}, runtime={IMPLEMENTATION_VERSION!r}"
        )
