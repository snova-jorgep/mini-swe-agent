"""Resolve the config_env label for a benchmark run.

CANONICAL COPY: <suite-root>/config_env.py

Submodules keep a verbatim copy at their own root so they run standalone. When
changing this file, propagate it to every copy (see the spec's "Files affected").
"""

import json
import os
import sys
from pathlib import Path

import yaml

REGISTRY_FILENAME = "config_envs.yaml"
REGISTRY_ENV_VAR = "FCSO_CONFIG_ENVS_FILE"
SIDECAR_FILENAME = "run_config_env.json"

EXTERNAL_PROVIDER_PROD = "external_provider_prod"
DEFAULT_CONFIG_ENV = "cloud-prod"
DEFAULT_INTERNAL_PROVIDERS = {"sambanova"}


def load_registry(start=None):
    """Find and parse config_envs.yaml, or return None if there isn't one.

    Absence is normal: a submodule cloned on its own has no suite root above it.
    """
    override = os.getenv(REGISTRY_ENV_VAR)
    if override:
        path = Path(override)
        if not path.is_file():
            raise ValueError(f"{REGISTRY_ENV_VAR} points at a missing file: {path}")
        return yaml.safe_load(path.read_text())

    start = Path(start) if start else Path(__file__).resolve().parent
    for directory in [start, *start.parents]:
        candidate = directory / REGISTRY_FILENAME
        if candidate.is_file():
            return yaml.safe_load(candidate.read_text())
    return None


def config_env_from_argv(argv=None):
    """Pop '--config-env <id>' out of argv and return the id, or None.

    For runners that parse sys.argv by hand instead of with argparse (e.g.
    '--dry-run' in sys.argv). The consumed pair is REMOVED, because such scripts
    also read argv positionally.
    """
    argv = sys.argv if argv is None else argv
    flag = "--config-env"

    # Only the space-separated form is supported. Silently ignoring '--flag=value'
    # would leave the token in argv, where positional logic misreads it.
    for arg in argv:
        if arg.startswith(f"{flag}="):
            raise SystemExit(f"Use '{flag} <value>', not '{arg}'")

    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise SystemExit(f"{flag} requires a value")
    value = argv[i + 1]
    del argv[i : i + 2]
    return value


def resolve_config_env(cli_value, config, registry=None):
    """Apply precedence: CLI > config field > registry default > cloud-prod.

    registry=None means "not supplied, load it" - NOT "there is no registry".
    Validation is skipped only when no config_envs.yaml exists on disk, which is
    the standalone-clone case. There is deliberately no way to disable it.
    """
    if registry is None:
        registry = load_registry()

    value = cli_value or (config or {}).get("config_env")
    if not value:
        value = (registry or {}).get("default_config_env") or DEFAULT_CONFIG_ENV

    if value == EXTERNAL_PROVIDER_PROD:
        raise ValueError(
            f"'{EXTERNAL_PROVIDER_PROD}' is reserved and assigned automatically to "
            "non-internal providers; it cannot be selected for a run."
        )

    known = (registry or {}).get("config_envs") or {}
    if known and value not in known:
        raise ValueError(
            f"Unknown config_env '{value}'. Valid ids: {', '.join(sorted(known))}. "
            f"Add it to {REGISTRY_FILENAME} if it is a new deployment."
        )
    return value


def internal_providers_from(config, registry=None):
    """Lowercase set of providers whose endpoints we control."""
    if registry is None:
        registry = load_registry()
    declared = (config or {}).get("internal_providers")
    if declared is None:
        declared = (registry or {}).get("internal_providers")
    if declared is None:
        return set(DEFAULT_INTERNAL_PROVIDERS)
    return {str(p).lower() for p in declared}


def row_config_env(config_env, provider, internal):
    """The config_env for one result row, decided by its provider."""
    return config_env if str(provider).lower() in internal else EXTERNAL_PROVIDER_PROD


def write_sidecar(run_dir, config_env, resolved_base_urls):
    """Persist config_env next to a run so later report scripts can recover it."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / SIDECAR_FILENAME
    path.write_text(
        json.dumps(
            {"config_env": config_env, "resolved_base_urls": resolved_base_urls},
            indent=2,
        )
    )
    return path


def read_sidecar(run_dir):
    """Read config_env written at generation time, or None."""
    path = Path(run_dir) / SIDECAR_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("config_env")
