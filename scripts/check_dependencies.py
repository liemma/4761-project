from __future__ import annotations

import importlib
from importlib import metadata


def _version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "<not installed>"


def _check_import(module: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, ""
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    # Core runtime deps (package + scripts)
    core = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
    ]

    # Notebook/data deps (optional, but needed for exploreAllen + MERFISH notebooks)
    notebooks = [
        ("anndata", "anndata"),
        ("jupyter", "jupyter"),
        ("ipykernel", "ipykernel"),
        ("abc-atlas-access", "abc_atlas_access"),
    ]

    print("== Core dependencies ==")
    for dist, mod in core:
        ok, err = _check_import(mod)
        status = "OK" if ok else "MISSING"
        print(f"{status:7s} {dist:16s} version={_version(dist):12s} import={mod}")
        if err:
            print(f"         import error: {err}")

    print("\n== Notebook/data dependencies (optional) ==")
    for dist, mod in notebooks:
        ok, err = _check_import(mod)
        status = "OK" if ok else "MISSING"
        print(f"{status:7s} {dist:16s} version={_version(dist):12s} import={mod}")
        if err:
            print(f"         import error: {err}")

    print("\n== Package import check ==")
    ok, err = _check_import("spatial_mrf")
    print(("OK" if ok else "MISSING"), "spatial_mrf import")
    if err:
        print("     import error:", err)


if __name__ == "__main__":
    main()

