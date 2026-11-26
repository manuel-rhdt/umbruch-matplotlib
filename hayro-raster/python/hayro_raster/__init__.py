"""Locates the ``hayro-raster`` executable installed with this package."""
import os
import shutil
import sysconfig
from pathlib import Path

__all__ = ["find_hayro_raster"]

_EXE = "hayro-raster.exe" if os.name == "nt" else "hayro-raster"


def find_hayro_raster():
    """Path of the hayro-raster executable installed with this package, or None.

    Looks in the scripts directory of the running environment (also the user scheme and,
    for editable setups, ``PATH``) rather than assuming that the environment is activated.
    """
    candidates = [sysconfig.get_path("scripts")]
    for scheme in (f"{os.name}_user", "posix_user", "nt_user"):
        try:
            candidates.append(sysconfig.get_path("scripts", scheme))
        except KeyError:
            pass
    for d in candidates:
        if d and (Path(d) / _EXE).is_file():
            return str(Path(d) / _EXE)
    return shutil.which("hayro-raster")
