#!/usr/bin/env python3
from pathlib import Path
import site

for root in map(Path, site.getsitepackages()):
    prefix = root / "cmeel.prefix"
    if (prefix / "lib/cmake/proxsuite/proxsuiteConfig.cmake").exists():
        print(prefix)
        raise SystemExit(0)
raise SystemExit("ProxSuite CMake prefix was not found")
