from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "query_d405_intrinsics.py"
SPEC = importlib.util.spec_from_file_location("query_d405_intrinsics", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_parse_intrinsics() -> None:
    output = '''Intrinsic of "Color" / 640x360 {
        Width: 640, Height: 360, PPX: 320.5, PPY: 179.5,
        Fx: 510.2, Fy: 510.3
    }'''
    assert module.parse_intrinsics(output) == [
        {
            "stream": "Color",
            "width": 640,
            "height": 360,
            "ppx": 320.5,
            "ppy": 179.5,
            "fx": 510.2,
            "fy": 510.3,
        }
    ]
