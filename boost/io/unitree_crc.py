"""Compatibility wrapper for Unitree SDK2's optional CRC native library."""

from __future__ import annotations

from unitree_sdk2py.utils.crc import CRC


def make_unitree_crc() -> CRC:
    """Use the SDK's C CRC when present, with its built-in Python fallback otherwise.

    The pinned SDK source contains ``utils/lib/crc_*.so``, but its editable/wheel
    installation may omit that directory. The SDK's CRC class already includes
    an identical pure Python implementation; its Linux constructor just does not
    select it when the shared object is missing.
    """
    try:
        return CRC()
    except OSError as exc:
        message = str(exc)
        if (
            "/utils/lib/crc_" not in message
            or "No such file or directory" not in message
        ):
            raise
        crc = CRC.__new__(CRC)
        crc.platform = "Portable"
        return crc
