"""Validate required recording samples without waiting or decoding images."""

import math
from operator import index


def validate_recording_frames(frames, now_ns, max_age_s=2.0):
    """Reject missing or stalled local samples before saving a partial color row.

    receive_monotonic_ns is the host receive time, independent of camera clock
    calibration. Legacy samples without it remain usable but cannot be aged.
    """
    if not math.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError('recording frame timeout must be finite and positive')
    if not frames:
        raise RuntimeError('No enabled recording cameras; refusing an empty image row')
    for name, frame in frames.items():
        if frame is None:
            raise RuntimeError(f'{name}: required recording frame is missing')
        encoded = getattr(frame, 'jpg', None)
        if encoded is not None and len(encoded) == 0:
            raise RuntimeError(f'{name}: recording JPEG is empty')
        stamp = getattr(frame, 'receive_monotonic_ns', None)
        if stamp is None:
            continue
        try:
            stamp = index(stamp)
        except TypeError as error:
            raise RuntimeError(f'{name}: invalid recording frame receive timestamp') from error
        if stamp <= 0 or stamp > now_ns:
            raise RuntimeError(f'{name}: invalid recording frame receive timestamp')
        age_s = (now_ns - stamp) / 1e9
        if age_s > max_age_s:
            raise RuntimeError(f'{name}: recording stream stalled for {age_s:.2f}s '
                               f'(limit {max_age_s:g}s); refusing stale images')
