"""Validate XR motion freshness without importing the robot or XR servers."""


def xr_input_is_fresh(*, motion_data_ready, motion_data_monotonic_ns, now_ns,
                      timeout_s=1.0):
    """Require a received motion packet and a recent host monotonic timestamp."""
    return bool(motion_data_ready and motion_data_monotonic_ns > 0
                and 0 <= now_ns - motion_data_monotonic_ns <= timeout_s * 1e9)
