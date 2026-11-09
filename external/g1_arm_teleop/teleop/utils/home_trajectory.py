"""Fixed-start quintic arm home targets; no SDK, DDS, or wall-clock access."""

import math

import numpy as np


QUINTIC_MAX_SLOPE = 1.875
QUINTIC_MAX_CURVATURE = 10. * math.sqrt(3.) / 3.


def _vector(values, name):
    vector = np.array(values, dtype=float, copy=True)
    if vector.shape != (14,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must contain 14 finite joint values')
    return vector


def home_trajectory_duration(start_q, target_q, duration_s=3., max_velocity=.6, max_acceleration=1.):
    """Choose duration so every planned joint respects both derivative limits."""
    start, target = _vector(start_q, 'home start'), _vector(target_q, 'home target')
    if not all(math.isfinite(value) and value > 0
               for value in (duration_s, max_velocity, max_acceleration)):
        raise ValueError('home duration, velocity and acceleration must be finite and positive')
    distance = float(np.max(np.abs(target - start)))
    return max(float(duration_s), QUINTIC_MAX_SLOPE * distance / max_velocity,
               math.sqrt(QUINTIC_MAX_CURVATURE * distance / max_acceleration))


class HomeTrajectory:
    """Blend q and feedforward effort with zero endpoint velocity/acceleration."""

    def __init__(self, start_q, target_q, start_tauff, target_tauff, start_monotonic,
                 duration_s=3., max_velocity=.6, max_acceleration=1.):
        self.start_q, self.target_q = _vector(start_q, 'home start'), _vector(target_q, 'home target')
        self.start_tauff = _vector(start_tauff, 'home starting feedforward')
        self.target_tauff = _vector(target_tauff, 'home feedforward')
        if not math.isfinite(start_monotonic):
            raise ValueError('home start time must be finite')
        self.start_monotonic = float(start_monotonic)
        self.duration_s = home_trajectory_duration(self.start_q, self.target_q, duration_s,
                                                 max_velocity, max_acceleration)

    def progress(self, now_monotonic):
        if not math.isfinite(now_monotonic):
            raise ValueError('home sample time must be finite')
        elapsed = max(0., float(now_monotonic) - self.start_monotonic)
        return dict(duration_s=self.duration_s, elapsed_s=elapsed,
                    remaining_s=max(0., self.duration_s - elapsed),
                    fraction=min(1., elapsed / self.duration_s),
                    planned_complete=elapsed >= self.duration_s)

    def sample(self, now_monotonic):
        fraction = self.progress(now_monotonic)['fraction']
        if fraction == 0.:
            return self.start_q.copy(), self.start_tauff.copy()
        if fraction == 1.:
            return self.target_q.copy(), self.target_tauff.copy()
        blend = fraction ** 3 * (10. + fraction * (-15. + 6. * fraction))
        return (self.start_q + (self.target_q - self.start_q) * blend,
                self.start_tauff + (self.target_tauff - self.start_tauff) * blend)
