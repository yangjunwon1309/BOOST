"""Apply validated RGB sensor controls without importing the optional SDK."""

from collections.abc import Mapping
import math


COLOR_OPTIONS = frozenset({
    "enable_auto_exposure", "enable_auto_white_balance",
    "auto_exposure_priority", "power_line_frequency",
    "exposure", "gain", "white_balance", "brightness", "contrast",
    "saturation", "gamma", "hue", "sharpness", "backlight_compensation",
})
AUTO_OPTIONS = ("enable_auto_exposure", "enable_auto_white_balance")


def apply_realsense_color_options(sensor, rs, options):
    """Validate every requested control, then apply auto flags before manual values.

    An absent mapping preserves the device's existing settings. Invalid names,
    unsupported controls and manual values with auto enabled fail explicitly.
    Returns the values read back from the device for startup logging.
    """
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise ValueError("color_options must be a mapping of RGB option names to numbers")

    validated = {}
    for name, value in options.items():
        if name not in COLOR_OPTIONS:
            raise ValueError(f"unknown RealSense RGB color option {name!r}")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"color_options.{name} must be a finite number or boolean")
        option = getattr(rs.option, name)
        if not sensor.supports(option) or sensor.is_option_read_only(option):
            raise ValueError(f"RealSense RGB option {name!r} is unsupported or read-only")
        limits = sensor.get_option_range(option)
        value = float(value)
        if not limits.min <= value <= limits.max:
            raise ValueError(
                f"color_options.{name}={value} is outside [{limits.min}, {limits.max}]"
            )
        if limits.step > 0:
            steps = (value - limits.min) / limits.step
            if not math.isclose(steps, round(steps), abs_tol=1e-5):
                raise ValueError(f"color_options.{name} must use steps of {limits.step}")
        validated[name] = (option, value)

    for manual, automatic in (
        ("exposure", "enable_auto_exposure"),
        ("gain", "enable_auto_exposure"),
        ("white_balance", "enable_auto_white_balance"),
    ):
        if manual in validated:
            enabled = (
                validated[automatic][1] if automatic in validated
                else sensor.get_option(getattr(rs.option, automatic))
            )
            if enabled:
                raise ValueError(f"color_options.{manual} requires {automatic}: false")

    ordered = [name for name in AUTO_OPTIONS if name in validated]
    ordered.extend(name for name in validated if name not in AUTO_OPTIONS)
    applied = {}
    for name in ordered:
        option, value = validated[name]
        sensor.set_option(option, value)
        applied[name] = sensor.get_option(option)
    return applied
