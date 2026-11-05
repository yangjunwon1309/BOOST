import unittest
from types import SimpleNamespace

from teleimager.realsense_controls import COLOR_OPTIONS, apply_realsense_color_options


class RGBSensor:
    def __init__(self):
        self.values = {name: 0.0 for name in COLOR_OPTIONS}
        self.values.update(enable_auto_exposure=1.0, enable_auto_white_balance=1.0)
        self.writes = []
        self.unsupported = set()
        self.read_only = set()

    def supports(self, option):
        return option not in self.unsupported

    def is_option_read_only(self, option):
        return option in self.read_only

    def get_option_range(self, option):
        ranges = {
            "enable_auto_exposure": (0, 1, 1),
            "enable_auto_white_balance": (0, 1, 1),
            "brightness": (-64, 64, 1),
            "exposure": (1, 10000, 1),
            "gain": (0, 128, 1),
            "white_balance": (2800, 6500, 10),
        }
        low, high, step = ranges.get(option, (0, 100, 1))
        return SimpleNamespace(min=low, max=high, step=step)

    def get_option(self, option):
        return self.values[option]

    def set_option(self, option, value):
        self.writes.append((option, value))
        self.values[option] = value


class RealSenseControlsTest(unittest.TestCase):
    def setUp(self):
        self.sensor = RGBSensor()
        self.rs = SimpleNamespace(option=SimpleNamespace(**{name: name for name in COLOR_OPTIONS}))

    def test_absent_options_preserve_device_settings(self):
        self.assertEqual(apply_realsense_color_options(self.sensor, self.rs, None), {})
        self.assertEqual(self.sensor.writes, [])

    def test_manual_settings_disable_auto_first_even_if_yaml_lists_values_first(self):
        applied = apply_realsense_color_options(self.sensor, self.rs, {
            "exposure": 100,
            "white_balance": 4600,
            "enable_auto_white_balance": False,
            "enable_auto_exposure": False,
        })
        self.assertEqual(self.sensor.writes[:2], [
            ("enable_auto_exposure", 0.0), ("enable_auto_white_balance", 0.0),
        ])
        self.assertEqual(applied["exposure"], 100)
        self.assertEqual(applied["white_balance"], 4600)

    def test_manual_controls_are_rejected_when_auto_is_still_enabled(self):
        for options in ({"exposure": 100}, {"gain": 64}, {"white_balance": 4600}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "requires.*false"):
                    apply_realsense_color_options(self.sensor, self.rs, options)
        self.assertEqual(self.sensor.writes, [])

    def test_invalid_controls_fail_before_any_hardware_changes(self):
        for invalid in (
            {"brightnes": 0}, {"brightness": 65}, {"brightness": float("nan")},
            {"brightness": "20"}, {"white_balance": 4605}, False,
        ):
            with self.subTest(options=invalid):
                options = {"contrast": 50, **invalid} if isinstance(invalid, dict) else invalid
                with self.assertRaises(ValueError):
                    apply_realsense_color_options(self.sensor, self.rs, options)
        self.assertEqual(self.sensor.writes, [])

    def test_unsupported_and_read_only_controls_are_rejected(self):
        for attribute in ("unsupported", "read_only"):
            with self.subTest(attribute=attribute):
                getattr(self.sensor, attribute).add("brightness")
                with self.assertRaisesRegex(ValueError, "unsupported or read-only"):
                    apply_realsense_color_options(self.sensor, self.rs, {"brightness": 0})
                getattr(self.sensor, attribute).clear()
        self.assertEqual(self.sensor.writes, [])


if __name__ == "__main__":
    unittest.main()
