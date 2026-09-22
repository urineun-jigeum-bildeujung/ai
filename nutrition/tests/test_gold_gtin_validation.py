from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "nutrition"))

from nutrition.build_nutrition_evidence_gold_cohort_p0 import gtin, gtin_check


class GoldGtinValidationTests(unittest.TestCase):
    def test_all_supported_ascii_gtin_lengths_and_check_digits(self):
        for value in ("73513537", "036000291452", "9501101530003", "00012345600012"):
            with self.subTest(value=value):
                self.assertTrue(gtin_check(value))
                self.assertEqual(gtin(value)[0], value)

    def test_formatted_non_ascii_and_invalid_check_digit_values_are_rejected(self):
        for value in ("03600029145", "036000291452 ", "036000-291452", "٠٣٦٠٠٠٢٩١٤٥٢"):
            with self.subTest(value=value):
                self.assertFalse(gtin_check(value))
                self.assertIsNone(gtin(value)[0])


if __name__ == "__main__":
    unittest.main()
