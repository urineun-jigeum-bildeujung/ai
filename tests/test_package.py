"""Package bootstrap tests that run without optional dependencies."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import gollajugae_ai  # noqa: E402


class PackageBootstrapTest(unittest.TestCase):
    def test_package_version_is_defined(self) -> None:
        self.assertEqual(gollajugae_ai.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
