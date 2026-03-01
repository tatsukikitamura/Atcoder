#!/usr/bin/env python3
"""Run all test cases (small + large).

Equivalent to: python3 tests.py [options]

Usage:
  python3 tests_all.py [options] [filter...]

Options:
  --no-build         Skip build step
  --arch ARCH        Build architecture (rocketlake, icelake, native)
  -j N               Run N tests in parallel (default: 1)
  filter             Only run test cases whose names contain this string
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tests  # noqa: E402

tests.main()
