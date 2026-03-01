#!/usr/bin/env python3
"""Run small test cases only.

Equivalent to: python3 tests.py small [options]

Usage:
  python3 tests_small.py [options] [filter...]

Options:
  --no-build         Skip build step
  --arch ARCH        Build architecture (rocketlake, icelake, native)
  -j N               Run N tests in parallel (default: 1)
  filter             Additional filters (ANDed with 'small')
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tests  # noqa: E402

# Inject 'small' as a default filter at position 1
sys.argv.insert(1, 'small')
tests.main()
