"""Import side effect: make the repo root (common, schema, register) importable from the tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
