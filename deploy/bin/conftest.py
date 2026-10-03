# Every deploy/bin test runs under live_guard: no write or exec under the live engine/gateway dirs (RL 2026-10-03).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pytest_plugins = ["live_guard"]
