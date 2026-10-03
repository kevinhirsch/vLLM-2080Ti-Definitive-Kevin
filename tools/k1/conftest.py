# Lane K1: tests must not touch the live engine/gateway (RL standing rule 2026-10-03).
import sys

sys.path.insert(0, "/home/kevin/projects/lanes/windows")
pytest_plugins = ["live_guard"]
