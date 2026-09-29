"""Evaluate a checkpoint using the separately vendored TD-M(PC)² baseline."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/tdmpc_square_original'))
runpy.run_module('tdmpc_square.evaluate', run_name='__main__')
