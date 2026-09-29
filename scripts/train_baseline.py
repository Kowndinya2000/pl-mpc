"""Launch the separately vendored baseline without shadowing it from the working directory."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/tdmpc_square_original'))
runpy.run_module('tdmpc_square.train', run_name='__main__')
