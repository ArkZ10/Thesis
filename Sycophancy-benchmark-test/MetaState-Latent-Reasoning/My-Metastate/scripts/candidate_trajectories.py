#!/usr/bin/env python3
"""Run, inspect, or replay the local candidate-trajectory feasibility study."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/MetaState'))
from candidate_trajectory.runner import main
if __name__=='__main__':main(ROOT)
