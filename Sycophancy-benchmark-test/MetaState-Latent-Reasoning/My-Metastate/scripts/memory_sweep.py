#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/MetaState'))
from candidate_trajectory.memory_sweep import main
if __name__=='__main__':main(ROOT)
