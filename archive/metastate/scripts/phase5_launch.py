"""Launch/resume the prepared phase-5 workflow in an independent process."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
run=Path(json.loads((ROOT/'result/phase5/latest.json').read_text())['directory'])
log=(run/'workflow.log').open('a')
worker=subprocess.Popen([str(ROOT/'.env/bin/python'),str(ROOT/'scripts/phase5.py'),'run','--directory',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
(run/'process.json').write_text(json.dumps({'pid':worker.pid,'directory':str(run)},indent=2)+'\n')
print(json.dumps({'launched':True,'pid':worker.pid,'directory':str(run)},indent=2))
