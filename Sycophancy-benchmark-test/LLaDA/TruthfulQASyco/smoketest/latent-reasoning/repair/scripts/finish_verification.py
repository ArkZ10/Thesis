"""Run verification, grading, and offline report creation without live monitoring."""
import argparse,subprocess,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
code=1
try:
    code=subprocess.run([sys.executable,'-u',str(root/'scripts/verify_development.py'),'all','--directory',str(a.directory)],cwd=root).returncode
finally:
    report=subprocess.run([sys.executable,str(root/'scripts/report.py')],cwd=root).returncode
    (a.directory/'completion.json').write_text(json.dumps({'verification_exit_code':code,'report_exit_code':report},indent=2)+'\n')
sys.exit(code or report)
