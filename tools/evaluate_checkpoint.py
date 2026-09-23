"""Model-only subprocess first; app reference comparison afterwards."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,str(ROOT/'tools/evaluate_isolated.py'),'--model',str(Path(a.model).resolve()),'--out',str(out/'neural-evaluation.json')],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/compare_minima.py'),'--evaluation',str(out/'neural-evaluation.json'),'--out',str(out/'minimum-comparison.json')],check=True)
if __name__=='__main__':main()
