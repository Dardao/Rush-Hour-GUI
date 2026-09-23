"""Run the GUI's supervised Worker without rendering, retaining actual metrics."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QCoreApplication
from rushhour.gui import Worker
from rushhour.expert_policy import PolicyNetwork
from rushhour.boarddata import load_initial_boards

def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=2500);p.add_argument('--epochs',type=int,default=70);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    puzzles,_=load_initial_boards(root/'resources/initial_boards.json');app=QCoreApplication([]);os.chdir(out)
    worker=Worker('supervised',PolicyNetwork((224,512,512,140),42),puzzles,epochs=a.epochs,diagnostic=True,diagnostic_count=a.count,augment=False)
    rows=[];errors=[];results=[];t0=time.monotonic()
    def progress(payload):
        if 'metrics' in payload:
            rows.append(payload['metrics']);r=rows[-1]
            print(r['epoch'],r['train_solved'],r['train_evaluated'],r['loss'],round(time.monotonic()-t0,1),flush=True)
    worker.progress.connect(progress);worker.error.connect(errors.append);worker.result.connect(results.append);worker.run()
    if errors:raise RuntimeError(errors)
    assert len(rows)==a.epochs and results
    summary=dict(count=a.count,epochs=a.epochs,seconds=time.monotonic()-t0,last=rows[-1],artifact_dir=str(worker.artifact_dir))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('DONE',out,flush=True)
if __name__=='__main__':main()
