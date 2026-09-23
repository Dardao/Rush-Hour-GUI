"""Independent saved-weight evaluation in a directory without experience data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path

PROGRAM = r'''
import hashlib,json,sys
from pathlib import Path
import numpy as np
from rushhour import core
from rushhour.expanded_rl import ExpandedRL
from rushhour.model import rollout
from rushhour.research import boards
def forbidden(*args,**kwargs):raise AssertionError('No oracle or learning allowed')
core.shortest=forbidden;core.load_apk=forbidden;core.examples=forbidden
ExpandedRL.train_batch=forbidden;ExpandedRL.actor_critic=forbidden
def digest(net):return hashlib.sha256(b''.join(a.tobytes() for k in ['w','b','m','v'] for a in getattr(net,k))).hexdigest()
net=ExpandedRL.load('model.npz');before=digest(net);ps=boards()
paths,status=rollout(net,ps,limit=300)
after=digest(net);assert before==after
rows=[]
for p,path,s in zip(ps,paths,status):
 state=p.start
 for a in path:state=p.move(state,a)
 assert (s=='solved')==p.is_solved(state)
 rows.append(dict(name=p.name,status=s,actions=len(path),moves=core.display_moves(p,path,s=='solved'),path=list(map(int,path))))
out=dict(total=len(ps),solved=status.count('solved'),by_mode={m:sum(s=='solved' for p,s in zip(ps,status) if p.name.startswith(m)) for m in ['Easy','Medium','Hard','Expert']},max_actions=max(map(len,paths)),model_sha256=hashlib.sha256(Path('model.npz').read_bytes()).hexdigest(),weights_frozen=before==after,experience_files_available=False,oracle_available=False,external_answers=False,policy='masked neural argmax only; repeat-state stop; 300 action limit',rows=rows)
Path('result.json').write_text(json.dumps(out,separators=(',',':')))
print({k:v for k,v in out.items() if k!='rows'})
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='rushhour-neural-only-') as temp:
        temp=Path(temp);(temp/'rushhour').mkdir();(temp/'resources').mkdir()
        for name in ['__init__.py','core.py','model.py','expanded_rl.py','research.py']:
            shutil.copy2(root/'rushhour'/name,temp/'rushhour'/name)
        shutil.copy2(root/'resources/boards.json',temp/'resources/boards.json');shutil.copy2(a.model,temp/'model.npz')
        (temp/'evaluate.py').write_text(PROGRAM)
        subprocess.run([sys.executable,'evaluate.py'],cwd=temp,check=True)
        shutil.copy2(temp/'result.json',a.out)
if __name__=='__main__':main()
