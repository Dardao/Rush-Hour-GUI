"""Summarize completed experiments without using teacher data for RL evaluation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rushhour.boarddata import load_initial_boards
from rushhour.core import display_moves
from rushhour.expanded_rl import ExpandedRL
from rushhour.model import rollout

def main():
    root=Path(__file__).resolve().parents[1];ps,_=load_initial_boards(root/'resources/initial_boards.json');results=[]
    for name in ['rl2500-replay30','rl2500-replay300']:
        d=root/'experiments'/name;summary=json.loads((d/'summary.json').read_text())
        rows=[json.loads(s) for s in (d/'epochs.jsonl').read_text().splitlines()]
        assert len(rows)==20 and [r['epoch'] for r in rows]==list(range(1,21))
        manifest=summary['manifest'];assert len(manifest['training_names'])==2500 and len(set(manifest['training_names']))==2500
        assert not manifest['teacher_weights'] and not manifest['expert_actions'] and not manifest['minimum_used_for_training']
        net=ExpandedRL.load(d/'final.npz');paths,status=rollout(net,ps)
        moves=[display_moves(p,path,s=='solved') for p,path,s in zip(ps,paths,status)]
        assert status==rows[-1]['statuses'] and moves==rows[-1]['moves']
        fresh=ExpandedRL(42);initial=ExpandedRL.load(d/'initial.npz')
        assert all(np.array_equal(a,b) for a,b in zip(fresh.w+fresh.b,initial.w+initial.b))
        row=dict(name=name,epochs=20,training_puzzles=2500,episodes=50000,final_greedy=status.count('solved'),
                 best_greedy=max(r['solved'] for r in rows),final_explore=rows[-1]['explore'],
                 ever_successful_puzzles=rows[-1]['replay_size'],efficiency=rows[-1]['efficiency'],
                 greedy_status_counts={s:status.count(s) for s in sorted(set(status))},seconds=rows[-1]['seconds'],
                 replay_max=manifest['config']['replay_records'],fresh_initial_weights_verified=True,checkpoint_reload=True,
                 by_mode={mode:dict(solved=sum(s=='solved' for p,s in zip(ps,status) if p.name.startswith(mode)),
                                       count=sum(p.name.startswith(mode) for p in ps)) for mode in ['Easy','Medium','Hard','Expert']})
        results.append(row)
    (root/'verification/research2500.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
