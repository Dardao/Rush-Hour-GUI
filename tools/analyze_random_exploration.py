"""One uniform-random pass: inspect self-discovered path quality, no training."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rushhour.boarddata import load_initial_boards
from rushhour.rl import episodes,replay_rewards
from rushhour.core import display_moves

class UniformScores:
    def forward(self,x):return np.zeros((len(x),141),np.float32)

def erase_observed_loops(p,path):
    states=[p.start];actions=[];indices={p.start:0}
    for action in path:
        nxt=p.move(states[-1],action)
        if nxt in indices:
            end=indices[nxt]
            for state in states[end+1:]:del indices[state]
            states=states[:end+1];actions=actions[:end]
        else:
            actions.append(action);states.append(nxt);indices[nxt]=len(states)-1
    assert p.is_solved(states[-1])
    return actions

def main():
    root=Path(__file__).resolve().parents[1];ps,_=load_initial_boards(root/'resources/initial_boards.json')
    ps=[replace(p,minimum_possible=None,minimum_source='',solution=()) for p in ps]
    rng=np.random.default_rng(42);ps=[ps[int(i)] for i in rng.permutation(len(ps))]
    solved=[];totals={k:0 for k in ['Easy','Medium','Hard','Expert']}
    for start in range(0,len(ps),100):
        group=ps[start:start+100];paths,status,_,_=episodes(UniformScores(),group,[False]*len(group),rng,epsilon=1)
        for p,path,s in zip(group,paths,status):
            if s!='solved':continue
            compact=erase_observed_loops(p,path)
            before=replay_rewards(p,path)[1];after=replay_rewards(p,compact)[1]
            assert after>=before-1e-9
            solved.append(dict(name=p.name,raw_actions=len(path),loop_free_actions=len(compact),raw_moves=display_moves(p,path,True),loop_free_moves=display_moves(p,compact,True),raw_return=before,loop_free_return=after))
            for mode in totals:
                if p.name.startswith(mode):totals[mode]+=1
    fields=['raw_actions','loop_free_actions','raw_moves','loop_free_moves','raw_return','loop_free_return']
    result=dict(method='one uniform random pass; no network learning; observed cycles erased only for analysis',total=2500,solved=len(solved),by_mode=totals,means={k:float(np.mean([r[k] for r in solved])) for k in fields},raw_positive=sum(r['raw_return']>0 for r in solved),loop_free_positive=sum(r['loop_free_return']>0 for r in solved),records=solved)
    (root/'verification/random-exploration.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='records'},flush=True)
if __name__=='__main__':main()
