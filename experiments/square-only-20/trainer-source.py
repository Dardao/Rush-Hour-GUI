"""Off-policy actor-critic on self-discovered experience, no expert data.

AWR-style objective: -exp((observed return - V)/temperature) * log pi(a|s).
The critic regresses discounted environmental returns. Neither policy nor critic
uses a shortest-path oracle, app minima, puzzle ID inputs, or teacher weights.
This IS self-imitation / weighted regression on the agent's own experience.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from rushhour.research import boards,atomic_json,erase_loops,ROOT
from rushhour.expanded_rl import ExpandedRL
from rushhour.model import rollout
from rushhour.rl import GAMMA,transition_reward,replay_rewards
from rushhour.core import display_moves
from rushhour.discounting import transition as square_transition
from rushhour.discounting import successful_return

def experience_rows(ps,paths,discount_unit="square",gamma=.99):
    # Deduplicate exact encoded states, keeping highest *observed* suffix return.
    # No successor enumeration, Bellman search or external targets are used.
    rows={}
    for puzzle in ps:
        if puzzle.name not in paths: continue
        state=puzzle.start;seen={state};episode=[]
        for action in paths[puzzle.name]:
            legal=puzzle.actions(state);assert action in legal
            mask=np.zeros(140,bool);mask[legal]=True
            nxt=puzzle.move(state,action)
            r,d = square_transition(puzzle,nxt,action,seen,gamma) if discount_unit=="square" else (transition_reward(puzzle,nxt,action,seen),gamma)
            episode.append((puzzle.encode(state),mask,int(action),r,d))
            state=nxt;seen.add(state)
        assert puzzle.is_solved(state)
        discounted=0.
        for x,mask,action,r,d in reversed(episode):
            discounted=r+d*discounted;key=x.tobytes()
            if key not in rows or discounted>rows[key][3]:rows[key]=(x,mask,action,discounted)
    values=list(rows.values())
    return np.stack([r[0] for r in values]),np.stack([r[1] for r in values]),np.array([r[2] for r in values]),np.array([r[3] for r in values],np.float32)

def evaluate(net,ps):
    paths,statuses=rollout(net,ps)
    result=dict(solved=statuses.count('solved'),total=len(ps),by_mode={m:sum(s=='solved' for p,s in zip(ps,statuses) if p.name.startswith(m)) for m in ['Easy','Medium','Hard','Expert']},loops=statuses.count('loop'),limits=statuses.count('limit'))
    costs=[display_moves(p,path,s=="solved") for p,path,s in zip(ps,paths,statuses)]
    result.update(mean_solved_moves=float(np.mean([m for m,s in zip(costs,statuses) if s=="solved"])) if "solved" in statuses else None, total_success_moves=sum(m for m,s in zip(costs,statuses) if s=="solved"))
    return result,paths,statuses

def failure_rows(ps,paths,status,x,actual_paths):
    wanted=set()
    for p,s,actual in zip(ps,status,actual_paths):
        if p.name not in paths:continue
        if s=="solved" and display_moves(p,actual,True)<=display_moves(p,paths[p.name],True):continue
        state=p.start
        for a in paths[p.name]:
            wanted.add(p.encode(state).tobytes());state=p.move(state,a)
    return np.array([i for i,row in enumerate(x) if row.tobytes() in wanted],dtype=np.int64)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experience',required=True);ap.add_argument('--out',required=True);ap.add_argument('--objective',choices=['awr','sil'],default='awr');ap.add_argument('--epochs',type=int,default=150);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--count',type=int,default=2500);ap.add_argument('--temperature',type=float,default=.25);ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--online-refresh',action='store_true');ap.add_argument('--resume')
    ap.add_argument('--focus-failures',action='store_true')
    ap.add_argument('--stop-after',type=int,default=0,help='Stop early without changing the planned LR schedule')
    ap.add_argument('--discount-unit',choices=['action','square'],default='square')
    ap.add_argument('--gamma',type=float,default=.99)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);ps=boards()[:a.count]
    paths=json.loads(Path(a.experience).read_text());paths={p.name:paths[p.name] for p in ps if p.name in paths}
    net=ExpandedRL.load(a.resume) if a.resume else ExpandedRL(a.seed);rng=np.random.default_rng(a.seed)
    atomic_json(out/'manifest.json',dict(config=vars(a),architecture=[224,512,512,141],initialization='resume own RL checkpoint' if a.resume else 'fresh random',external_answers=False,expert_initialization=False,minimum_metadata=False,experience_sha256=hashlib.sha256(Path(a.experience).read_bytes()).hexdigest(),board_sha256=hashlib.sha256((ROOT/'resources/boards.json').read_bytes()).hexdigest(),algorithm='archive exploration + '+a.objective+' self-imitation actor-critic',gamma=a.gamma,discount_unit=a.discount_unit,checkpoint_selection='most solved, then fewest actual squares; no app minima',reward='success +1, per-square -0.01 incl auto-exit, revisit -0.03',temperature=a.temperature))
    net.save(out/'initial.npz');atomic_json(out/'experience.json',paths)
    x,mask,actions,returns=experience_rows(ps,paths,a.discount_unit,a.gamma);print('rows',len(x),'return range',float(returns.min()),float(returns.max()),flush=True)
    history=[];best=(-1,float("-inf"));t0=time.monotonic();focus=np.array([],dtype=np.int64)
    for epoch in range(0,a.epochs+1):
        if epoch:
            lr=a.lr*(.05+.95*(1+np.cos(np.pi*(epoch-1)/max(1,a.epochs-1)))/2)
            order=rng.permutation(len(x));losses=[];weights=[]
            if a.focus_failures and len(focus):
                extra=rng.choice(focus,size=min(len(x)//2,len(focus)*32),replace=True)
                order=np.concatenate([order,extra]);rng.shuffle(order)
            for start in range(0,len(order),512):
                ix=order[start:start+512];value=net.forward(x[ix])[:,140]
                if a.objective=='awr':
                    advantage=np.exp(np.clip((returns[ix]-value)/a.temperature,-3,3)).astype(np.float32)
                    advantage/=max(float(advantage.mean()),1e-8)
                    target=returns[ix]
                else:
                    advantage=np.maximum(returns[ix]-value,0)
                    target=np.maximum(returns[ix],value)
                losses.append(net.actor_critic(x[ix],actions[ix],target,mask[ix],lr=lr,entropy_weight=0,policy_advantages=advantage))
                weights.append(float(advantage.mean()))
        if epoch%5==0 or epoch==1 or epoch==a.epochs or epoch==a.stop_after:
            result,new_paths,status=evaluate(net,ps)
            result.update(epoch=epoch,seconds=time.monotonic()-t0,replay_states=len(x),focus_states=len(focus),loss=float(np.mean(losses)) if epoch else None,mean_actor_weight=float(np.mean(weights)) if epoch else None)
            history.append(result);atomic_json(out/'history.json',history);net.save(out/'last.npz')
            atomic_json(out/'rng.json',rng.bit_generator.state)
            score=(result['solved'],-result['total_success_moves'])
            if score>best:
                best=score;net.save(out/'best.npz');atomic_json(out/'best-evaluation.json',dict(metrics=result,rows=[dict(name=p.name,status=s,path=list(map(int,path)),moves=display_moves(p,path,s=='solved')) for p,s,path in zip(ps,status,new_paths)]))
            print(out.name,result,flush=True)
            if a.online_refresh and epoch:
                updated=0
                for p,path,s in zip(ps,new_paths,status):
                    if s!='solved':continue
                    path=erase_loops(p,path)
                    if p.name not in paths or replay_rewards(p,path)[1]>replay_rewards(p,paths[p.name])[1]+1e-8:
                        paths[p.name]=path;updated+=1
                if updated:
                    atomic_json(out/'experience.json',paths);x,mask,actions,returns=experience_rows(ps,paths,a.discount_unit,a.gamma)
                    print('self-policy improved experiences',updated,'new rows',len(x),flush=True)
            if a.stop_after and epoch>=a.stop_after:break
            if a.focus_failures:focus=failure_rows(ps,paths,status,x,new_paths)
    net.save(out/'final.npz');atomic_json(out/'summary.json',dict(final=history[-1],best=best,epochs_completed=epoch))
if __name__=='__main__':main()
