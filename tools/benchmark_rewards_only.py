"""Actual RL from random weights. No APK, teacher weights, or expert paths."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from rushhour.boarddata import load_initial_boards
from rushhour.core import display_moves
from rushhour.expanded_rl import ExpandedRL
from rushhour.model import rollout
from rushhour.rl import episodes,epsilon_at,replay_successes,replay_rewards
from rushhour.training import diagnostic_split,SuccessReplay

def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,choices=[4,20,88,2500],default=20);p.add_argument('--epochs',type=int,default=1000);p.add_argument('--seed',type=int,default=42);p.add_argument('--replay-records',type=int,default=30);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    boards=Path(__file__).resolve().parents[1]/'resources/initial_boards.json';allp,digest=load_initial_boards(boards)
    train=allp if a.count==2500 else diagnostic_split(allp,a.count)[0]
    minima={p.name:p.minimum_possible for p in train}
    # Remove even the app minimum from the objects reachable by the learner.
    for p in allp:p.minimum_possible=None;p.minimum_source='';assert p.solution==()
    net=ExpandedRL(a.seed);rng=np.random.default_rng(a.seed);replay=SuccessReplay(train,capacity=len(train));history=[];best=(-1,-1);t0=time.monotonic()
    manifest=dict(config=vars(a),architecture=[224,512,512,141],parameters=sum(w.size for w in net.w+net.b),board_source_sha256=hashlib.sha256(boards.read_bytes()).hexdigest(),apk_sha256=digest,
                  training_names=[p.name for p in train],initialization='fresh random',teacher_weights=False,expert_actions=False,minimum_used_for_training=False,augmentation=False,algorithm='epsilon-greedy clipped V-trace + self-generated positive-advantage replay')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));net.save(out/'initial.npz')
    with (out/'epochs.jsonl').open('w') as log:
        for epoch in range(1,a.epochs+1):
            chosen=[train[int(i)] for i in rng.permutation(len(train))];explored=0;rewards=[]
            for start in range(0,len(chosen),100):
                ps=chosen[start:start+100]
                paths,st,r,loss=episodes(net,ps,[True]*len(ps),rng,epsilon=epsilon_at(epoch,a.epochs))
                explored+=st.count('solved');rewards.extend(r)
                for puzzle,path,s in zip(ps,paths,st):
                    if s=='solved':replay.add(puzzle,path)
            replay_successes(net,replay.records(),rng,augment=False,max_records=a.replay_records)
            paths,st=rollout(net,train);moves=[display_moves(p,path,s=='solved') for p,path,s in zip(train,paths,st)]
            eff=sum(minima[p.name]/m if s=='solved' else 0 for p,m,s in zip(train,moves,st))/len(train)
            row=dict(epoch=epoch,epsilon=epsilon_at(epoch,a.epochs),explore=explored,solved=st.count('solved'),moves=moves,statuses=st,efficiency=eff,reward=float(np.mean(rewards)),replay_size=len(replay),seconds=time.monotonic()-t0)
            history.append(row);log.write(json.dumps(row)+'\n');log.flush()
            observed=float(np.mean([replay_rewards(p,path)[1] for p,path in zip(train,paths)]))
            row['checkpoint_score_reward']=observed
            row['by_mode']={mode:dict(solved=sum(s=='solved' for p,s in zip(train,st) if p.name.startswith(mode)),count=sum(p.name.startswith(mode) for p in train)) for mode in ['Easy','Medium','Hard','Expert']}
            if (row['solved'],observed)>best:best=(row['solved'],observed);net.save(out/'best.npz')
            if epoch%100==0 or a.count==2500 or epoch==a.epochs:print(a.count,epoch,'explore',explored,'greedy',row['solved'],'sec',round(row['seconds'],1),flush=True)
    temp=out/'epochs.complete.tmp';temp.write_text(''.join(json.dumps(r)+'\n' for r in history));temp.replace(out/'epochs.jsonl')
    net.save(out/'final.npz');(out/'summary.json').write_text(json.dumps(dict(manifest=manifest,last=history[-1],best=best,last100_mean=float(np.mean([r['solved'] for r in history[-100:]]))),indent=2))
    print('DONE',out,flush=True)

if __name__=='__main__':main()
