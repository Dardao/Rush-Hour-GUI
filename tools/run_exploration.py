"""Compile and audit rules-only exploration; persist all discovered success paths."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, argparse, subprocess, time, hashlib, json, shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rushhour.research import boards, ROOT, erase_loops, atomic_json
from rushhour.rl import replay_rewards
from rushhour.core import display_moves

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['archive','archive-score','archive-refine','random'],default='archive');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--budget',type=int,default=200000);ap.add_argument('--out',required=True);ap.add_argument('--refinement-steps',type=int,default=100000)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);ps=boards()
    source=ROOT/'tools/explore.cpp';binary=out/'explore'
    shutil.copy2(source,out/'explore-source.cpp')
    subprocess.run(['g++','-O3','-std=c++17',str(source),'-o',str(binary)],check=True)
    lines=[str(len(ps))]
    for p in ps:
        lines.append(f'{p.name} {len(p.cars)}')
        lines.extend(f'{int(c.horizontal)} {c.length} {c.lane} {s}' for c,s in zip(p.cars,p.start))
    (out/'boards.txt').write_text('\n'.join(lines)+'\n');t0=time.monotonic()
    subprocess.run([str(binary),str(out/'boards.txt'),str(out/'observed.tsv'),a.mode,str(a.seed),str(a.budget),str(a.refinement_steps)],check=True)
    rows=[];paths={}
    for p,line in zip(ps,(out/'observed.tsv').read_text().splitlines()):
        name,ok,steps,resets,visited,raw=line.split('\t');assert name==p.name
        path=[int(x) for x in raw.split(',') if x]
        row=dict(name=name,solved=bool(int(ok)),steps=int(steps),resets=int(resets),visited=int(visited))
        if int(ok):
            path=erase_loops(p,path);state=p.start
            for action in path:
                assert not p.is_solved(state)
                state=p.move(state,action)
            assert p.is_solved(state)
            paths[name]=path;row.update(actions=len(path),moves=display_moves(p,path,True),reward=replay_rewards(p,path)[1])
        rows.append(row)
    assert len(rows)==2500
    summary=dict(config=vars(a),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),boards_sha256=hashlib.sha256((ROOT/'resources/boards.json').read_bytes()).hexdigest(),solved=len(paths),seconds=time.monotonic()-t0,transitions=sum(r['steps'] for r in rows),by_mode={m:sum(r['solved'] for r in rows if r['name'].startswith(m)) for m in ['Easy','Medium','Hard','Expert']},rows=rows)
    atomic_json(out/'experience.json',paths);atomic_json(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='rows'},flush=True)
if __name__=='__main__':main()
