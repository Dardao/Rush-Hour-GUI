"""Allocate extra sampled exploration using only disagreement/observed archive size.

App minima are never loaded. Each selected board receives an independent seed.
Parallel simulator workers do not share learned weights or use solution oracles.
"""
import argparse,hashlib,json,subprocess,sys,shutil,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rushhour.research import boards,erase_loops,atomic_json,ROOT
from rushhour.core import display_moves

def main():
    a=argparse.ArgumentParser();a.add_argument('--runs',nargs='+',required=True);a.add_argument('--out',required=True);a.add_argument('--top-large',type=int,default=125);a.add_argument('--steps',type=int,default=10000000);a.add_argument('--seed',type=int,default=1042);a.add_argument('--workers',type=int,default=6);args=a.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);ps=boards();runs=[Path(x) for x in args.runs]
    records=[json.loads((r/'summary.json').read_text())['rows'] for r in runs];paths=[json.loads((r/'experience.json').read_text()) for r in runs]
    sizes={p.name:max(next(x['visited'] for x in rec if x['name']==p.name) for rec in records) for p in ps}
    differences={p.name for p in ps if len({display_moves(p,paths0[p.name],True) for paths0 in paths if p.name in paths0})>1}
    large=set(sorted(sizes,key=lambda name:(-sizes[name],name))[:args.top_large]);selected=[p for p in ps if p.name in differences|large]
    source=ROOT/'tools/explore.cpp';shutil.copy2(source,out/'explore-source.cpp');binary=(out/'explore').resolve();subprocess.run(['g++','-O3','-std=c++17',str(source),'-o',str(binary)],check=True)
    atomic_json(out/'selection.json',dict(config=vars(args),names=[p.name for p in selected],cost_disagreement=sorted(differences),largest_archives=sorted(large),app_reference_used=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    t0=time.monotonic()
    def work(item):
        i,p=item;inp=out/(p.name+'.txt');res=out/(p.name+'.tsv');seed=args.seed+i
        lines=['1',f'{p.name} {len(p.cars)}']+[f'{int(c.horizontal)} {c.length} {c.lane} {s}' for c,s in zip(p.cars,p.start)];inp.write_text('\n'.join(lines)+'\n')
        subprocess.run([str(binary),str(inp),str(res),'archive-refine',str(seed),str(args.steps+10000000),str(args.steps)],check=True,stdout=subprocess.DEVNULL)
        cols=res.read_text().strip('\n').split('\t');assert cols[0]==p.name and cols[1]=='1'
        path=erase_loops(p,[int(v) for v in cols[5].split(',') if v]);cost=display_moves(p,path,True)
        return dict(name=p.name,seed=seed,steps=int(cols[2]),resets=int(cols[3]),visited=int(cols[4]),moves=cost,path=path)
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work,enumerate(selected)):
            results.append(row)
            if len(results)%20==0:print('completed',len(results),'/',len(selected),flush=True)
    atomic_json(out/'experience.json',{r['name']:r['path'] for r in results})
    atomic_json(out/'summary.json',dict(config=vars(args),total=len(results),transitions=sum(r['steps'] for r in results),seconds=time.monotonic()-t0,rows=[{k:v for k,v in r.items() if k!='path'} for r in results]))
    print('refined',len(results),'boards in',round(time.monotonic()-t0,1),'seconds',flush=True)
if __name__=='__main__':main()
