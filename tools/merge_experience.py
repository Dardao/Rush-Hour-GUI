"""Keep shorter actually observed success paths. No app references are read."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rushhour.research import boards,erase_loops,atomic_json
from rushhour.core import display_moves

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',nargs='+',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    ps=boards();merged={};costs={};sources=[]
    for f in a.inputs:
        pth=Path(f);paths=json.loads(pth.read_text());changed=0
        for p in ps:
            if p.name not in paths:continue
            path=erase_loops(p,paths[p.name]);cost=display_moves(p,path,True)
            if p.name not in merged or cost<costs[p.name]:
                merged[p.name]=path;costs[p.name]=cost;changed+=1
        sources.append(dict(path=f,sha256=hashlib.sha256(pth.read_bytes()).hexdigest(),improved_or_added=changed))
    atomic_json(a.out,merged)
    atomic_json(Path(a.out).with_suffix('.provenance.json'),dict(sources=sources,total=len(merged),mean_moves=sum(costs.values())/len(costs),minimum_reference_used=False))
    print(len(merged),'observed success paths; mean moves',sum(costs.values())/len(costs))
if __name__=='__main__':main()
