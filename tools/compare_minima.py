"""Post-hoc evaluation only. The trainer never imports or reads app minima."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rushhour.research import atomic_json, ROOT
from rushhour.discounting import successful_return

def compare(evaluation, reference):
    minimum={r['name']:r['minimum_possible'] for r in reference['values']}
    rows=evaluation['rows']; assert len(rows)==len(minimum)==2500
    assert set(minimum)=={r['name'] for r in rows}
    rr=[]
    for row in rows:
        r=dict(row);m=minimum[r['name']];solved=r['status']=='solved'
        r.update(minimum_possible=m,minimum_match=solved and r['moves']==m,
                 gap=r['moves']-m if solved else None,
                 efficiency=m/r['moves'] if solved else 0,
                 discounted_return=successful_return(r['moves']) if solved else None,
                 ideal_discounted_return=successful_return(m))
        rr.append(r)
    def summary(items):
        solved=[r for r in items if r['status']=='solved']
        return dict(total=len(items),solved=len(solved),minimum_match=sum(r['minimum_match'] for r in items),
                    below_reference=sum(r['moves']<r['minimum_possible'] for r in solved),
                    mean_solved_moves=sum(r['moves'] for r in solved)/len(solved) if solved else None,
                    mean_minimum=sum(r['minimum_possible'] for r in items)/len(items),
                    efficiency=sum(r['efficiency'] for r in items)/len(items))
    return dict(summary=summary(rr),by_mode={m:summary([r for r in rr if r['name'].startswith(m)]) for m in ['Easy','Medium','Hard','Expert']},rows=rr)

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=compare(json.loads(Path(a.evaluation).read_text()),json.loads((ROOT/'evaluation/minimum-possible.json').read_text()))
    atomic_json(a.out,result);print(result['summary']);print(result['by_mode'])
if __name__=='__main__':main()
