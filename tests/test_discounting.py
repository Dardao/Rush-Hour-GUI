import unittest
from rushhour.discounting import square_transition, successful_return, transition
from rushhour.research import boards
from rushhour.core import display_moves
import json
from pathlib import Path
class DiscountTests(unittest.TestCase):
    def test_partition_invariance(self):
        for parts in [[13],[3,5,5],[1]*13,[2,4,7]]:
            g=0
            for i in range(len(parts)-1,-1,-1):
                r,d=square_transition(parts[i],i==len(parts)-1)
                g=r+d*g
            self.assertAlmostEqual(g,successful_return(13),12)
    def test_shorter_is_better(self):
        for n in range(1,500):self.assertGreater(successful_return(n),successful_return(n+1))
    def test_undiscounted(self):
        self.assertAlmostEqual(successful_return(13,1),.87)
    def test_exit_matches_display(self):
        paths=json.loads(Path('resources/baseline-self-experience.json').read_text())
        for p in boards()[:4]:
            state=p.start;seen={state};trans=[]
            for a in paths[p.name]:
                state=p.move(state,a);trans.append(transition(p,state,a,seen));seen.add(state)
            g=0
            for r,d in reversed(trans):g=r+d*g
            self.assertAlmostEqual(g,successful_return(display_moves(p,paths[p.name],True)),10)
if __name__=='__main__':unittest.main()
