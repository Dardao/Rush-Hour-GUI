import tempfile,unittest
from pathlib import Path
import numpy as np
from rushhour.boarddata import load_initial_boards
from rushhour.training import diagnostic_split
from rushhour.supervised import build_examples
from rushhour.expert_policy import PolicyNetwork
from rushhour.expanded_rl import ExpandedRL
from rushhour.rl import episodes

class SupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.puzzles,_=load_initial_boards(Path(__file__).resolve().parents[1]/'resources/initial_boards.json')

    def test_examples_exclude_holdout(self):
        train,valid,test=diagnostic_split(self.puzzles,20)
        x,y,mask,rows=build_examples(train)
        allowed={p.name for p in train};heldout={p.name for p in valid+test}
        self.assertEqual({r[3].name for r in rows},allowed)
        self.assertFalse({r[3].name for r in rows}&heldout)
        self.assertTrue(mask[np.arange(len(y)),y].all())
        for row in rows:
            state=row[3].start
            for a in row[5]:state=row[3].move(state,a)
            self.assertEqual(state,row[4])

    def test_copy_and_checkpoint_continue_exact_adam(self):
        x,y,mask,_=build_examples(self.puzzles[:4])
        net=PolicyNetwork((224,32,16,140));net.train_batch(x,y,mask)
        clone=net.copy()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.npz';net.save(path);loaded=PolicyNetwork.load(path)
        for other in [clone,loaded]:
            self.assertEqual(other.t,net.t)
            other.train_batch(x,y,mask)
        net.train_batch(x,y,mask)
        for other in [clone,loaded]:
            for key in ['w','b','m','v']:
                for a,b in zip(getattr(net,key),getattr(other,key)):np.testing.assert_array_equal(a,b)

    def test_learning_reduces_teacher_loss(self):
        x,y,mask,_=build_examples(self.puzzles[:4])
        net=PolicyNetwork((224,64,32,140));before=net.loss_grad(x,y,mask)[0]
        for _ in range(60):net.train_batch(x,y,mask)
        self.assertLess(net.loss_grad(x,y,mask)[0],before*.4)

    def test_rl_visualization_callback_does_not_change_learning(self):
        a,b=ExpandedRL(42),ExpandedRL(42)
        updates=[]
        first=episodes(a,self.puzzles[:4],[True]*4,np.random.default_rng(19),limit=32,epsilon=.7)
        second=episodes(b,self.puzzles[:4],[True]*4,np.random.default_rng(19),limit=32,epsilon=.7,
                        on_update=lambda loss,n:updates.append((loss,n,b.copy())))
        self.assertTrue(updates)
        self.assertEqual(first[:2],second[:2])
        for key in ['w','b','m','v']:
            for x,y in zip(getattr(a,key),getattr(b,key)):np.testing.assert_array_equal(x,y)
