import tempfile
import unittest
from pathlib import Path
import numpy as np
from rushhour.core import demo_puzzles, parse_line, split, examples
from rushhour.model import Network, rollout


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.puzzles = demo_puzzles()

    def test_solutions(self):
        for p in self.puzzles:
            state = p.start
            for action in p.solution:
                self.assertEqual(p.encode(state).shape, (224,))
                state = p.move(state, action)
            self.assertIsNone(state)

    def test_split(self):
        a, b = split(self.puzzles)
        self.assertTrue(a and b)
        self.assertFalse({p.group_key() for p in a} & {p.group_key() for p in b})

    def test_bad_data(self):
        with self.assertRaises(ValueError): parse_line('bad')

    def test_training(self):
        x, y, v, mask = examples(self.puzzles[:30])
        net = Network()
        first, _ = net.train_batch(x, y, v, mask)
        for _ in range(50): last, _ = net.train_batch(x, y, v, mask)
        self.assertLess(last, first*0.7)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.npz'; net.save(path)
            np.testing.assert_array_equal(net.forward(x), Network.load(path).forward(x))

    def test_rollout_is_legal(self):
        paths, statuses = rollout(Network(), self.puzzles[:20])
        for p, path in zip(self.puzzles, paths):
            state = p.start
            for a in path: state = p.move(state, a)
        self.assertEqual(len(statuses), 20)


if __name__ == '__main__': unittest.main()
