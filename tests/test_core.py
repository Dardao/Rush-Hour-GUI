import tempfile
import unittest
from pathlib import Path
import numpy as np
from rushhour.core import demo_puzzles, parse_line, split, split_all, split_role, permute_vehicles, examples
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
        train, valid, test = split_all(self.puzzles)
        groups = [{p.group_key() for p in part} for part in (train, valid, test)]
        self.assertTrue(all(groups))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertTrue(all(split_role(p) == role for part, role in zip((train, valid, test), ('train', 'validation', 'test')) for p in part))

    def test_vehicle_permutation_preserves_physics_and_remaps_action_slots(self):
        p = next(p for p in self.puzzles if len(p.cars) > 3)
        q = permute_vehicles(p, np.random.default_rng(9))
        self.assertEqual(q.cars[0], p.cars[0])
        self.assertEqual({(c.label, c.horizontal, c.length, c.lane, s) for c, s in zip(q.cars, q.start)},
                         {(c.label, c.horizontal, c.length, c.lane, s) for c, s in zip(p.cars, p.start)})
        for action in p.actions(p.start):
            old_i, code = divmod(action, 10)
            new_i = next(i for i, car in enumerate(q.cars) if car.label == p.cars[old_i].label)
            self.assertIn(new_i * 10 + code, q.actions(q.start))

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
