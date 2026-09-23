import unittest
from unittest.mock import patch
import numpy as np
from rushhour.core import Puzzle, Vehicle, permute_vehicles, split_all
from rushhour.model import Network
from rushhour.rl import episodes, replay_successes, replay_rewards
from rushhour.training import curriculum_pool, canonical_frame, SuccessReplay, easy_four, diagnostic_split


class TrainingTests(unittest.TestCase):
    def fixture(self, name='p', minimum=20):
        return Puzzle(name, 0, (Vehicle('x', True, 2, 2), Vehicle('a', False, 2, 3),
                      Vehicle('b', True, 2, 5)), (0, 1, 0), minimum_possible=minimum)

    def test_easy_four_exact_scope(self):
        puzzles = [self.fixture(f'Easy{i:03d}', 28) for i in range(1, 6)]
        self.assertEqual(easy_four(list(reversed(puzzles))), puzzles[:4])
        with self.assertRaises(ValueError):
            easy_four(puzzles[1:])

    def test_diagnostic_rejects_incomplete_larger_cohort(self):
        with self.assertRaises(ValueError):
            diagnostic_split([self.fixture('Easy001')], 20)
        with self.assertRaises(ValueError):
            diagnostic_split([], 5)

    def test_fixed_replay_never_permutes_and_still_learns(self):
        p = self.fixture()
        net = Network()
        before = [w.copy() for w in net.w]
        with patch('rushhour.rl.permute_vehicles', side_effect=AssertionError('Unexpected permutation')):
            losses, used, steps = replay_successes(net, [(p, (16,))], np.random.default_rng(42), augment=False)
        self.assertEqual((used, steps), (1, 1))
        self.assertTrue(losses)
        self.assertTrue(any(np.any(a != b) for a, b in zip(before, net.w)))

    def test_curriculum_cumulative(self):
        puzzles = [self.fixture(str(n), n) for n in (13, 25, 26, 35, 36, 45, 46, 93)]
        expected = (2, 4, 6, 8)
        for epoch, count in zip((1, 26, 51, 76), expected):
            eligible, ceiling, fallback = curriculum_pool(puzzles, epoch, 100)
            self.assertEqual(len(eligible), count)
            self.assertFalse(fallback)
        self.assertEqual(curriculum_pool(puzzles, 1, 1)[0], puzzles)
        self.assertTrue(curriculum_pool([self.fixture(minimum=90)], 1, 100)[2])

    def test_permutation_full_trajectory_and_rewards(self):
        p = self.fixture(); q = permute_vehicles(p, np.random.default_rng(4))
        a = next(i for i, car in enumerate(q.cars) if car.label == 'a')*10+6
        state = q.move(q.start, a)
        frame = canonical_frame([p], [q], {'states': [state], 'paths': [[a]], 'statuses': ['solved'],
            'last_rewards': [.93], 'total_rewards': [.93], 'step': 1})
        self.assertEqual(frame['paths'], [[16]])
        self.assertEqual(frame['states'], [p.move(p.start, 16)])
        self.assertEqual(replay_rewards(p, [16]), replay_rewards(q, [a]))
        self.assertEqual(q.solution, ())
        self.assertEqual([len(g) for g in split_all([p])], [len(g) for g in split_all([q])])

    def test_replay_scope_dedup_capacity_and_validity(self):
        a, b, c = [self.fixture(n) for n in 'abc']
        replay = SuccessReplay([a, b], capacity=1)
        with self.assertRaises(ValueError): replay.add(c, [16])
        with self.assertRaises(ValueError): replay.add(a, [5])
        with self.assertRaises(ValueError): replay.add(a, [16, 5])
        replay.add(a, [16]); replay.add(a, [16])
        self.assertEqual(len(replay), 1)
        replay.add(b, [16])
        self.assertEqual(replay.records()[0][0], b)

    def test_positive_replay_hand_return_and_no_bad_advantage(self):
        p = self.fixture()
        class Capture:
            def __init__(self, value): self.value = value; self.calls = []
            def forward(self, x):
                out = np.zeros((len(x),141)); out[:,140] = self.value; return out
            def actor_critic(self, x, actions, returns, mask, **kw):
                self.calls.append((returns, kw)); return 0.
        net = Capture(0)
        _, used, steps = replay_successes(net, [(p, (16,))], np.random.default_rng(1))
        self.assertEqual((used, steps), (1,1))
        np.testing.assert_allclose(net.calls[0][0], [.93])
        np.testing.assert_allclose(net.calls[0][1]['policy_advantages'], [.93])
        net = Capture(2)
        replay_successes(net, [(p, (16,))], np.random.default_rng(1))
        self.assertFalse(net.calls)
        for path in ((5,), (139,), ()):
            self.assertEqual(replay_successes(net, [(p, path)], np.random.default_rng(1))[1:], (0,0))


if __name__ == '__main__': unittest.main()
