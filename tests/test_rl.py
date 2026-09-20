import tempfile
import unittest
from pathlib import Path
import numpy as np
from rushhour.core import Puzzle, Vehicle
from rushhour.model import Network, policy, rollout
from rushhour.rl import episodes, reward, replay_rewards, epsilon_action, epsilon_at, vtrace_targets


class NoAnswer:
    def __init__(self, puzzle): self.puzzle = puzzle
    def __getattr__(self, name):
        if name == 'solution': raise AssertionError('RL accessed reference solution')
        return getattr(self.puzzle, name)


class RLTests(unittest.TestCase):
    def setUp(self):
        self.p = NoAnswer(Puzzle('test', 0, (Vehicle('x', True, 2, 2),), (4,)))

    def test_actor_direction(self):
        net = Network(); x = self.p.encode(self.p.start)[None]
        mask = np.zeros((1, 140), bool); mask[0, self.p.actions(self.p.start)] = True
        before = policy(net.forward(x), mask)[0, 5]
        target = net.forward(x)[0, 140] + 1
        net.actor_critic(x, np.array([5]), np.array([target]), mask, entropy_weight=0)
        self.assertGreater(policy(net.forward(x), mask)[0, 5], before)

    def test_rewards(self):
        self.assertEqual(reward(None, set()), 0.99)
        self.assertEqual(reward((1,), {(1,)}), -0.04)
        self.assertEqual(reward((2,), {(1,)}), -0.01)

    def test_validation_never_updates(self):
        net = Network(); before = [w.copy() for w in net.w+net.b]
        result = episodes(net, [self.p], [False], np.random.default_rng(3), limit=20)
        for a, b in zip(before, net.w+net.b): np.testing.assert_array_equal(a, b)
        self.assertEqual(net.t, 0)
        self.assertEqual(result[3], [])

    def test_trials_and_checkpoint(self):
        self.p = NoAnswer(Puzzle("blocked", 0, (Vehicle("x", True, 2, 2), Vehicle("a", False, 2, 3)), (0, 1)))
        net = Network(); frames = []
        result = episodes(net, [self.p]*8, [True]*8, np.random.default_rng(3), live=frames.append, limit=20)
        self.assertGreater(net.t, 0)
        self.assertTrue(np.isfinite(result[3]).all())
        self.assertTrue(frames)
        for path in result[0]:
            state = self.p.start
            for action in path: state = self.p.move(state, action)
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/'rl.npz'; net.save(file); loaded = Network.load(file)
            self.assertEqual(net.t, loaded.t)
            for a, b in zip(net.w+net.b+net.m+net.v, loaded.w+loaded.b+loaded.m+loaded.v):
                np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-12)

    def test_clear_exit_and_final_frame(self):
        p = Puzzle('blocked', 0, (Vehicle('x', True, 2, 2), Vehicle('a', False, 2, 3)), (0, 1))
        self.assertFalse(p.is_solved(p.start))
        class ClearNet:
            def forward(self, x):
                out = np.zeros((len(x), 141)); out[:, 16] = 100
                return out
        frames = []
        result = episodes(ClearNet(), [p], [False], np.random.default_rng(1), live=frames.append, greedy=True)
        self.assertEqual(result[0], [[16]])
        self.assertEqual(result[1], ['solved'])
        self.assertAlmostEqual(result[2][0], 0.93)
        self.assertEqual(frames[-1]['states'], [(0, 3)])
        self.assertEqual(rollout(ClearNet(), [p]), ([[16]], ['solved']))

    def test_initial_clear_needs_no_action(self):
        net = Network(); frames = []
        result = episodes(net, [self.p], [True], np.random.default_rng(1), live=frames.append)
        self.assertEqual(result[0], [[]])
        self.assertEqual(result[1], ['solved'])
        self.assertEqual(net.t, 0)
        self.assertEqual(frames[-1]['step'], 0)

    def test_default_limit_300(self):
        p = Puzzle('blocked', 0, (Vehicle('x', True, 2, 2), Vehicle('a', False, 2, 3)), (0, 1))
        class LoopNet:
            def forward(self, x):
                out = np.zeros((len(x), 141)); out[:, 0] = out[:, 5] = 1000
                return out
        result = episodes(LoopNet(), [p], [False], np.random.default_rng(1), epsilon=0)
        self.assertEqual(len(result[0][0]), 300)
        self.assertEqual(result[1], ['limit'])

    def test_distance_and_exit_rewards(self):
        self.assertAlmostEqual(reward((2,), set(), distance=3), -0.03)
        self.assertAlmostEqual(reward((2,), {(2,)}, distance=3), -0.06)
        self.assertAlmostEqual(reward((2,), set(), solved=True, distance=3, exit_distance=4), 0.93)
        p = Puzzle('easy1', 0, (Vehicle('x', True, 2, 2), Vehicle('a', False, 2, 3), Vehicle('b', True, 2, 4), Vehicle('o', False, 3, 4)), (2, 3, 4, 0))
        last, total = replay_rewards(p, [0, 10, 22, 15, 37])
        self.assertAlmostEqual(last, 0.93)
        self.assertAlmostEqual(total, 0.87)  # +1 minus all 13 displayed squares

    def test_epsilon_distribution(self):
        scores = np.array([0., 10., 1., -1., 100.])
        mask = np.array([True, True, True, True, False])
        rng = np.random.default_rng(6)
        actions = []
        for _ in range(10000):
            a, mu = epsilon_action(scores, mask, rng, 0.2)
            self.assertAlmostEqual(mu, .85 if a == 1 else .05)
            actions.append(a)
        self.assertNotIn(4, actions)
        self.assertAlmostEqual(actions.count(1)/len(actions), .85, delta=.02)
        self.assertEqual(epsilon_action(scores, mask, rng, 0), (1, 1))
        random_actions = [epsilon_action(scores, mask, rng, 1)[0] for _ in range(10000)]
        for a in range(4): self.assertAlmostEqual(random_actions.count(a)/10000, .25, delta=.02)

    def test_epsilon_schedule(self):
        self.assertEqual(epsilon_at(1,300), 1)
        self.assertAlmostEqual(epsilon_at(240,300), .05)
        self.assertAlmostEqual(epsilon_at(300,300), .05)
        self.assertEqual(epsilon_at(1,1), 1)

    def test_vtrace_targets(self):
        def transition(r, done, value=0, pi=.5, mu=.5):
            return (0, None, None, 0, r, done, value, pi, mu)
        targets, advantages = vtrace_targets([transition(-.1,False), transition(1,True)], np.zeros(1))
        np.testing.assert_allclose(targets,[.89,1])
        np.testing.assert_allclose(advantages,[.89,1])
        targets, advantages = vtrace_targets([transition(1,True,pi=.1,mu=.5)], np.zeros(1))
        np.testing.assert_allclose(targets,[.2])
        np.testing.assert_allclose(advantages,[.2])

    def test_cancel(self):
        net = Network()
        self.assertIsNone(episodes(net, [self.p], [True], np.random.default_rng(3), stopped=lambda:True))
        self.assertEqual(net.t, 0)


if __name__ == '__main__': unittest.main()
