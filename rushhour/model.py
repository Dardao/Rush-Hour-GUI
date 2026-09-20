"""Small CPU MLP with actual backpropagation + Adam. No mock training metrics."""
import numpy as np
from .core import MAX_MOVES


class Network:
    shapes = ((224, 128), (128, 64), (64, 141))

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.w = [(rng.standard_normal((i, o)) * np.sqrt(2 / i)).astype(np.float32) for i, o in self.shapes]
        self.b = [np.zeros(o, np.float32) for _, o in self.shapes]
        self.m = [np.zeros_like(p) for p in self.w + self.b]
        self.v = [np.zeros_like(p) for p in self.w + self.b]
        self.t = 0

    def copy(self):
        other = Network()
        other.w, other.b = [x.copy() for x in self.w], [x.copy() for x in self.b]
        other.m, other.v = [x.copy() for x in self.m], [x.copy() for x in self.v]
        other.t = self.t
        return other

    def actor_critic(self, x, actions, returns, mask, lr=0.0003, entropy_weight=0.02, policy_advantages=None):
        """Actor-critic update with detached targets/advantages; supports V-trace corrections."""
        h1 = np.maximum(x @ self.w[0] + self.b[0], 0)
        h2 = np.maximum(h1 @ self.w[1] + self.b[1], 0)
        out = h2 @ self.w[2] + self.b[2]
        prob = policy(out, mask)
        logp = np.log(np.maximum(prob, 1e-12))
        n = len(x)
        advantage = returns - out[:, 140] if policy_advantages is None else np.asarray(policy_advantages, dtype=np.float32)  # detached
        entropy = -(prob * logp).sum(1)
        actor_loss = -(logp[np.arange(n), actions] * advantage).mean()
        err = out[:, 140] - returns
        critic_loss = np.where(abs(err) < 1, 0.5*err**2, abs(err)-0.5).mean()
        grad = np.zeros_like(out)
        pg = prob.copy(); pg[np.arange(n), actions] -= 1
        grad[:, :140] = (pg * advantage[:, None] + entropy_weight * prob * (logp + entropy[:, None])) / n
        grad[:, 140] = 0.5 * np.clip(err, -1, 1) / n
        dw2, db2 = h2.T @ grad, grad.sum(0)
        d2 = (grad @ self.w[2].T) * (h2 > 0)
        dw1, db1 = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ self.w[1].T) * (h1 > 0)
        grads = [x.T @ d1, dw1, dw2, d1.sum(0), db1, db2]
        norm = np.sqrt(sum(float((g*g).sum()) for g in grads))
        self.t += 1
        for j, (p, g) in enumerate(zip(self.w + self.b, grads)):
            g = g * min(1.0, 1.0 / max(norm, 1e-12))
            self.m[j] = 0.9*self.m[j] + 0.1*g
            self.v[j] = 0.999*self.v[j] + 0.001*g*g
            p -= lr*(self.m[j]/(1-0.9**self.t))/(np.sqrt(self.v[j]/(1-0.999**self.t))+1e-8)
        return float(actor_loss + 0.5*critic_loss - entropy_weight*entropy.mean())

    def forward(self, x):
        h1 = np.maximum(x @ self.w[0] + self.b[0], 0)
        h2 = np.maximum(h1 @ self.w[1] + self.b[1], 0)
        return h2 @ self.w[2] + self.b[2]

    def train_batch(self, x, y, value, mask, lr=0.001):
        h1 = np.maximum(x @ self.w[0] + self.b[0], 0)
        h2 = np.maximum(h1 @ self.w[1] + self.b[1], 0)
        out = h2 @ self.w[2] + self.b[2]
        logits = np.where(mask, out[:, :140], -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        n = len(x)
        policy_loss = -np.log(np.maximum(prob[np.arange(n), y], 1e-12)).mean()
        err = out[:, 140] - value
        value_loss = np.where(abs(err) < 1, 0.5 * err**2, abs(err) - 0.5).mean()
        accuracy = (prob.argmax(1) == y).mean()
        grad = np.zeros_like(out)
        prob[np.arange(n), y] -= 1
        grad[:, :140] = prob / n
        grad[:, 140] = 0.5 * np.clip(err, -1, 1) / n
        dw2, db2 = h2.T @ grad, grad.sum(0)
        d2 = (grad @ self.w[2].T) * (h2 > 0)
        dw1, db1 = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ self.w[1].T) * (h1 > 0)
        dw0, db0 = x.T @ d1, d1.sum(0)
        self.t += 1
        for j, (p, g) in enumerate(zip(self.w + self.b, [dw0, dw1, dw2, db0, db1, db2])):
            g = np.clip(g, -5, 5)
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g*g
            p -= lr * (self.m[j] / (1 - 0.9**self.t)) / (np.sqrt(self.v[j] / (1 - 0.999**self.t)) + 1e-8)
        return float(policy_loss + 0.5 * value_loss), float(accuracy)

    def save(self, path):
        with open(path, 'wb') as f:
            np.savez_compressed(f, **{f'w{i}': w for i, w in enumerate(self.w)},
                                **{f'b{i}': b for i, b in enumerate(self.b)}, version=np.array([2]),
                                **{f'm{i}': a for i, a in enumerate(self.m)},
                                **{f'v{i}': a for i, a in enumerate(self.v)}, adam_step=np.array([self.t]))

    def action_accuracy(self, x, y, mask, stopped=lambda: False):
        """Evaluate a fixed epoch-end model on every reference state."""
        correct = 0
        for start in range(0, len(x), 512):
            if stopped():
                return None
            end = start + 512
            scores = np.where(mask[start:end], self.forward(x[start:end])[:, :140], -np.inf)
            correct += int((scores.argmax(1) == y[start:end]).sum())
        return correct / len(x) if len(x) else None

    @classmethod
    def load(cls, path):
        net = cls()
        with np.load(path, allow_pickle=False) as data:
            if data['version'].tolist() != [2]:
                raise ValueError('RL 전용 체크포인트가 필요합니다. 이전 지도학습 가중치는 불러오지 않습니다.')
            for prefix, params in [('w', net.w), ('b', net.b)]:
                for i, p in enumerate(params):
                    a = data[f'{prefix}{i}']
                    if a.shape != p.shape or not np.isfinite(a).all():
                        raise ValueError('Invalid checkpoint shape or non-finite weights')
                    p[:] = a
            for prefix, params in [('m', net.m), ('v', net.v)]:
                for i, p in enumerate(params):
                    a = data[f'{prefix}{i}']
                    if a.shape != p.shape or not np.isfinite(a).all() or (prefix == 'v' and (a < 0).any()):
                        raise ValueError('Invalid optimizer state')
                    p[:] = a
            net.t = int(data['adam_step'][0])
            if net.t < 0: raise ValueError('Invalid optimizer step')
        return net


def policy(out, mask):
    logits = np.where(mask, out[:, :140], -1e9)
    logits = logits - logits.max(1, keepdims=True)
    prob = np.exp(logits) * mask
    return prob / np.maximum(prob.sum(1, keepdims=True), 1e-12)


def rollout(net, puzzles, limit=MAX_MOVES, stopped=lambda: False):
    states = [p.start for p in puzzles]
    paths = [[] for _ in puzzles]
    statuses = ['solved' if p.is_solved(p.start) else 'running' for p in puzzles]
    seen = [{p.start} for p in puzzles]
    for _ in range(limit):
        if stopped():
            break
        active = [i for i, s in enumerate(statuses) if s == 'running']
        if not active:
            break
        predictions = net.forward(np.stack([puzzles[i].encode(states[i]) for i in active]))
        for row, i in enumerate(active):
            p = puzzles[i]
            legal = p.actions(states[i])
            if not legal:
                statuses[i] = 'blocked'
                continue
            a = max(legal, key=lambda a: predictions[row, a])
            paths[i].append(a)
            states[i] = p.move(states[i], a)
            if p.is_solved(states[i]):
                statuses[i] = 'solved'
            elif states[i] in seen[i]:
                statuses[i] = 'loop'
            else:
                seen[i].add(states[i])
    statuses = [s if s != 'running' else 'limit' for s in statuses]
    return paths, statuses
