"""Global curriculum and train-only bounded self-imitation replay."""
from collections import OrderedDict
from .rl import replay_rewards


def curriculum_pool(train, epoch, epochs):
    # Cumulative thresholds keep easier puzzles in every later stage.
    fraction = epoch / max(1, epochs)
    ceiling = 25 if fraction <= .25 else (35 if fraction <= .5 else (45 if fraction <= .75 else None))
    eligible = [p for p in train if ceiling is None or
                (p.minimum_possible is not None and p.minimum_possible <= ceiling)]
    # Custom data with no eligible metadata gets a documented full-pool fallback.
    fallback = not eligible
    return (eligible or list(train)), ceiling, fallback


def canonical_actions(original, augmented, path):
    slots = {c.label: i for i, c in enumerate(original.cars)}
    return [slots[augmented.cars[int(a)//10].label]*10+int(a)%10 for a in path]


def canonical_frame(originals, augmented, frame):
    """Undo slot augmentation for display without changing the learner's actions."""
    result = {k: list(frame[k]) for k in ('states', 'paths', 'statuses', 'last_rewards', 'total_rewards')}
    result['step'] = frame['step']
    for j, (p, q) in enumerate(zip(originals, augmented)):
        slots = {c.label: i for i, c in enumerate(q.cars)}
        result['states'][j] = tuple(frame['states'][j][slots[c.label]] for c in p.cars)
        result['paths'][j] = canonical_actions(p, q, frame['paths'][j])
    return result


class SuccessReplay:
    def __init__(self, train, capacity=500):
        self.allowed = {p.name: p for p in train}
        self.capacity = capacity
        self.entries = OrderedDict()

    def add(self, puzzle, path):
        p = self.allowed.get(puzzle.name)
        if p is not puzzle:
            raise ValueError('Replay accepts canonical training puzzles only')
        if not path: return  # No decisions to reinforce in initially solved boards.
        state = p.start
        for index, action in enumerate(path):
            state = p.move(state, action)
            if p.is_solved(state):
                if index != len(path)-1: raise ValueError('Actions after success')
                break
        if not p.is_solved(state): raise ValueError('Replay requires a solved trajectory')
        score = replay_rewards(p, path)[1]
        existing = self.entries.get(p.name)
        if existing and existing[2] >= score: return
        self.entries[p.name] = (p, tuple(path), score)
        self.entries.move_to_end(p.name)
        while len(self.entries) > self.capacity: self.entries.popitem(last=False)

    def records(self):
        return [(p, path) for p, path, _ in self.entries.values()]

    def __len__(self):
        return len(self.entries)
