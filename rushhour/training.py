"""Global curriculum and train-only bounded self-imitation replay."""
from collections import OrderedDict
from .rl import replay_rewards
from .core import split_all


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
    if 'selected_by' in frame:result['selected_by']=list(frame['selected_by'])
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


def easy_four(puzzles):
    """Select all four diagnostic puzzles, never silently accepting a partial set."""
    names = [f'Easy{i:03d}' for i in range(1, 5)]
    found = {p.name: p for p in puzzles}
    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError('Missing diagnostic puzzles: ' + ', '.join(missing))
    return [found[name] for name in names]


def diagnostic_split(puzzles, count=4):
    """Fixed diagnostic cohorts; holdouts never move into training."""
    if count == 4:
        return easy_four(puzzles), [], []
    if count in (2034,2500):
        if len(puzzles)!=2500:raise ValueError('전체 2500개 초기 보드가 필요합니다.')
        return (list(puzzles),[],[]) if count==2500 else split_all(puzzles)
    if count not in (20, 88):
        raise ValueError('Diagnostic size must be 4, 20 or 88')
    found = {p.name: p for p in puzzles}
    names = [f'Easy{i:03d}' for i in range(1, 101)]
    if any(name not in found for name in names):
        raise ValueError('Easy001–100 are required for this diagnostic')
    train, valid, test = split_all([found[name] for name in names])
    if len(train) < count:
        raise ValueError('Insufficient training puzzles in Easy001–100')
    return train[:count], valid, test
