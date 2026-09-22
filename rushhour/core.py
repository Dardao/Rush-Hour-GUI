from dataclasses import dataclass
from collections import deque
import hashlib
import random
import re
import zipfile
from pathlib import Path

import numpy as np


MAX_MOVES = 300


@dataclass(frozen=True)
class Vehicle:
    label: str
    horizontal: bool
    length: int
    lane: int


@dataclass
class Puzzle:
    name: str
    difficulty: int
    cars: tuple
    start: tuple
    solution: tuple = ()
    source: str = 'synthetic'
    minimum_possible: int | None = None
    minimum_source: str = ''

    def board(self, state):
        grid = [-1] * 36
        for i, (car, pos) in enumerate(zip(self.cars, state)):
            for k in range(car.length):
                x, y = (pos + k, car.lane) if car.horizontal else (car.lane, pos + k)
                if not (0 <= x < 6 and 0 <= y < 6) or grid[y * 6 + x] != -1:
                    raise ValueError('Overlapping or out-of-bounds vehicle')
                grid[y * 6 + x] = i
        return grid

    def is_solved(self, state):
        """Solved as soon as every cell between the target and exit is clear."""
        if state is None:
            return True
        target = self.cars[0]
        grid = self.board(state)
        return all(grid[target.lane * 6 + x] == -1
                   for x in range(state[0] + target.length, 6))

    def actions(self, state):
        grid = self.board(state)
        result = []
        for i, (car, pos) in enumerate(zip(self.cars, state)):
            for direction, sign in enumerate((-1, 1)):
                for distance in range(1, 6):
                    tip = pos - distance if sign < 0 else pos + car.length - 1 + distance
                    if i == 0 and sign == 1 and tip == 6:
                        result.append(i * 10 + direction * 5 + distance - 1)
                        break  # Crossing the exit is a terminal action.
                    if tip < 0 or tip >= 6:
                        break
                    cell = car.lane * 6 + tip if car.horizontal else tip * 6 + car.lane
                    if grid[cell] != -1:
                        break
                    result.append(i * 10 + direction * 5 + distance - 1)
        return result

    def move(self, state, action):
        if action not in self.actions(state):
            raise ValueError(f'Illegal action {action}')
        i, code = divmod(int(action), 10)
        delta = (code % 5 + 1) * (-1 if code < 5 else 1)
        new = list(state)
        new[i] += delta
        if i == 0 and new[0] + self.cars[0].length > 6:
            return None
        return tuple(new)

    def encode(self, state):
        x = np.zeros((14, 16), dtype=np.float32)
        for i, (car, pos) in enumerate(zip(self.cars, state)):
            x[i, :4] = (1, i == 0, car.horizontal, car.length == 3)
            x[i, 4 + car.lane] = 1
            x[i, 10 + pos] = 1
        return x.ravel()

    def group_key(self):
        # Conservative topology grouping prevents cross-split reachable states.
        return tuple(sorted((c.horizontal, c.length, c.lane, i == 0)
                            for i, c in enumerate(self.cars)))


def app_level_name(index):
    # RushHour.CHALLENGE_TYPES = [625, 625, 625, 625].
    # Index is the reconstructed app ordinal, not the raw text line number.
    if not 1 <= index <= 2500:
        raise ValueError('APK app level must be between 1 and 2500')
    group, number = divmod(index - 1, 625)
    return f'{("Easy", "Medium", "Hard", "Expert")[group]}{number + 1:03d}'


def parse_line(line, index=1, include_solution=True):
    if not re.fullmatch(r'[0-3][a-z-]{36}(?:[a-z][udlr][1-5])+', line):
        raise ValueError(f'Line {index}: unsupported puzzle format')
    board, commands = line[1:37], line[37:]
    labels = ['x'] + sorted(set(board) - {'-', 'x'})
    if 'x' not in board or len(labels) > 14:
        raise ValueError(f'Line {index}: missing target or more than 14 vehicles')
    cars, state = [], []
    for label in labels:
        cells = [j for j, c in enumerate(board) if c == label]
        if len(cells) not in (2, 3):
            raise ValueError('Invalid vehicle length')
        horizontal = cells[1] == cells[0] + 1
        lane = cells[0] // 6 if horizontal else cells[0] % 6
        pos = cells[0] % 6 if horizontal else cells[0] // 6
        expected = [(lane * 6 + pos + k) if horizontal else (pos + k) * 6 + lane
                    for k in range(len(cells))]
        if cells != expected:
            raise ValueError('Non-contiguous vehicle')
        cars.append(Vehicle(label, horizontal, len(cells), lane))
        state.append(pos)
    if not cars[0].horizontal or cars[0].lane != 2 or cars[0].length != 2:
        raise ValueError('Expected a length-2 horizontal target in row 3')
    # Keep the raw difficulty metadata, but label by the app's level selector.
    p = Puzzle(app_level_name(index), int(line[0]), tuple(cars), tuple(state), source='APK reference')
    p.board(p.start)
    if not include_solution:
        p.source = 'APK initial state / RL'
        return p
    actions = []
    current = p.start
    for k in range(0, len(commands), 3):
        label, direction, distance = commands[k:k+3]
        i = labels.index(label)
        if (direction in 'lr') != cars[i].horizontal or current is None:
            raise ValueError('Invalid direction or commands after exit')
        action = i * 10 + (5 if direction in 'rd' else 0) + int(distance) - 1
        current = p.move(current, action)
        actions.append(action)
    if current is not None:
        raise ValueError('Reference solution does not reach the exit')
    p.solution = tuple(actions)
    return p


def load_apk(path):
    # Read data only. Never execute APK or extract its executable files.
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as z:
        info = z.getinfo('res/raw/puzzles.txt')
        if info.file_size > 10_000_000:
            raise ValueError('Puzzle resource exceeds 10 MB limit')
        data = z.read(info)
        lines = data.decode('utf-8-sig').splitlines()
        # Screenshot-verified profile: Easy001 is the DEX-embedded board;
        # raw lines 1/625/1250/1875 match Easy002/Medium001/Hard001/Expert001.
        # Restrict reconstruction to the exact audited APK, never guess for others.
        if digest == '89ed0e4dedaac48f9d0f63374d13b5b57aead2a4257ff539d3e4fe05cbd94771':
            candidates = re.findall(rb'[0-3][a-z-]{36}(?:[a-z][udlr][1-5])+', z.read('classes.dex'))
            if len(lines) != 2499 or len(candidates) != 1:
                raise ValueError('Audited APK ordering profile does not match')
            lines.insert(0, candidates[0].decode('ascii'))
        else:
            raise ValueError('이 APK는 앱 표시 순서가 검증되지 않았습니다. 지원 APK를 선택하세요.')
    # RL import retains initial boards only, never reference action sequences.
    puzzles = [parse_line(s.strip(), i + 1, include_solution=False) for i, s in enumerate(lines)]
    for i, (puzzle, line) in enumerate(zip(puzzles, lines)):
        puzzle.minimum_possible = 13 if i == 0 else sum(int(c) for c in line.strip()[39::3])
        puzzle.minimum_source = 'user-confirmed screen' if i == 0 else f'APK puzzles.txt line {i}'
    puzzles[0].source = 'APK DEX / screenshot-confirmed Easy001'
    return puzzles, digest


def shortest(p, start=None, limit=100000):
    start = p.start if start is None else start
    queue, parents = deque([start]), {start: None}
    while queue:
        state = queue.popleft()
        for a in p.actions(state):
            nxt = p.move(state, a)
            if nxt is None:
                path = [a]
                while parents[state] is not None:
                    state, previous = parents[state]
                    path.append(previous)
                return tuple(reversed(path))
            if nxt not in parents:
                parents[nxt] = (state, a)
                queue.append(nxt)
                if len(parents) > limit:
                    raise ValueError('BFS state limit reached')
    raise ValueError('Unsolvable puzzle')


def demo_puzzles(count=200, seed=7):
    """Independent synthetic layouts, not APK content. Intentionally easy smoke tests."""
    rng, result, seen = random.Random(seed), [], set()
    while len(result) < count:
        cars = [Vehicle('x', True, 2, 2)]
        state = [0]
        for j, lane in enumerate(rng.sample([2, 3, 4, 5], rng.randint(1, 4))):
            cars.append(Vehicle(chr(97+j), False, rng.choice([2, 3]), lane))
            state.append(rng.randint(0, 3))
        p = Puzzle(f'DEMO-{len(result)+1:03d}', 0, tuple(cars), tuple(state))
        for _ in range(rng.randint(4, 20)):
            candidates = [(a, p.move(tuple(state), a)) for a in p.actions(tuple(state))]
            candidates = [(a, s) for a, s in candidates if s is not None]
            if candidates:
                _, state = rng.choice(candidates)
        p.start = tuple(state)
        key = (p.cars, p.start)
        if key not in seen:
            seen.add(key)
            p.solution = shortest(p)
            result.append(p)
    return result


def split(puzzles):
    train, valid, test = split_all(puzzles)
    return train, valid + test


def split_role(puzzle):
    """Stable topology-level 80/10/10 assignment; no topology crosses splits."""
    bucket = int(hashlib.sha256(repr(puzzle.group_key()).encode()).hexdigest()[:8], 16) % 10
    return 'validation' if bucket == 0 else ('test' if bucket == 1 else 'train')


def split_all(puzzles):
    """Return train/validation/test pools without using positions or answers."""
    train, valid, test = [], [], []
    destinations = {'train': train, 'validation': valid, 'test': test}
    for p in puzzles:
        destinations[split_role(p)].append(p)
    return train, valid, test


def permute_vehicles(puzzle, rng):
    """Keep the target at slot 0 and permute all other vehicle/action slots."""
    if len(puzzle.cars) <= 2:
        order = list(range(len(puzzle.cars)))
    else:
        order = [0] + list(rng.permutation(np.arange(1, len(puzzle.cars))))
    return Puzzle(puzzle.name, puzzle.difficulty,
                  tuple(puzzle.cars[i] for i in order),
                  tuple(puzzle.start[i] for i in order),
                  source=puzzle.source, minimum_possible=puzzle.minimum_possible,
                  minimum_source=puzzle.minimum_source)


def examples(puzzles):
    xs, ys, vs, masks = [], [], [], []
    for p in puzzles:
        state = p.start
        for step, action in enumerate(p.solution):
            xs.append(p.encode(state))
            ys.append(action)
            vs.append((len(p.solution) - step) / 64.0)
            mask = np.zeros(140, dtype=bool)
            mask[p.actions(state)] = True
            masks.append(mask)
            state = p.move(state, action)
    return np.array(xs), np.array(ys), np.array(vs, dtype=np.float32), np.array(masks)


def display_moves(puzzle, path, solved=False, final_state=None):
    """Squares travelled; automatic exit reaches the edge then crosses by 1.

    Display accounting only. Does not create RL transitions or change rewards.
    """
    if final_state is not None:
        return sum(int(a) % 5 + 1 for a in path) + (max(0, 6 - (final_state[0] + puzzle.cars[0].length)) + 1 if solved else 0)
    state = puzzle.start
    total = 0
    for action in path:
        total += int(action) % 5 + 1
        state = puzzle.move(state, action)
        if state is None:
            return total
    if solved:
        total += max(0, 6 - (state[0] + puzzle.cars[0].length)) + 1
    return total
