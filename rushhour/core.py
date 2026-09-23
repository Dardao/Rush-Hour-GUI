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


def parse_line(*args, **kwargs):
    raise RuntimeError("parse_line is disabled in the rewards-only research package")


def load_apk(*args, **kwargs):
    raise RuntimeError("load_apk is disabled in the rewards-only research package")


def shortest(*args, **kwargs):
    raise RuntimeError("shortest is disabled in the rewards-only research package")


def demo_puzzles(*args, **kwargs):
    raise RuntimeError("demo_puzzles is disabled in the rewards-only research package")


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


def examples(*args, **kwargs):
    raise RuntimeError("examples is disabled in the rewards-only research package")


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
