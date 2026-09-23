"""Research inputs: initial geometry only, no APK/minima/expert actions."""
import json
from pathlib import Path
from .core import Puzzle, Vehicle

ROOT = Path(__file__).resolve().parents[1]

def boards():
    rows = json.loads((ROOT/'resources/boards.json').read_text())['puzzles']
    out = []
    for r in rows:
        if set(r) != {'name', 'cars', 'start'}:
            raise ValueError('Only geometry and display names allowed')
        p = Puzzle(r['name'], 0, tuple(Vehicle(**c) for c in r['cars']), tuple(r['start']))
        p.board(p.start)
        out.append(p)
    assert len(out) == len({p.name for p in out}) == 2500
    return out

def erase_loops(p, path):
    states, actions, indices = [p.start], [], {p.start: 0}
    for a in path:
        nxt = p.move(states[-1], a)
        if nxt in indices:
            end = indices[nxt]
            for s in states[end+1:]:
                del indices[s]
            states, actions = states[:end+1], actions[:end]
        else:
            actions.append(int(a)); states.append(nxt); indices[nxt] = len(states)-1
    assert p.is_solved(states[-1])
    return actions

def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, separators=(',', ':')))
    temp.replace(path)
