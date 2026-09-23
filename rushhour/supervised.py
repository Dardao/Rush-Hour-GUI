"""Teacher data and training, imported only by the supervised worker mode."""
import json
import math
import time
from pathlib import Path

import numpy as np

from .boarddata import load_initial_boards
from .core import display_moves
from .training import diagnostic_split


def build_examples(puzzles):
    root = Path(__file__).resolve().parents[1] / 'resources'
    teachers = json.loads((root / 'supervised/teacher_paths.json').read_text())['paths']
    reference = {p.name: p for p in load_initial_boards(root / 'initial_boards.json')[0]}
    records = {}
    for p in puzzles:
        original = reference.get(p.name)
        if original is None or original.cars != p.cars or original.start != p.start:
            raise ValueError(f'{p.name}: bundled teacher does not match this board')
        state, prefix, trajectory = p.start, [], []
        for action in teachers[p.name]:
            if p.is_solved(state):
                raise ValueError('Teacher has actions after success')
            legal = p.actions(state)
            if action not in legal:
                raise ValueError(f'{p.name}: invalid teacher action')
            mask = np.zeros(140, bool); mask[legal] = True
            x = p.encode(state).astype(np.uint8)
            trajectory.append((x, action, mask, p, state, tuple(prefix)))
            prefix.append(action); state = p.move(state, action)
        if not p.is_solved(state):
            raise ValueError(f'{p.name}: incomplete teacher trajectory')
        for j, row in enumerate(trajectory):
            key, remaining = row[0].tobytes(), len(trajectory) - j
            if key in records and records[key][1] <= remaining:
                continue
            records[key] = (row, remaining)
    rows = [row for row, _ in records.values()]
    return (np.stack([r[0] for r in rows]).astype(np.float32),
            np.array([r[1] for r in rows]), np.stack([r[2] for r in rows]), rows)


def action_text(puzzle, action):
    car, code = divmod(int(action), 10)
    direction = ('←' if code < 5 else '→') if puzzle.cars[car].horizontal else ('↑' if code < 5 else '↓')
    return f'{puzzle.cars[car].label.upper()} {direction} {code % 5 + 1}칸'


def run_supervised(worker):
    w = worker
    if not getattr(w.net, 'policy_only', False):
        raise ValueError('Supervised training requires its own policy model')
    train, valid, test = diagnostic_split(w.puzzles, w.diagnostic_count)
    w.start_phase('prepare', '지도학습 데이터 확인 · 선택한 학습 문제의 교사 경로만 사용')
    x, y, mask, examples = build_examples(train)
    w.net.training_scope = f'supervised_train_{len(train)}_validation_{len(valid)}_test_{len(test)}'
    rng = np.random.default_rng(w.seed)
    run_id = time.time_ns()
    w.artifact_dir = Path(f'checkpoints/supervised-{run_id}')
    w.artifact_dir.mkdir(parents=True)
    Path('runs').mkdir(exist_ok=True)
    log_path = Path(f'runs/supervised-{run_id}.jsonl')
    w.net.save(w.artifact_dir / 'initial.npz')
    history = [dict(type='manifest', method='supervised', seed=w.seed,
                    training_names=[p.name for p in train], validation_names=[p.name for p in valid],
                    test_names=[p.name for p in test], examples=len(x), epochs=w.epochs,
                    widths=list(w.net.widths), teacher_source='resources/supervised/teacher_paths.json')]
    best = -1
    with log_path.open('w') as log:
        log.write(json.dumps(history[0]) + '\n')
        for w.epoch in range(1, w.epochs + 1):
            if w.isInterruptionRequested(): break
            w.start_phase('learn', f'Epoch {w.epoch} · 교사 행동 학습 · {len(x):,}개 상태 · batch 512')
            order, losses, correct, seen = rng.permutation(len(x)), 0.0, 0.0, 0
            lr = .001 * (.02 + .98 * (1 + math.cos(math.pi * (w.epoch - 1) / max(1, w.epochs - 1))) / 2)
            last_emit = 0.0
            for start in range(0, len(order), 512):
                if w.isInterruptionRequested(): break
                ids = order[start:start + 512]
                # Predictions shown here are before this batch's weight update.
                emit = time.monotonic() - last_emit >= .15 or start + 512 >= len(order) or w.live_delay
                if emit:
                    idx = int(ids[0]); scores = w.net.forward(x[idx:idx+1])[0, :140]
                    legal = np.flatnonzero(mask[idx]); predicted = int(legal[np.argmax(scores[legal])])
                    exp = np.exp(scores[legal] - scores[legal].max()); probabilities = exp / exp.sum()
                    prob = float(probabilities[np.flatnonzero(legal == y[idx])[0]])
                loss, accuracy = w.net.train_batch(x[ids], y[ids], mask[ids], lr=lr)
                losses += loss * len(ids); correct += accuracy * len(ids); seen += len(ids)
                if emit:
                    last_emit = time.monotonic()
                    # At most one current teacher state per board in this batch.
                    samples = {examples[int(i)][3].name: examples[int(i)] for i in ids}
                    samples[examples[idx][3].name]=examples[idx]
                    states = [dict(puzzle=r[3], state=r[4], path=r[5], teacher=r[1]) for r in samples.values()]
                    w.progress.emit(dict(lesson=states, sample=examples[idx][3].name,
                                         target=action_text(examples[idx][3], y[idx]),
                                         predicted=action_text(examples[idx][3], predicted), probability=prob,
                                         epoch=w.epoch, done=seen, total=len(x), loss=losses/seen,
                                         accuracy=correct/seen, update_count=w.net.t, net=w.net.copy()))
                    if w.live_delay: w.msleep(w.live_delay)
            if w.isInterruptionRequested(): break
            w.start_phase('evaluate', f'Epoch {w.epoch} · 가중치 고정 · 학습 {len(train)} + 검증 {len(valid)} greedy 평가')
            evaluated = train + valid
            result = w.evaluate_groups(w.net.copy(), evaluated, rng)
            if result is None or w.isInterruptionRequested(): break
            paths, statuses = result
            solved = statuses[:len(train)].count('solved')
            vals = statuses[len(train):]
            moves = [display_moves(p, path, s == 'solved') for p, path, s in zip(evaluated, paths, statuses)]
            m = dict(supervised=True, epoch=w.epoch, loss=losses/seen, action_accuracy=correct/seen,
                     train_solved=solved, train_evaluated=len(train), train_success_rate=solved/len(train),
                     solved=vals.count('solved'), evaluated=len(valid),
                     validation_success_rate=vals.count('solved')/len(valid) if valid else None,
                     efficiency_mean=float(np.mean([p.minimum_possible/n if s == 'solved' else 0
                         for p,n,s in zip(evaluated,moves,statuses)])),
                     examples=seen, lr=lr, update_count=w.net.t,
                     puzzle_names=[p.name for p in evaluated], statuses=statuses, moves=moves)
            if solved > best:
                best = solved; w.net.save(w.artifact_dir / 'best.npz')
            history.append(m); log.write(json.dumps(m)+'\n'); log.flush()
            w.progress.emit({'metrics': m, 'net': w.net.copy()})
    temp = log_path.with_suffix('.complete.tmp')
    temp.write_text(''.join(json.dumps(r)+'\n' for r in history)); temp.replace(log_path)
    w.net.save(w.artifact_dir / 'final.npz')
    (w.artifact_dir/'run.json').write_text(json.dumps(dict(log_path=str(log_path),
        interrupted=w.isInterruptionRequested(), epochs_completed=len(history)-1), indent=2))
    w.result.emit(('supervised', w.net.copy()))
