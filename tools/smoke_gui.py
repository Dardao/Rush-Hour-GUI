"""Offscreen GUI and real worker test; writes screenshot only if requested."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from rushhour.gui import Window, configure_style
import rushhour.gui as gui
import numpy as np
from rushhour.core import demo_puzzles, split

app = QApplication([])
configure_style(app)
window = Window()
window.show()
errors = []
frames = {}
validation_weights = None
original_episodes = gui.episodes

def checked_episodes(net, puzzles, learning, rng, *args, **kwargs):
    frozen = kwargs.get('greedy', False)
    if frozen:
        assert not any(learning)
        assert len(puzzles) == 100
        before = [a.copy() for a in net.w + net.b + net.m + net.v]
        before_t = net.t
    result = original_episodes(net, puzzles, learning, rng, *args, **kwargs)
    if frozen:
        assert net.t == before_t
        for a, b in zip(before, net.w + net.b + net.m + net.v): np.testing.assert_array_equal(a, b)
        paths = result[0]
        for p, path in zip(puzzles, paths):
            state = p.start
            for action in path:
                logits = net.forward(p.encode(state)[None])[0]
                assert action == max(p.actions(state), key=lambda a: logits[a])
                state = p.move(state, action)
    return result

gui.episodes = checked_episodes

def run():
    window.puzzles = demo_puzzles()  # Test fixture only; application starts empty.
    for p in window.puzzles: p.minimum_possible = 1
    window.page = 1
    window.show_page()
    assert window.boards.learning is not None
    window.epochs.setValue(1)
    window.train()
    assert window.worker.puzzles == window.puzzles[100:200]
    window.worker.progress.connect(check_scope)
    window.worker.error.connect(errors.append)
    window.worker.finished.connect(done)

def check_scope(payload):
    global validation_weights
    if 'live' in payload:
        phase = payload['phase']
        frame = payload['live']
        learning = payload['learning']
        if phase == '학습 탐험':
            assert all(not frame['paths'][i] and frame['states'][i] == p.start
                       for i, p in enumerate(payload['puzzles']) if not learning[i])
        else:
            assert '학습 탐험' in frames
            assert phase == '전체 greedy 평가'
            if frame['step'] == 0:
                assert all(not path for path in frame['paths'])
                assert frame['states'] == [p.start for p in payload['puzzles']]
        frames[phase] = frame
    if 'metrics' in payload:
        m = payload['metrics']
        train, valid = split(window.puzzles[100:200])
        learning = window.boards.learning
        final = frames['전체 greedy 평가']['statuses']
        assert m['train_solved'] == sum(v == 'solved' for i, v in enumerate(final) if learning[i])
        assert m['solved'] == sum(v == 'solved' for i, v in enumerate(final) if not learning[i])
        assert m['efficiency_count'] == 100
        assert abs(m['efficiency_mean'] - sum(m['efficiency_values']) / 100) < 1e-12
        assert all(v == 0 for v, status in zip(m['efficiency_values'], final) if status != 'solved')
        assert m['episodes'] == len(train)
        assert abs(m['episode_reward'] - sum(frames['학습 탐험']['total_rewards'][i] for i in range(len(learning)) if learning[i])/len(train)) < 1e-9
        assert m['evaluated'] == len(valid)
        assert m['puzzle_names'] == [p.name for p in window.puzzles[100:200]]

def done():
    QTimer.singleShot(100, finish)

def finish():
    assert not errors, errors
    assert 'Epoch 1' in window.metrics_label.text(), window.metrics_label.text()
    assert window.boards.puzzles
    assert len(window.boards.puzzles) == 100
    assert window.boards.puzzles == window.puzzles[100:200]
    assert len(window.success_plot.history) == 1
    assert len(window.accuracy_plot.history) == 1
    assert len(window.efficiency_plot.history) == 1
    plots = [window.success_plot, window.accuracy_plot, window.efficiency_plot]
    assert max(p.width() for p in plots) - min(p.width() for p in plots) <= 1
    assert abs(sum(p.width() for p in plots) - window.heat.width()) < 12
    assert window.boards.learning is not None
    window.show_page()
    if len(sys.argv) > 1:
        window.grab().save(sys.argv[1])
    print('GUI + RL worker + validation + live exploration PASS')
    app.quit()

QTimer.singleShot(50, run)
QTimer.singleShot(60000, lambda: os._exit(2))
app.exec()
