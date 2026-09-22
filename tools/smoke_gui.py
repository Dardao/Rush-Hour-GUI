"""Real QThread training, frozen evaluation, live page navigation and test isolation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import rushhour.gui as gui
from rushhour.core import demo_puzzles, split_all, split_role

app = QApplication([]); gui.configure_style(app)
gui.Window.restore_apk = lambda self: None
window = gui.Window(); window.show()
errors, rows, phases = [], [], set()
original_episodes = gui.episodes
original_replay = gui.replay_successes
navigation_checked = False


def checked_episodes(net, puzzles, learning, rng, *args, **kwargs):
    frozen = kwargs.get('greedy', False)
    assert len(puzzles) <= 100
    if any(learning): assert all(split_role(p) == 'train' for p in puzzles)
    if frozen:
        assert not any(learning)
        before = [a.copy() for a in net.w+net.b+net.m+net.v]; before_t = net.t
    result = original_episodes(net, puzzles, learning, rng, *args, **kwargs)
    if frozen:
        assert net.t == before_t
        for a,b in zip(before, net.w+net.b+net.m+net.v): np.testing.assert_array_equal(a,b)
        if result is not None:
            for p,path in zip(puzzles,result[0]):
                state = p.start
                for a in path:
                    out = net.forward(p.encode(state)[None])[0]
                    assert a == max(p.actions(state), key=lambda action: out[action])
                    state = p.move(state,a)
    return result


def checked_replay(net, records, *args, **kwargs):
    assert all(split_role(p) == 'train' for p,path in records)
    return original_replay(net,records,*args,**kwargs)


gui.episodes = checked_episodes; gui.replay_successes = checked_replay


def check(payload):
    global navigation_checked
    try:
        if 'phase_start' in payload: phases.add(payload['phase_start'])
        if 'live' in payload:
            if not navigation_checked:
                assert window.page_slider.isEnabled()
                window.page_slider.setValue(2)
                assert window.boards.puzzles == window.puzzles[200:300]
                navigation_checked = True
            assert window.page == 2
            assert window.boards.puzzles == window.puzzles[200:300]
            if payload['phase'] in ('explore','evaluate'):
                assert all(split_role(p) != 'test' for p in payload['puzzles'])
            for p,state,path in zip(payload['puzzles'],payload['live']['states'],payload['live']['paths']):
                current = p.start
                for a in path:
                    nxt = p.move(current,a)
                    if nxt is not None: current = nxt
                assert current == state
        if 'metrics' in payload:
            m=payload['metrics']; rows.append(m)
            train,valid,test=split_all(window.puzzles)
            assert m['train_evaluated']==len(train) and m['evaluated']==len(valid)
            assert m['episodes']==m['curriculum_eligible']
            assert len(m['training_names'])==len(set(m['training_names']))
            assert m['efficiency_count']==len(train)+len(valid)
            assert abs(m['efficiency_mean']-sum(m['efficiency_values'])/m['efficiency_count'])<1e-10
            assert not {p.name for p in test} & set(m['puzzle_names'])
    except Exception as exc:
        errors.append(repr(exc))
        window.worker.requestInterruption()


def run():
    window.puzzles=demo_puzzles(220)
    for i,p in enumerate(window.puzzles): p.minimum_possible=15+(i%4)*15
    window.show_page()
    assert window.page_slider.maximum()==2
    worker=gui.Worker('train',window.net.copy(),window.puzzles,epochs=2,limit=20)
    window.launch(worker)
    worker.progress.connect(check); worker.error.connect(errors.append); worker.finished.connect(done)


def done(): QTimer.singleShot(100,finish)


def finish():
    assert not errors, errors
    assert len(rows)==2 and phases=={'explore','evaluate'} and navigation_checked
    assert len(window.success_plot.history)==2
    assert len(window.efficiency_plot.history)==2
    assert window.boards.puzzles==window.puzzles[200:300]
    window.page_slider.setValue(0)
    assert window.boards.puzzles==window.puzzles[:100]
    assert all(window.boards.statuses[i] != 'ready' for i,p in enumerate(window.boards.puzzles) if split_role(p)!='test')
    assert all(window.boards.statuses[i] == 'ready' for i,p in enumerate(window.boards.puzzles) if split_role(p)=='test')
    plots=(window.success_plot,window.accuracy_plot,window.efficiency_plot)
    assert len({p.height() for p in plots})==1
    if len(sys.argv)>1: window.grab().save(sys.argv[1])
    # Explicit held-out test uses frozen copies and does not append training graphs.
    worker=gui.Worker('test',window.net.copy(),split_all(window.puzzles)[2],limit=20)
    result=[]; worker.error.connect(errors.append); worker.result.connect(result.append)
    worker.run()
    assert not errors and result and result[0][0]=='test'
    assert result[0][1]['count']==len(split_all(window.puzzles)[2])
    print('PASS: global train/val isolation, replay, canonical animation, live slider, cache and explicit test')
    app.quit()


QTimer.singleShot(30,run)
QTimer.singleShot(60000,lambda:os._exit(2))
app.exec()
