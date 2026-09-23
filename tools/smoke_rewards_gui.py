"""Real GUI learning/evaluation phases, action source labels and full-scope wiring."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import rushhour.gui as gui
from rushhour.boarddata import load_initial_boards
from rushhour.expanded_rl import ExpandedRL

root=Path(__file__).resolve().parents[1];app=QApplication([]);gui.configure_style(app)
gui.Window.restore_apk=lambda self:None
window=gui.Window();window.puzzles,_=load_initial_boards(root/'resources/initial_boards.json');window.diagnostic_size.setCurrentIndex(0);window.show_page();window.show()
assert isinstance(window.net,ExpandedRL) and window.net.t==0
assert not window.heat.names[0].endswith('128')
errors=[];phases=set();sources=set();metrics=[];stage=[0];navigated=[False]

def progress(payload):
    if 'phase_start' in payload:phases.add(payload['phase_start'])
    if 'live' in payload:
        sources.update(payload['live']['selected_by'])
        if not navigated[0]:
            assert window.page_slider.isEnabled()
            window.page_slider.setValue(24);assert len(window.boards.puzzles)==100
            navigated[0]=True
    if 'metrics' in payload:metrics.append(payload['metrics'])

def launch():
    if stage[0]==0:
        worker=gui.Worker('train',window.net.copy(),window.puzzles,epochs=2,limit=20,diagnostic=True,diagnostic_count=4,augment=False)
    else:
        window.net=ExpandedRL();window.sync_model_view();window.diagnostic_size.setCurrentIndex(3);window.show_page()
        worker=gui.Worker('train',window.net.copy(),window.puzzles,epochs=1,limit=1,diagnostic=True,diagnostic_count=2500,augment=False)
    window.launch(worker);worker.progress.connect(progress);worker.error.connect(errors.append);worker.finished.connect(lambda:QTimer.singleShot(50,finish))

def finish():
    try:
        assert not errors,errors
        assert {'explore','learn','evaluate'}<=phases and {'random','greedy'}<=sources
        assert navigated[0]
        if stage[0]==0:
            assert len(metrics)==2 and metrics[-1]['train_evaluated']==4
            window.page_slider.setValue(0)
            assert window.boards.roles[:4]==['train']*4
            assert window.boards.roles[4:] == ['unused']*96
            assert all(s in ('random','greedy') for s in window.boards.action_sources[:4])
            stage[0]=1;QTimer.singleShot(10,launch);return
        final=metrics[-1]
        assert final['train_evaluated']==2500 and final['episodes']==2500 and final['evaluated']==0
        assert final['replay_capacity']==2500 and final['limit']==1
        assert not window.test_button.isEnabled()
        assert len(window.monitor_cache)==2500 and window.boards.roles==['train']*100
        result=dict(fresh_random_start=True,live_phases=sorted(phases),action_sources=sorted(sources),page_navigation=True,scopes_verified=[4,2500],full_scope_smoke_action_limit=1,full_scope_not_a_convergence_test=True)
        (root/'verification/gui-live.json').write_text(json.dumps(result,indent=2));print('PASS',result,flush=True);app.exit(0)
    except Exception:
        import traceback;traceback.print_exc();app.exit(1)

QTimer.singleShot(30,launch);QTimer.singleShot(60000,lambda:os._exit(2));sys.exit(app.exec())
