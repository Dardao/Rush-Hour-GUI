"""Real Qt workers: both training modes, model isolation and tab switching."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication
import rushhour.gui as gui
from rushhour.expert_policy import PolicyNetwork
from rushhour.expanded_rl import ExpandedRL

root=Path(__file__).resolve().parents[1]
app=QApplication([])
font=Path('/tmp/NotoSansCJKkr-Regular.otf')
if font.exists():QFontDatabase.addApplicationFont(str(font))
gui.configure_style(app);window=gui.TabbedWindow();window.resize(1800,1200);window.show()
errors=[];events={'supervised':set(),'rl':set()};metrics={'supervised':[],'rl':[]};stage=[0];switched=[False]
original_read=Path.read_text

def guarded_read(path,*args,**kwargs):
    if 'teacher_paths' in str(path) and stage[0]==1:raise AssertionError('RL attempted to read teacher data')
    return original_read(path,*args,**kwargs)
guard=patch.object(Path,'read_text',guarded_read);guard.start()


def receive(mode,payload):
    if 'phase_start' in payload:events[mode].add(payload['phase_start'])
    if 'update' in payload:events[mode].add('weight_update')
    if 'lesson' in payload:
        events[mode].add('teacher_batch')
        assert payload['net'].t>0 and payload['done']>0
        assert 0<=payload['probability']<=1
        if not switched[0]:
            window.tabs.setCurrentIndex(1);assert window.rl.net.t==0
            window.tabs.setCurrentIndex(0);switched[0]=True
        if 'screenshot' not in events[mode]:
            QTimer.singleShot(0,lambda:window.grab().save(str(root/'verification/supervised-live.png')));events[mode].add('screenshot')
    if 'live' in payload and mode=='rl':
        window.rl.page_slider.setValue(24);assert len(window.rl.boards.puzzles)==100
        window.rl.page_slider.setValue(0)
    if 'metrics' in payload:metrics[mode].append(payload['metrics'])


def launch():
    mode='supervised' if stage[0]==0 else 'rl';panel=window.supervised if stage[0]==0 else window.rl
    window.tabs.setCurrentIndex(stage[0]);panel.diagnostic_size.setCurrentIndex(0);panel.speed.setValue(1)
    worker=gui.Worker('supervised' if mode=='supervised' else 'train',panel.net.copy(),panel.puzzles,epochs=3,limit=30,diagnostic=True,diagnostic_count=4,augment=False)
    worker.progress.connect(lambda p:receive(mode,p));worker.error.connect(errors.append)
    panel.launch(worker);worker.finished.connect(lambda:QTimer.singleShot(80,finish))


def finish():
    try:
        assert not errors,errors
        if stage[0]==0:
            assert {'prepare','learn','evaluate','teacher_batch'}<=events['supervised']
            assert len(metrics['supervised'])==3 and window.supervised.net.t>0
            assert window.rl.net.t==0
            stage[0]=1;QTimer.singleShot(1,launch);return
        assert {'explore','learn','evaluate','weight_update'}<=events['rl']
        assert len(metrics['rl'])==3 and window.rl.net.t>0
        assert not np.shares_memory(window.supervised.net.w[0],window.rl.net.w[0])
        assert len(window.supervised.success_plot.history)==3 and len(window.rl.success_plot.history)==3
        window.grab().save(str(root/'verification/rl-live.png'))
        assert window.tabs.tabText(0)=='지도학습' and window.tabs.tabText(1)=='강화학습'
        out=dict(tab_order=['지도학습','강화학습'],independent_weights=True,independent_histories=True,teacher_file_blocked_during_rl=True,live_tab_switch=True,phases={k:sorted(v) for k,v in events.items()},epochs_per_mode=3)
        (root/'verification/tabs.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print('PASS',out,flush=True)
        guard.stop();window.close();app.exit(0)
    except Exception:
        import traceback;traceback.print_exc();guard.stop();app.exit(1)

QTimer.singleShot(50,launch);QTimer.singleShot(60000,lambda:os._exit(2))
with tempfile.TemporaryDirectory(prefix='rushhour-tabs-') as tmp:
    os.chdir(tmp);sys.exit(app.exec())
