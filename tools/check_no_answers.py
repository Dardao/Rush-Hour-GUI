"""Metamorphic test: answer metadata cannot influence learning or selection."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PySide6.QtCore import QCoreApplication
import rushhour.gui as gui
from rushhour.boarddata import load_initial_boards
from rushhour.expanded_rl import ExpandedRL
from rushhour.expert_policy import PolicyNetwork

root=Path(__file__).resolve().parents[1];app=QCoreApplication([]);ps,_=load_initial_boards(root/'resources/initial_boards.json')
original=gui.episodes;original_replay=gui.replay_successes;source_counts={'random':0,'greedy':0};training_calls=[0]
def guarded(net,puzzles,learning,rng,*args,**kwargs):
    if any(learning):
        assert all(p.minimum_possible is None and p.solution==() for p in puzzles)
        training_calls[0]+=1
    before=[a.copy() for a in net.w+net.b+net.m+net.v] if kwargs.get('greedy') else None
    result=original(net,puzzles,learning,rng,*args,**kwargs)
    if before is not None:
        for a,b in zip(before,net.w+net.b+net.m+net.v):np.testing.assert_array_equal(a,b)
    return result
def guarded_replay(net,records,*args,**kwargs):
    assert all(p.minimum_possible is None and p.solution==() for p,path in records)
    return original_replay(net,records,*args,**kwargs)
def run(puzzles):
    rows=[];errors=[];results=[]
    w=gui.Worker('train',ExpandedRL(),puzzles,epochs=2,limit=20,diagnostic=True,diagnostic_count=4,augment=False)
    def on_progress(payload):
        if 'metrics' in payload:rows.append(payload['metrics'])
        if 'live' in payload:
            for source in payload['live']['selected_by']:
                if source in source_counts:source_counts[source]+=1
    w.progress.connect(on_progress);w.error.connect(errors.append);w.result.connect(results.append);w.run()
    assert not errors,errors
    return results[-1][1],rows,ExpandedRL.load(w.artifact_dir/'best.npz')

with tempfile.TemporaryDirectory() as temp,patch.object(gui,'episodes',guarded),patch.object(gui,'replay_successes',guarded_replay),patch.object(PolicyNetwork,'load',side_effect=AssertionError('Teacher weights requested')):
    old=Path.cwd();os.chdir(temp)
    a,rows_a,best_a=run(ps)
    changed=[replace(p,minimum_possible=999999,minimum_source='deliberately false',solution=(139,139)) for p in ps]
    b,rows_b,best_b=run(changed)
    os.chdir(old)
for x,y in [(a,b),(best_a,best_b)]:
    for k in ['w','b','m','v']:
        for p,q in zip(getattr(x,k),getattr(y,k)):np.testing.assert_array_equal(p,q)
assert [r['checkpoint_score_reward'] for r in rows_a]==[r['checkpoint_score_reward'] for r in rows_b]
assert source_counts['random'] and source_counts['greedy']
result=dict(training_calls=training_calls[0],teacher_load_blocked=True,answer_metadata_removed=True,false_minima_and_paths_do_not_change_weights_or_best_selection=True,frozen_evaluation=True,action_source_frames=source_counts)
(root/'verification/no-answers.json').write_text(json.dumps(result,indent=2));print('PASS',result)
