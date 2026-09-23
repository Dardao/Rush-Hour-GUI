import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rushhour.seed_sweep import epoch_means


class SeedSweepTests(unittest.TestCase):
    def test_means_align_epochs_and_ignore_missing_validation(self):
        rows = epoch_means({1: [{'epoch': 1, 'loss': 2}, {'epoch': 2, 'loss': 4}],
                            2: [{'epoch': 1, 'loss': 6}]})
        self.assertEqual([r['loss'] for r in rows], [4, 4])
        self.assertEqual([r['n_seeds'] for r in rows], [2, 1])
        self.assertIsNone(rows[0]['validation_efficiency_mean'])

    def test_independent_runs_and_stop(self):
        from rushhour.gui import Worker
        from rushhour.seed_sweep import run_seed_sweep
        from rushhour.expert_policy import PolicyNetwork
        from rushhour.boarddata import load_initial_boards
        puzzles, _ = load_initial_boards(Path(__file__).resolve().parents[1]/'resources/initial_boards.json')
        class Signal:
            def __init__(self): self.events=[]
            def emit(self, value): self.events.append(value)
        class Harness:
            evaluate_groups = Worker.evaluate_groups
            callback = Worker.callback
            start_phase = Worker.start_phase
            def isInterruptionRequested(self):
                return self.stop_after_first and getattr(self,'seed',1)==2
        for stop in (False, True):
            w=Harness(); w.stop_after_first=stop
            w.net=PolicyNetwork((224,16,8,140),42); w.puzzles=puzzles
            w.seed_range=range(1,3);w.epochs=2;w.diagnostic_count=4
            w.live_delay=0;w.limit=10;w.progress=Signal();w.result=Signal()
            old=os.getcwd()
            with tempfile.TemporaryDirectory() as d:
                try:
                    os.chdir(d);run_seed_sweep(w)
                    data=json.loads((w.sweep_dir/'batch.json').read_text())
                    self.assertEqual(data['status'], 'interrupted' if stop else 'completed')
                    self.assertTrue((w.sweep_dir/'mean.csv').exists())
                    for run in data['runs']:
                        initial=PolicyNetwork.load(Path(run['checkpoint_dir'])/'initial.npz')
                        expected=PolicyNetwork((224,16,8,140),run['seed'])
                        self.assertEqual(initial.t,0)
                        np.testing.assert_array_equal(initial.w[0],expected.w[0])
                        records=[json.loads(l) for l in Path(run['log_path']).read_text().splitlines()]
                        self.assertEqual(records[0]['seed'],run['seed'])
                        for row in records[1:]:
                            self.assertIsNone(row['validation_efficiency_mean'])
                            self.assertEqual(row['seed'],run['seed'])
                    self.assertEqual(len(w.result.events),1)
                    if not stop:
                        self.assertEqual([r['epochs_completed'] for r in data['runs']],[2,2])
                        self.assertNotEqual(data['runs'][0]['log_path'],data['runs'][1]['log_path'])
                finally: os.chdir(old)
