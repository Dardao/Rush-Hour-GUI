"""Independent supervised repetitions and epoch-aligned summaries."""
import csv
import json
import time
from pathlib import Path
from .expert_policy import PolicyNetwork

METRICS = ('train_success_rate', 'validation_success_rate', 'action_accuracy',
           'loss', 'train_efficiency_mean', 'validation_efficiency_mean')


def epoch_means(histories):
    grouped = {}
    for history in histories.values():
        for row in history:
            grouped.setdefault(row['epoch'], []).append(row)
    result = []
    for epoch, rows in sorted(grouped.items()):
        out = {'epoch': epoch, 'n_seeds': len(rows)}
        for key in METRICS:
            values = [r[key] for r in rows if r.get(key) is not None]
            out[key] = sum(values) / len(values) if values else None
        result.append(out)
    return result


def run_seed_sweep(w):
    from .supervised import run_supervised
    folder = Path(f'runs/supervised-sweep-{time.time_ns()}')
    folder.mkdir(parents=True)
    w.sweep_dir = folder
    widths = tuple(w.net.widths)
    manifest = dict(seeds=list(w.seed_range), epochs=w.epochs, widths=widths,
                    diagnostic_count=w.diagnostic_count, runs=[], status='running')
    histories = {}
    def save():
        temp = folder / 'batch.tmp'
        temp.write_text(json.dumps(manifest, indent=2)); temp.replace(folder / 'batch.json')
        with (folder / 'mean.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'n_seeds', *METRICS])
            writer.writeheader(); writer.writerows(epoch_means(histories))
    save()
    try:
        for index, seed in enumerate(w.seed_range, 1):
            if w.isInterruptionRequested(): break
            w.seed = seed
            w.net = PolicyNetwork(widths, seed=seed)
            w.epoch = 0
            w.progress.emit(dict(seed_start=seed, seed_index=index, seed_total=len(w.seed_range)))
            run = run_supervised(w, emit_result=False)
            histories[seed] = [{k: row.get(k) for k in ('epoch', *METRICS)}
                               for row in run.pop('history')]
            manifest['runs'].append(run)
            save()
        manifest['status'] = 'interrupted' if w.isInterruptionRequested() else 'completed'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        raise
    finally:
        save()
    w.result.emit(('supervised', w.net.copy()))
