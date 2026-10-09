"""Fixed 14-checkpoint queue; resumes completed runs and preserves failed logs."""
from __future__ import annotations
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/final_confirmatory/semantic_intervention/runs'
PRIOR = ROOT / 'experiments/semantic_pairing_permutation/runs'
DATA = Path('/root/autodl-tmp/dataset1/dataset')
PYTHON = '/root/venvs/fair-v21/bin/python'
TASKS = {
    'Jarvis-Bandgap_MBJ': 'jarvis/mbj_bandgap',
    'Jarvis-Bandgap_OPT': 'jarvis/optb88vdw_bandgap',
    'Jarvis-FormationEnergy': 'jarvis/formation_energy_peratom',
    'Jarvis-BulkModulusKv': 'jarvis/bulk_modulus_kv',
    'MP-Bandgap': 'mp_2018/MP-bandgap',
    'MP-BulkModuli': 'mp_2018_small/bulk',
}
FOLLOWUP = ['Jarvis-Bandgap_MBJ', 'Jarvis-FormationEnergy', 'Jarvis-BulkModulusKv', 'MP-Bandgap']
RUNS = [(42, task) for task in TASKS] + [(seed, task) for seed in (7, 1234) for task in FOLLOWUP]


def stamp(message):
    line = f'[{datetime.now():%F %T}] {message}'
    print(line, flush=True)
    with (EXP / 'pipeline.log').open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def status(stage, **fields):
    data = dict(stage=stage, updated_at=datetime.now().isoformat(timespec='seconds'),
                total_runs=14, total_group_results=28, **fields)
    path = EXP / 'queue_status.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    lock = (EXP / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight = json.loads((EXP / 'preflight.json').read_text())
    if preflight['status'] != 'PASS' or preflight['num_runs'] != 14:
        raise RuntimeError('Label-free preflight must pass before any inference')
    for name, expected in preflight['output_hashes'].items():
        if hashlib.sha256((EXP / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Preflight output changed: {name}')
    for source in preflight['source_files']:
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise RuntimeError(f'Preflight input changed: {source["path"]}')
    protocol = {name: hashlib.sha256((EXP / name).read_bytes()).hexdigest()
                for name in ('preflight.py', 'evaluate.py', 'protocol.md', 'pipeline.py')}
    (EXP / 'protocol_checksums.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    completed = 0
    try:
        for seed, task in RUNS:
            out = EXP / 'runs' / f'seed_{seed}' / task
            if (out / 'summary.json').exists():
                saved = json.loads((out / 'summary.json').read_text())
                valid = saved['task'] == task and saved['seed'] == seed and len(saved['groups']) == 2
                for group_type in ('reduced_composition', 'element_set'):
                    subdir = out / group_type
                    matches = [g for g in saved.get('groups', []) if g.get('group_type') == group_type]
                    if len(matches) != 1 or not (subdir / 'summary.json').is_file():
                        valid = False
                        continue
                    child = json.loads((subdir / 'summary.json').read_text())
                    valid &= child == matches[0] and child.get('status') in ('complete', 'ineligible')
                    if child.get('status') == 'complete':
                        for name in ('predictions.npz', 'sample_metrics.csv', 'permutation_metrics.csv'):
                            path = subdir / name
                            valid &= path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == child.get('output_sha256', {}).get(name)
                if valid:
                    completed += 1
                    stamp(f'reuse seed={seed} task={task}')
                    continue
                (out / 'summary.json').rename(out / f'incomplete_summary_{datetime.now():%Y%m%dT%H%M%S}.json')
            free = shutil.disk_usage(ROOT).free
            if free < 512 * 1024 * 1024:
                raise RuntimeError(f'Disk guard stopped: free bytes={free}')
            source = SOURCE / f'seed_{seed}' / 'gpsr' / task
            checkpoint = source / 'best_valid.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            out.mkdir(parents=True, exist_ok=True)
            log_path = out / 'run.log'
            if log_path.exists():
                log_path.rename(out / f'failed_run_{datetime.now():%Y%m%dT%H%M%S}.log')
            status('running', completed_runs=completed, running_task=task, running_seed=seed,
                   disk_free_bytes=free)
            stamp(f'start seed={seed} task={task} groupings=2 draws_each=20')
            cmd = [PYTHON, str(EXP / 'evaluate.py'), '--task', task, '--seed', str(seed),
                   '--data-root', str(DATA), '--data-rel', TASKS[task],
                   '--checkpoint', str(checkpoint), '--source-run', str(source),
                   '--prior-run', str(PRIOR / f'seed_{seed}' / task),
                   '--groups', str(EXP / 'sample_groups.csv'), '--out', str(out)]
            with log_path.open('w', encoding='utf-8') as handle:
                code = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
            if code:
                raise RuntimeError(f'Failed seed={seed} task={task} exit={code}; inspect {log_path}')
            completed += 1
            stamp(f'done seed={seed} task={task}')
        status('finalizing', completed_runs=completed)
        subprocess.run([PYTHON, str(EXP / 'finalize.py')], cwd=ROOT, check=True)
        status('complete', completed_runs=completed)
        stamp('pipeline complete; awaiting local sync and release verification')
    except Exception as error:
        status('failed', completed_runs=completed, error=str(error))
        stamp(f'failure: {error}')
        raise


if __name__ == '__main__':
    main()
