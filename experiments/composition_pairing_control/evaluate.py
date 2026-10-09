"""Exploratory composition-conditioned text pairing at a frozen SFTGNN checkpoint.

No training, checkpoint selection, or changes to the trained GPSR injector.
Only donor assignment changes. Correct/OFF/global controls come from the
completed pairing study, after a fresh row-wise checkpoint replay audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.semantic_pairing_permutation.evaluate import (
    collect_text_and_ids, configure, derangement, load_state, predict,
)
from data.dataset import get_dataloader
from train_and_test import build_model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assignments(keys, rng):
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    eligible = np.array(sorted(i for values in groups.values() if len(values) > 1
                               for i in values), dtype=np.int64)
    mapping = np.arange(len(keys), dtype=np.int64)
    for values in groups.values():
        if len(values) > 1:
            idx = np.asarray(values)
            mapping[idx] = idx[derangement(len(idx), rng)]
    donors = mapping[eligible]
    if len(eligible):
        assert np.all(donors != eligible)
        assert np.array_equal(np.sort(donors), eligible)
        assert all(keys[i] == keys[j] for i, j in zip(eligible, donors))
    return eligible, donors


def self_test():
    rng = np.random.default_rng(20261009)
    keys = ['A', 'A', 'B', 'C', 'C', 'C']
    for _ in range(100):
        eligible, donors = assignments(keys, rng)
        assert eligible.tolist() == [0, 1, 3, 4, 5]
        assert donors[0] == 1 and donors[1] == 0
    assert len(assignments(['A', 'B'], rng)[0]) == 0
    print('conditional assignment self-test passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--data-rel')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--prior-run', type=Path)
    parser.add_argument('--groups', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(target=args.task, seed=args.seed, batch_size=32,
                          cif_dir=str(args.data_root / args.data_rel / 'cif'),
                          text_emb_path=str(args.data_root / args.data_rel / 'MatBert_embeddings.pt'))
    configure(cfg)
    model = build_model().cuda().eval()
    model.load_state_dict(load_state(args.checkpoint))
    for param in model.parameters():
        param.requires_grad_(False)
    model.injector.record_stats = False
    rho_tracker = {'max': 0.}
    def check_bound(module, inputs, outputs):
        value = float(module.last_rho_final.max().item())
        rho_tracker['max'] = max(rho_tracker['max'], value)
        if not math.isfinite(value) or value > .1001:
            raise RuntimeError(f'Unchanged GPSR bound violated: rho_final={value}')
    model.injector.register_forward_hook(check_bound)
    _, _, loader, mean, std = get_dataloader()
    all_text, ids = collect_text_and_ids(loader)
    if not torch.isfinite(all_text).all():
        raise RuntimeError('Non-finite text embeddings')
    if len(ids) != len(set(ids)):
        raise RuntimeError('Duplicate sample IDs')
    with args.groups.open(encoding='utf-8', newline='') as handle:
        group_rows = [row for row in csv.DictReader(handle) if row['task'] == args.task]
    group_rows.sort(key=lambda row: int(row['sample_index']))
    if ids != [row['sample_id'] for row in group_rows]:
        raise RuntimeError('Preflight/dataloader sample ID order differs')
    prior_csv = pd.read_csv(args.prior_run / 'sample_metrics.csv', dtype={'sample_id': str})
    if ids != prior_csv.sample_id.tolist():
        raise RuntimeError('Prior prediction ID order differs')
    with np.load(args.prior_run / 'predictions.npz', allow_pickle=False) as archive:
        prior = {key: archive[key].copy() for key in ('correct', 'off', 'shuffled', 'truth')}
    correct, truth = predict(model, loader, mean, std)
    if not np.all(np.isfinite(correct)) or not np.all(np.isfinite(truth)) or not math.isfinite(float(std)) or float(std) <= 0:
        raise RuntimeError('Non-finite original predictions/truth or invalid target standard deviation')
    formal_files = list(args.source_run.glob('*_test_pred_true.csv'))
    if len(formal_files) != 1:
        raise RuntimeError('Expected exactly one formal prediction CSV')
    formal = pd.read_csv(formal_files[0])
    native_diff = np.max(np.abs(correct - formal.y_pred.to_numpy())) / max(abs(float(std)), 1e-12)
    prior_native_diff = np.max(np.abs(correct - prior['correct'])) / max(abs(float(std)), 1e-12)
    true_diff = np.max(np.abs(truth - formal.y_true.to_numpy()))
    if not all(math.isfinite(float(value)) for value in (native_diff, prior_native_diff, true_diff)) or native_diff > 1e-5 or prior_native_diff > 1e-5 or true_diff > 1e-5:
        raise RuntimeError(f'Checkpoint replay failed: native={native_diff}, prior={prior_native_diff}, truth={true_diff}')
    if not np.allclose(truth, prior['truth'], rtol=0, atol=1e-7):
        raise RuntimeError('Prior truth differs')
    input_hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256(path)
                    for path in (args.checkpoint, formal_files[0], args.prior_run / 'predictions.npz', args.groups, Path(__file__))}
    text_np = all_text.numpy().astype(np.float32, copy=False)
    embed_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in text_np]
    pd.DataFrame({'sample_index': np.arange(len(ids)), 'sample_id': ids,
                  'embedding_sha256': embed_hashes}).to_csv(args.out / 'embedding_hashes.csv', index=False)
    audits = {'task': args.task, 'seed': args.seed, 'trained': False,
              'checkpoint_selection': False, 'target_std': float(std),
              'formal_reproduction_max_abs_pred_diff_physical': float(native_diff * abs(float(std))),
              'formal_reproduction_max_abs_pred_diff_native': float(native_diff),
              'prior_reproduction_max_abs_pred_diff_native': float(prior_native_diff),
              'formal_reproduction_max_abs_true_diff': float(true_diff),
              'input_sha256': input_hashes, 'groups': []}
    for group_number, group_type in enumerate(('reduced_composition', 'element_set')):
        output = args.out / group_type
        output.mkdir(parents=True, exist_ok=True)
        keys = [row[group_type] for row in group_rows]
        rng = np.random.default_rng(20261009 + args.seed + group_number * 100000)
        eligible, _ = assignments(keys, rng)
        saved_path = output / 'summary.json'
        if saved_path.exists():
            saved = json.loads(saved_path.read_text())
            required = ('predictions.npz', 'sample_metrics.csv', 'permutation_metrics.csv')
            reusable = (saved.get('task') == args.task and saved.get('seed') == args.seed
                        and saved.get('group_type') == group_type
                        and saved.get('input_sha256') == input_hashes)
            files_valid = all((output / name).is_file() and
                              sha256(output / name) == saved.get('output_sha256', {}).get(name)
                              for name in required)
            if reusable and (saved.get('status') == 'ineligible' or
                    (saved.get('status') == 'complete' and files_valid)):
                audits['groups'].append(saved)
                rho_tracker['max'] = max(rho_tracker['max'], saved.get('max_rho_final_audited', 0.))
                print(f'reuse {args.task} seed={args.seed} {group_type}', flush=True)
                continue
            evidence = output / 'failed_evidence'
            evidence.mkdir(exist_ok=True)
            import shutil
            for path in output.iterdir():
                if path.is_file():
                    destination = evidence / path.name
                    counter = 1
                    while destination.exists():
                        destination = evidence / f'{path.name}.{counter}'
                        counter += 1
                    shutil.copy2(path, destination)
        if len(eligible) == 0:
            result = {'task': args.task, 'seed': args.seed, 'group_type': group_type,
                      'status': 'ineligible', 'num_samples': len(ids), 'num_eligible': 0,
                      'input_sha256': input_hashes}
            (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            audits['groups'].append(result)
            continue
        subset_loader = DataLoader(Subset(loader.dataset, eligible.tolist()), batch_size=32,
                                   shuffle=False, num_workers=0)
        subset_text, subset_ids = collect_text_and_ids(subset_loader)
        if subset_ids != [ids[i] for i in eligible] or not torch.equal(subset_text, all_text[eligible]):
            raise RuntimeError('Subset/donor text ordering mismatch')
        subset_correct, subset_truth = predict(model, subset_loader, mean, std)
        subset_native_diff = float(np.max(np.abs(subset_correct - prior['correct'][eligible])) / float(std))
        if not math.isfinite(subset_native_diff) or subset_native_diff > 1e-5 or not np.allclose(subset_truth, truth[eligible], rtol=0, atol=1e-7):
            raise RuntimeError(f'Same-path subset replay failed: native={subset_native_diff}')
        predictions, donor_rows, permutation_rows = [], [], []
        for permutation_index in range(20):
            indices, donors = assignments(keys, rng)
            assert np.array_equal(indices, eligible)
            replacement = all_text[donors]
            # The normal forward path enforces and records the unchanged tau=0.10.
            values, truths = predict(model, subset_loader, mean, std, replacement)
            if not np.all(np.isfinite(values)) or not np.all(np.isfinite(truths)):
                raise RuntimeError('Non-finite conditional predictions/truth')
            if not np.allclose(truths, truth[eligible], rtol=0, atol=1e-7):
                raise RuntimeError('Conditional prediction truth ordering differs')
            predictions.append(values)
            donor_rows.append(donors)
            own = text_np[eligible].astype(np.float64)
            assigned = text_np[donors].astype(np.float64)
            cosines = np.sum(own * assigned, axis=1) / np.maximum(
                np.linalg.norm(own, axis=1) * np.linalg.norm(assigned, axis=1), 1e-30)
            permutation_rows.append({
                'permutation_index': permutation_index,
                'mae': float(np.mean(np.abs(values - truths))),
                'fixed_point_fraction': 0.,
                'same_embedding_fraction': float(np.mean([embed_hashes[i] == embed_hashes[j]
                                                          for i, j in zip(eligible, donors)])),
                'same_description_fraction': float(np.mean([group_rows[i]['description_sha256'] ==
                                                            group_rows[j]['description_sha256']
                                                            for i, j in zip(eligible, donors)])),
                'donor_cosine_mean': float(cosines.mean()),
                'donor_cosine_q05': float(np.quantile(cosines, .05)),
                'donor_cosine_q95': float(np.quantile(cosines, .95)),
            })
            print(f'{args.task} seed={args.seed} {group_type} {permutation_index+1}/20', flush=True)
        conditional = np.asarray(predictions)
        donors = np.asarray(donor_rows)
        actual_correct = prior['correct'][eligible]
        actual_off = prior['off'][eligible]
        actual_global = prior['shuffled'][:, eligible]
        t = truth[eligible]
        error = np.abs(conditional - t[None, :])
        correct_mae = float(np.mean(np.abs(actual_correct - t)))
        off_mae = float(np.mean(np.abs(actual_off - t)))
        mean_mae = float(error.mean())
        global_mae = float(np.mean(np.abs(actual_global - t[None, :])))
        result = {
            'task': args.task, 'seed': args.seed, 'group_type': group_type, 'status': 'complete',
            'num_samples': len(ids), 'num_eligible': len(eligible),
            'coverage_fraction': float(len(eligible) / len(ids)),
            'num_permutations': 20,
            'num_distinct_assignments': len({row.tobytes() for row in donors}),
            'correct_mae': correct_mae, 'off_mae': off_mae,
            'conditional_mean_mae': mean_mae,
            'conditional_mae_sd_over_assignments': float(error.mean(axis=1).std(ddof=1)),
            'context_global_whole_test_donor_mean_mae': global_mae,
            'correct_advantage_pct_vs_conditional': 100 * (1 - correct_mae / mean_mae),
            'correct_advantage_pct_vs_global_context': 100 * (1 - correct_mae / global_mae),
            'conditional_worse_than_correct_assignments': int(np.sum(error.mean(axis=1) > correct_mae)),
            'fraction_samples_correct_better_than_conditional_mean': float(np.mean(
                np.abs(actual_correct - t) < error.mean(axis=0))),
            'same_embedding_fraction_mean': float(np.mean([row['same_embedding_fraction'] for row in permutation_rows])),
            'same_description_fraction_mean': float(np.mean([row['same_description_fraction'] for row in permutation_rows])),
            'donor_cosine_mean': float(np.mean([row['donor_cosine_mean'] for row in permutation_rows])),
            'max_rho_final_audited': rho_tracker['max'],
            'subset_reproduction_max_abs_pred_diff_native': subset_native_diff,
            'interpretation': 'Exploratory within-checkpoint control on eligible subset; no training; global donor pool differs.'}
        result['input_sha256'] = input_hashes
        pd.DataFrame(permutation_rows).to_csv(output / 'permutation_metrics.csv', index=False)
        pd.DataFrame({'sample_index': eligible, 'sample_id': subset_ids, 'group_key': [keys[i] for i in eligible],
                      'y_true': t, 'y_correct': actual_correct, 'y_off': actual_off,
                      'conditional_mean_abs_error': error.mean(axis=0),
                      'correct_abs_error': np.abs(actual_correct - t),
                      'global_context_mean_abs_error': np.abs(actual_global - t).mean(axis=0)}
                     ).to_csv(output / 'sample_metrics.csv', index=False)
        np.savez_compressed(output / 'predictions.npz', truth=t, correct=actual_correct,
                            off=actual_off, conditional=conditional, global_context=actual_global,
                            eligible_indices=eligible, donor_indices=donors)
        result['output_sha256'] = {name: sha256(output / name) for name in
                                  ('predictions.npz', 'sample_metrics.csv', 'permutation_metrics.csv')}
        (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        audits['groups'].append(result)
    audits['max_rho_final_audited'] = rho_tracker['max']
    (args.out / 'summary.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
    print(json.dumps(audits), flush=True)


if __name__ == '__main__':
    main()
