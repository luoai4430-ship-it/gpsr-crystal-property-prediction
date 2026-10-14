"""Build a task's ALIGNN graphs once, then train requested seeds and models."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from train_alignn_transfer import TASKS, build_graphs, load_records, train_one


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True,choices=TASKS)
    parser.add_argument('--seeds',nargs='+',type=int,required=True)
    parser.add_argument('--models',nargs='+',default=['graph','hard'],choices=['graph','early','hard'])
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    records=load_records(args.task)
    root=Path(args.output_root)
    build_graphs(args.task,records,root/f'{args.task}_graph_build.log')
    for seed in args.seeds:
        run_args=SimpleNamespace(seed=seed,epochs=args.epochs,batch_size=args.batch_size)
        task_dir=root/f'seed_{seed}'/args.task
        for mode in args.models:
            train_one(args.task,mode,records,run_args,task_dir)


if __name__=='__main__':
    main()
