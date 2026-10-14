"""Run both final confirmatory experiment groups with frozen protocols."""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/final_confirmatory"
SEM = EXP / "semantic_intervention"
CART = EXP / "cartnet_targeted_multiseed"
PY = "/root/venvs/fair-v21/bin/python"
DATA = Path("/root/autodl-tmp/dataset1/dataset")
TASKS = {
    "Jarvis-Bandgap_MBJ": "jarvis/mbj_bandgap",
    "Jarvis-Bandgap_OPT": "jarvis/optb88vdw_bandgap",
    "Jarvis-FormationEnergy": "jarvis/formation_energy_peratom",
    "Jarvis-BulkModulusKv": "jarvis/bulk_modulus_kv",
    "MP-Bandgap": "mp_2018/MP-bandgap",
    "MP-BulkModuli": "mp_2018_small/bulk",
}
METHODS = {
    "smm": ("smm", "SemanticMessageModulation"),
    "unbounded": ("gpsrnodegate-early", "GPSRNodeGateEarly"),
    "gpsr": ("gpsrnodegate-early-bounded", "GPSRNodeGateEarlyBoundedTau010"),
}
FAVORABLE = {"CONTROL_ADVANTAGE_SUPPORTED", "ACCURACY_CONTROL_TRADEOFF_SUPPORTED",
             "BOUNDED_INTERVENTION_ADVANTAGE_SUPPORTED"}


def stamp(msg):
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    (EXP / "pipeline.log").open("a", encoding="utf-8").write(line + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_status(stage, **extra):
    payload = {"stage": stage, "updated_at": datetime.now().isoformat(timespec="seconds"), **extra}
    (EXP / "queue_status.json").write_text(json.dumps(payload, indent=2))


def command(task, rel, method, seed):
    cli, _ = METHODS[method]
    cmd = [PY, "run.py", "--task", "train_and_test", "--target", task,
           "--model", cli, "--use_cif_csv", "--cif_dir", str(DATA/rel/"cif"),
           "--text_emb_path", str(DATA/rel/"MatBert_embeddings.pt"),
           "--num_epoch", "60", "--batch_size", "32", "--seed", str(seed),
           "--lr", "0.001", "--max_lr", "0.001", "--weight_decay", "0.0001",
           "--inject_position", "early"]
    if method == "gpsr": cmd += ["--max_residual_ratio", "0.10"]
    return cmd


def complete_sft(out):
    return (out / "best_valid.pth").is_file() and (out / "on_off_summary.json").is_file()


def collect_sft(out, log_path, task, prefix, started):
    run_dir = None
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Run output directory:" in line:
            run_dir = Path(line.split("Run output directory:", 1)[1].strip())
    if run_dir and not run_dir.is_absolute(): run_dir = ROOT / run_dir
    if not run_dir or not run_dir.is_dir(): raise RuntimeError("run directory not found")
    for f in run_dir.iterdir():
        if f.is_file(): shutil.copy2(f, out / ("run_detail.log" if f.name == "run.log" else f.name))
    shutil.rmtree(run_dir)
    parameter = ROOT / "parameter" / f"{prefix}_{task}_para.pth"
    if not parameter.is_file(): raise FileNotFoundError(parameter)
    shutil.copy2(parameter, out / "best_valid.pth")
    for p in (ROOT / "parameter").glob(f"{prefix}_{task}_*.pth"):
        if p.stat().st_mtime >= started - 5: p.unlink()
    if parameter.exists() and parameter.stat().st_mtime >= started - 5: parameter.unlink()


def run_sft_group(seed, tasks, methods, phase):
    jobs=[]
    for task in tasks:
        for method in methods:
            out=SEM/"runs"/f"seed_{seed}"/method/task
            if complete_sft(out): continue
            out.mkdir(parents=True, exist_ok=True)
            jobs.append((task,method,out,command(task,TASKS[task],method,seed)))
    # Concurrency 3 mirrors the completed final-10-task protocol and fits the 4090.
    pending=list(jobs); running={}; failed=[]; done=[]
    while pending or running:
        while pending and len(running)<3:
            task,method,out,cmd=pending.pop(0); log=out/"run.log"; h=log.open("w")
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="3",MKL_NUM_THREADS="3",
                     PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            st=time.time(); proc=subprocess.Popen(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env=env)
            key=f"{phase}/{seed}/{method}/{task}"; running[key]=(proc,h,task,method,out,log,st)
            stamp(f"start {key} pid={proc.pid}")
        write_status(phase,running=list(running),pending=[f"{m}/{t}" for t,m,_,_ in pending],done=done,failed=failed)
        time.sleep(10)
        for key,item in list(running.items()):
            proc,h,task,method,out,log,st=item; code=proc.poll()
            if code is None: continue
            h.close()
            if code==0:
                try:
                    collect_sft(out,log,task,METHODS[method][1],st)
                    rel=TASKS[task]
                    pcmd=[PY,"experiments/final_confirmatory/semantic_intervention/posthoc_off.py",
                          "--target",task,"--method",method,"--checkpoint",str(out/"best_valid.pth"),
                          "--run-dir",str(out),"--cif-dir",str(DATA/rel/"cif"),
                          "--text-emb-path",str(DATA/rel/"MatBert_embeddings.pt"),"--seed",str(seed)]
                    subprocess.run(pcmd,cwd=ROOT,check=True,stdout=(out/"off_pass.log").open("w"),stderr=subprocess.STDOUT)
                    done.append(key); stamp(f"done {key}")
                except Exception as e:
                    failed.append({"key":key,"error":repr(e)}); stamp(f"COLLECT/OFF FAIL {key}: {e!r}")
            else:
                failed.append({"key":key,"exit":code}); stamp(f"FAIL {key} exit={code}")
            del running[key]
    if failed: raise RuntimeError(f"{phase} failures: {failed}")


def graph_seed42():
    p=ROOT/"experiments/final_10task/summary/benchmark_seed42.csv"
    df=pd.read_csv(p)
    return df[df.model.eq("graph")].set_index("task")["test_mae"].to_dict()


def aggregate_a1():
    graph=graph_seed42(); result=[]; inter=[]; all_diag=[]
    for task in TASKS:
        result.append({"task":task,"method":"graph","seed":42,"test_mae":graph[task],
                       "relative_gain_vs_graph_pct":0.0,"source":"reused final_10task independent Graph"})
        for method in METHODS:
            rd=SEM/"runs/seed_42"/method/task
            s=json.loads((rd/"on_off_summary.json").read_text())
            result.append({"task":task,"method":method,"seed":42,"test_mae":s["mae_on"],
                           "relative_gain_vs_graph_pct":100*(graph[task]-s["mae_on"])/graph[task],
                           "source":"new matched confirmatory run"})
            inter.append(s)
            all_diag.append(pd.read_csv(rd/"semantic_on_off_diagnostics.csv"))
    rdf=pd.DataFrame(result); idf=pd.DataFrame(inter)
    rdf.to_csv(SEM/"task_results_seed42.csv",index=False)
    idf.to_csv(SEM/"intervention_metrics_seed42.csv",index=False)
    pd.concat(all_diag,ignore_index=True).to_csv(SEM/"semantic_on_off_diagnostics.csv",index=False)
    macro=[]
    for method in ("graph",*METHODS):
        x=rdf[rdf.method.eq(method)].set_index("task")
        ratios=[x.loc[t,"test_mae"]/graph[t] for t in TASKS]
        ii=idf[idf.method.eq(method)] if method!="graph" else None
        macro.append({"method":method,"macro_gnmae":np.mean(ratios),
                      "mean_semantic_rho":0.0 if method=="graph" else ii.mean_rho.mean(),
                      "max_observed_rho":0.0 if method=="graph" else ii.max_rho.max(),
                      "improved_tasks":sum(r<1 for r in ratios),"num_tasks":len(ratios)})
    mdf=pd.DataFrame(macro); mdf.to_csv(SEM/"macro_summary.csv",index=False)
    # Operational, frozen interpretation thresholds: competitive = within 0.02 GN-MAE;
    # materially smaller displacement = at least 20% lower mean rho.
    mm=mdf.set_index("method"); best=min(mm.loc[m,"macro_gnmae"] for m in METHODS)
    gpsr_comp=mm.loc["gpsr","macro_gnmae"] <= best+.02
    gpsr_small=all(mm.loc["gpsr","mean_semantic_rho"] <= .8*mm.loc[m,"mean_semantic_rho"] for m in ("smm","unbounded"))
    if mm.loc["smm","macro_gnmae"] < mm.loc["gpsr","macro_gnmae"] and mm.loc["smm","mean_semantic_rho"] < mm.loc["gpsr","mean_semantic_rho"]:
        verdict="GPSR_CONTROL_ADVANTAGE_NOT_SUPPORTED"
    elif gpsr_comp and gpsr_small and mm.loc["gpsr","macro_gnmae"] <= best+1e-12:
        verdict="CONTROL_ADVANTAGE_SUPPORTED"
    elif gpsr_comp and mm.loc["unbounded","macro_gnmae"] < mm.loc["gpsr","macro_gnmae"] and mm.loc["gpsr","mean_semantic_rho"] <= .8*mm.loc["unbounded","mean_semantic_rho"]:
        verdict="ACCURACY_CONTROL_TRADEOFF_SUPPORTED"
    elif gpsr_comp and abs(mm.loc["smm","macro_gnmae"]-mm.loc["gpsr","macro_gnmae"]) <= .02 and mm.loc["gpsr","mean_semantic_rho"] <= .8*mm.loc["smm","mean_semantic_rho"]:
        verdict="BOUNDED_INTERVENTION_ADVANTAGE_SUPPORTED"
    elif gpsr_comp and gpsr_small:
        verdict="CONTROL_ADVANTAGE_SUPPORTED"
    else: verdict="NO_CLEAR_INTERVENTION_ADVANTAGE"
    # Static scientific figure.
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.4,4.8))
    for _,r in mdf.iterrows(): ax.scatter(r.mean_semantic_rho,r.macro_gnmae,s=70,label=r.method); ax.annotate(r.method,(r.mean_semantic_rho,r.macro_gnmae),xytext=(5,5),textcoords="offset points")
    ax.set_xlabel("Mean semantic displacement (rho)"); ax.set_ylabel("Macro GN-MAE (lower is better)"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(SEM/"accuracy_control_tradeoff.pdf"); plt.close(fig)
    (SEM/"phase_a1_verdict.json").write_text(json.dumps({"verdict":verdict,"thresholds":{"competitive_gnmae_margin":.02,"material_rho_reduction":.20}},indent=2))
    return verdict


def count_parameters():
    code='''from types import SimpleNamespace\nimport run as r\nfrom train_and_test import build_model\nrows=[]\nbase=dict(task="train_and_test",target="Jarvis-Bandgap_MBJ",use_cif_csv=True,cif_dir="%s",text_emb_path="%s",num_epoch=60,batch_size=32,seed=42,lr=.001,max_lr=.001,weight_decay=.0001,alpha_init=None,alpha_max=.2,lambda_delta=0.,lambda_res=0.,rho_target=.1,sample_gate_bias=3.,inject_position="early",inject_after=0,text_shuffle_seed=-1,share_text_cache=False)\nfor name,model,tau in [("graph","graphonly",float("inf")),("smm","smm",float("inf")),("unbounded","gpsrnodegate-early",float("inf")),("gpsr","gpsrnodegate-early-bounded",.1)]:\n a=SimpleNamespace(**base,model=model,max_residual_ratio=tau); r.configure(a); m=build_model(); rows.append({"method":name,"trainable_parameters":sum(p.numel() for p in m.parameters() if p.requires_grad),"total_parameters":sum(p.numel() for p in m.parameters())})\nimport json; print(json.dumps(rows))\n''' % (DATA/TASKS["Jarvis-Bandgap_MBJ"]/"cif",DATA/TASKS["Jarvis-Bandgap_MBJ"]/"MatBert_embeddings.pt")
    out=subprocess.check_output([PY,"-c",code],cwd=ROOT,text=True)
    pd.DataFrame(json.loads(out.strip().splitlines()[-1])).to_csv(SEM/"parameter_counts.csv",index=False)


def run_cartnet():
    tasks=["Jarvis-Bandgap_MBJ","Jarvis-BulkModulusKv","MP-BulkModuli"]
    for seed in (7,1234):
        for task in tasks:
            for mode in ("graph","gpsr"):
                out=CART/"runs"/f"seed_{seed}"/task/mode
                if (out/"metrics.json").is_file(): continue
                out.mkdir(parents=True,exist_ok=True); log=out/"run.log"
                write_status("cartnet_targeted",running=f"{seed}/{task}/{mode}")
                stamp(f"start CartNet seed={seed} task={task} mode={mode}")
                cmd=[PY,"experiments/cartnet_gpsr/train_matched.py","--task",task,"--mode",mode,
                     "--seed",str(seed),"--epochs","500","--batch-size","64","--out",str(out)]
                with log.open("w") as h:
                    code=subprocess.call(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,
                                         env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
                if code: raise RuntimeError(f"CartNet failed {seed}/{task}/{mode} exit={code}")
                stamp(f"done CartNet seed={seed} task={task} mode={mode}")


def aggregate_cartnet():
    name_map={"Jarvis-Bandgap_MBJ":"j_mbj_results.csv","Jarvis-BulkModulusKv":"j_bulk_results.csv","MP-BulkModuli":"mp_bulk_results.csv"}
    summaries=[]
    for task,fname in name_map.items():
        rows=[]
        for seed in (42,7,1234):
            base=ROOT/"experiments/cartnet_gpsr/phase1_matched_10task"/task if seed==42 else CART/"runs"/f"seed_{seed}"/task
            g=json.loads((base/"graph/metrics.json").read_text()); s=json.loads((base/"gpsr/metrics.json").read_text())
            gain=100*(g["test_mae"]-s["test_mae"])/g["test_mae"]
            rows.append({"task":task,"seed":seed,"cartnet_mae":g["test_mae"],"gpsr_mae":s["test_mae"],"paired_gain_pct":gain,"improved":gain>0})
        df=pd.DataFrame(rows); df.to_csv(CART/fname,index=False)
        gains=df.paired_gain_pct.to_numpy(); neg=int((gains<0).sum()); pos=int((gains>0).sum())
        if task=="MP-BulkModuli":
            verdict="STABLE_NEGATIVE" if neg>=2 and gains.mean()<0 else ("SEED_SENSITIVE_NEGATIVE" if pos>=2 and gains.mean()>=-1 else "HIGH_VARIANCE")
        elif task=="Jarvis-Bandgap_MBJ":
            verdict="STRONG_STABLE_POSITIVE" if pos==3 and gains.mean()>=3 else ("STABLE_POSITIVE" if pos>=2 and gains.mean()>0 else "NOT_STABLE_POSITIVE")
        else: verdict="STABLE_POSITIVE" if pos>=2 and gains.mean()>0 else "NOT_STABLE_POSITIVE"
        summaries.append({"task":task,"cartnet_mean_mae":df.cartnet_mae.mean(),"cartnet_sample_std":df.cartnet_mae.std(ddof=1),"gpsr_mean_mae":df.gpsr_mae.mean(),"gpsr_sample_std":df.gpsr_mae.std(ddof=1),"mean_paired_gain_pct":gains.mean(),"paired_gain_sample_std":gains.std(ddof=1),"improved_runs":pos,"num_seeds":3,"verdict":verdict})
    pd.DataFrame(summaries).to_csv(CART/"paired_gain_summary.csv",index=False)


def write_reports(verdict):
    a=pd.read_csv(SEM/"macro_summary.csv").set_index("method"); im=pd.read_csv(SEM/"intervention_metrics_seed42.csv")
    b=pd.read_csv(CART/"paired_gain_summary.csv").set_index("task")
    hard=im[im.method.eq("gpsr")].max_rho.max()
    smm_gain=(1-a.loc["smm","macro_gnmae"])*100; unb_vs=(a.loc["gpsr","macro_gnmae"]-a.loc["unbounded","macro_gnmae"])*100
    sem_report=f'''# Matched Semantic Intervention Report\n\nVerdict: **{verdict}**\n\nSMM macro gain versus Graph: {smm_gain:.3f}%. Unbounded minus GPSR GN-MAE advantage (positive favors Unbounded): {unb_vs:.3f} percentage points.\n\nMean rho: SMM {a.loc['smm','mean_semantic_rho']:.6f}, Unbounded {a.loc['unbounded','mean_semantic_rho']:.6f}, GPSR {a.loc['gpsr','mean_semantic_rho']:.6f}. Maximum GPSR rho: {hard:.8f}.\n\nSMM is a paper/code-derived SFTGNN adaptation of official SAGE-CartNet commit fc298f82efd6141383b78070bce6ccc053f9bd0c; it is not an official SAGE-Net run. Graph metrics reuse the independent final-10-task seed-42 baseline.\n'''
    (SEM/"semantic_intervention_report.md").write_text(sem_report)
    cart_report="# Targeted CartNet Multi-seed Robustness Report\n\nOriginal ten-task seed-42 conclusion remains unchanged: 6/10 improved, Macro GN-MAE 1.0050, no aggregate gain.\n\n"+b.to_markdown()+"\n\nThis is an exploratory targeted three-task subset; no ten-task three-seed macro is computed.\n"
    (CART/"targeted_robustness_report.md").write_text(cart_report)
    overall=f'''# Final Confirmatory Experiment Report\n\n## Experiment A\n\n1. SMM vs Graph: macro gain {smm_gain:.3f}%.\n2. Unbounded vs GPSR: GN-MAE difference {unb_vs:.3f} percentage points (positive favors Unbounded).\n3. Mean representation displacement: SMM {a.loc['smm','mean_semantic_rho']:.6f}; Unbounded {a.loc['unbounded','mean_semantic_rho']:.6f}.\n4. GPSR maximum per-sample rho is {hard:.8f}; hard-bound check {'PASS' if hard<=.1001 else 'FAIL'}.\n5. Accuracy/control verdict: **{verdict}**.\n6. The supported formal case is **{verdict}**.\n\n## Experiment B\n\n7. MP-Bulk: {b.loc['MP-BulkModuli','verdict']}; mean paired gain {b.loc['MP-BulkModuli','mean_paired_gain_pct']:.3f}% ({int(b.loc['MP-BulkModuli','improved_runs'])}/3 improved).\n8. J-MBJ: {b.loc['Jarvis-Bandgap_MBJ','verdict']}; mean paired gain {b.loc['Jarvis-Bandgap_MBJ','mean_paired_gain_pct']:.3f}% ({int(b.loc['Jarvis-Bandgap_MBJ','improved_runs'])}/3 improved).\n9. J-Bulk: {b.loc['Jarvis-BulkModulusKv','verdict']}; mean paired gain {b.loc['Jarvis-BulkModulusKv','mean_paired_gain_pct']:.3f}% ({int(b.loc['Jarvis-BulkModulusKv','improved_runs'])}/3 improved).\n10. The CartNet boundary claim should be interpreted through these targeted verdicts while retaining the original ten-task no-aggregate-gain outcome.\n11. Claims that may be added: the measured accuracy/intervention trade-off and only the task-specific seed-stability statements supported above.\n12. Claims to weaken/remove: any claim of official SAGE-Net comparison, causal attribution of output sensitivity, ten-task multi-seed CartNet macro performance, or universal bulk-modulus failure.\n\nNo manuscript file was modified.\n'''
    (EXP/"final_confirmatory_experiment_report.md").write_text(overall)


def preflight():
    EXP.mkdir(parents=True,exist_ok=True); SEM.mkdir(parents=True,exist_ok=True); CART.mkdir(parents=True,exist_ok=True)
    subprocess.run([PY,"experiments/final_confirmatory/semantic_intervention/patch_sft.py"],cwd=ROOT,check=True)
    info={"created_at":datetime.now().isoformat(),"sftgnn":{"epochs":60,"batch_size":32,"seed":42,"optimizer":"AdamW","lr":1e-3,"max_lr":1e-3,"onecycle_pct_start":.3,"checkpoint":"minimum validation MAE"},"smm":{"label":"SMM paper-derived adaptation (not official SAGE-Net)","official_repository":"https://github.com/zhanggh1999/SAGE-Net","official_commit":"fc298f82efd6141383b78070bce6ccc053f9bd0c","module":"TextConditionedMiddleModule","projector":"Linear(768,128)-GELU-Linear(128,128)","injection":"after first valid SFTGNN block","hard_norm_constraint":False},"gpsr":{"tau":.10,"unchanged":True},"reuse":{"graph_seed42":"final_10task summary, fully matched independent Graph","semantic":"retrain because final_10task checkpoints were intentionally cleaned and ON/OFF requires same checkpoint"},"decision_thresholds":{"competitive_gnmae_margin":.02,"material_rho_reduction":.20},"source_hashes":{p:sha(ROOT/p) for p in ["run.py","train_and_test.py","model/semantic_residual.py","experiments/final_confirmatory/semantic_intervention/smm_model.py","experiments/cartnet_gpsr/train_matched.py"]}}
    (EXP/"preflight.json").write_text(json.dumps(info,indent=2))
    count_parameters()


def main():
    preflight(); write_status("semantic_a1")
    run_sft_group(42,list(TASKS),list(METHODS),"semantic_a1")
    verdict=aggregate_a1(); stamp(f"A1 verdict {verdict}")
    write_status("cartnet_targeted")
    run_cartnet(); aggregate_cartnet()
    if verdict in FAVORABLE:
        run_sft_group(7,["Jarvis-Bandgap_MBJ","Jarvis-FormationEnergy","Jarvis-BulkModulusKv","MP-Bandgap"],list(METHODS),"semantic_a2_seed7")
        run_sft_group(1234,["Jarvis-Bandgap_MBJ","Jarvis-FormationEnergy","Jarvis-BulkModulusKv","MP-Bandgap"],list(METHODS),"semantic_a2_seed1234")
        # Compact multi-seed tables.
        rows=[]
        for seed in (42,7,1234):
            for task in ("Jarvis-Bandgap_MBJ","Jarvis-FormationEnergy","Jarvis-BulkModulusKv","MP-Bandgap"):
                for method in METHODS:
                    s=json.loads((SEM/"runs"/f"seed_{seed}"/method/task/"on_off_summary.json").read_text()); rows.append(s)
        df=pd.DataFrame(rows); df.to_csv(SEM/"multiseed_task_summary.csv",index=False)
        df.groupby("method").agg(mean_mae=("mae_on","mean"),sample_std_mae=("mae_on","std"),mean_rho=("mean_rho","mean"),max_rho=("max_rho","max")).reset_index().to_csv(SEM/"multiseed_macro_summary.csv",index=False)
    write_reports(verdict); write_status("complete",verdict=verdict)
    (EXP/"pipeline.exit").write_text("0\n"); stamp("all confirmatory experiments complete")


if __name__=="__main__":
    try: main()
    except Exception as e:
        write_status("failed",error=repr(e)); (EXP/"pipeline.exit").write_text("1\n"); stamp(f"PIPELINE FAILED: {e!r}"); raise
