"""Idempotently register SMM and add evaluation-only sample diagnostics."""
from pathlib import Path

root = Path(__file__).resolve().parents[3]
trainer = root / "train_and_test.py"
run = root / "run.py"
semantic = root / "model" / "semantic_residual.py"

t = trainer.read_text()
t = t.replace("\n+", "\n")
needle = "from model.semantic_residual import GPSRSFTGNN, resolve_injection_block, as_text_matrix\n"
addition = needle + "from experiments.final_confirmatory.semantic_intervention.smm_model import SMMSFTGNN\n"
if "import SMMSFTGNN" not in t:
    t = t.replace(needle, addition)
case_needle = '        case name if name.startswith("GPSR"):\n'
case_text = '''        case "SemanticMessageModulation":
            num_layers = model_config.num_layers
            inject_after = resolve_injection_block("early", num_layers)
            model_config.gpsr_injection_after_block = inject_after
            backbone = SFTGNN(num_layers=num_layers, in_dim=data_config.getAtomDim())
            model = SMMSFTGNN(
                sftgnn=backbone, node_dim=getattr(model_config, 'node_feature', 128),
                text_dim=getattr(model_config, 'text_emb_dim', 768),
                injection_after_block=inject_after, name=model_config.model_name,
            )
        case name if name.startswith("GPSR"):
'''
if 'case "SemanticMessageModulation"' not in t:
    t = t.replace(case_needle, case_text)
trainer.write_text(t)

r = run.read_text()
r = r.replace("\n+", "\n")
if '"smm",' not in r:
    r = r.replace('"graphonly", "concat", "textresidual",\n',
                  '"graphonly", "concat", "textresidual", "smm",\n')
smm_cfg = '''    if args.model == "smm":
        model_config.model_name = "SemanticMessageModulation"
        model_config.gpsr_enabled = True
        model_config.gpsr_use_node_gate = False
        model_config.gpsr_alpha_init = 1.0
        model_config.gpsr_injection_position = "early"
        model_config.gpsr_injection_after_block = 0
        model_config.gpsr_max_residual_ratio = float("inf")
        model_config.gpsr_lambda_res = 0.0
        model_config.gpsr_rho_target = 0.10
        model_config.message = "SMM paper-derived SFTGNN adaptation; no explicit norm constraint"
        return

'''
marker = "    # --- GPSR: atom-level semantic residual injection (graph-preserving) ---\n"
if 'args.model == "smm"' not in r:
    r = r.replace(marker, smm_cfg + marker)
run.write_text(r)

s = semantic.read_text()
s = s.replace("\n+", "\n")
stats_marker = '            "graph_text_cosine": graph_text_cosine.detach(),\n'
stats_add = '''            "graph_text_cosine": graph_text_cosine.detach(),
            "cos_h_off_on": (scatter((node_feat * node_out).sum(-1), batch, dim=0, dim_size=num_graphs, reduce="sum") /
                             (h_norm * scatter(node_out.square().sum(-1), batch, dim=0, dim_size=num_graphs, reduce="sum").sqrt() + eps)).detach(),
            "cos_delta_h": (scatter((delta * node_feat).sum(-1), batch, dim=0, dim_size=num_graphs, reduce="sum") /
                            (d_final_norm * h_norm + eps)).detach(),
'''
if '"cos_h_off_on"' not in s:
    s = s.replace(stats_marker, stats_add)
semantic.write_text(s)

print("confirmatory SMM registration and diagnostics patch applied")
