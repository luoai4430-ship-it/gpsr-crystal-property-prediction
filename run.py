import argparse
import os

from data.config import Config as data_config
from data.config import JarvisTarget, MPTarget
from model.config import Config as model_config
from train_and_test import TrainConfigManager, test_with_pretrained_model, train_and_test


def build_parser() -> argparse.ArgumentParser:
    targets = [f"Jarvis-{t.name}" for t in JarvisTarget] + [f"MP-{t.name}" for t in MPTarget]
    parser = argparse.ArgumentParser(description="Train the paper's graph-only, concat, residual, or GPSR model.")
    parser.add_argument("--task", choices=["train_and_test", "test"], required=True)
    parser.add_argument("--target", choices=targets, required=True)
    parser.add_argument(
        "--model",
        choices=[
            "graphonly", "concat", "textresidual", "smm",
            "gpsrsimple", "gpsrnodegate", "gpsrnodegate-shuf",
            "gpsrnodegate-early", "gpsrnodegate-late",
            "gpsrnodegate-early-bounded", "gpsrnodegate-early-soft",
            "gpsrnodegate-early-bounded-srg", "gpsrnodegate-early-soft-srg",
        ],
        required=True,
    )
    parser.add_argument("--use_cif_csv", action="store_true")
    parser.add_argument("--cif_dir")
    parser.add_argument("--text_emb_path")
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha_init", type=float, default=None,
                        help="default: 0.03 (textresidual), 1e-3 (gpsr)")
    parser.add_argument("--alpha_max", type=float, default=0.20)
    parser.add_argument("--lambda_delta", type=float, default=0.0)
    parser.add_argument("--max_residual_ratio", type=float, default=float("inf"),
                        help="per-crystal Frobenius residual ratio cap; inf disables the bound")
    parser.add_argument("--lambda_res", type=float, default=0.0,
                        help="soft excess-residual penalty weight")
    parser.add_argument("--rho_target", type=float, default=0.10,
                        help="free residual-ratio threshold for soft penalty")
    parser.add_argument("--sample_gate_bias", type=float, default=3.0,
                        help="initial bias of the sample reliability gate output layer")
    parser.add_argument("--inject_position", choices=["early", "middle", "late"], default="middle")
    parser.add_argument("--inject_after", type=int, default=0,
                        help="explicit SFTConv block index after which GPSR injects; 0 = derive from --inject_position")
    parser.add_argument("--text_shuffle_seed", type=int, default=-1,
                        help=">=0 enables the shuffled-text control (dataset-level text permutation)")
    parser.add_argument("--share_text_cache", action="store_true",
                        help="graph-only runs: read the multimodal split caches instead of building a "
                             "second copy (the graph-only model never reads text_emb; identical tensors)")
    return parser


def configure(args: argparse.Namespace) -> None:
    dataset, target = args.target.split("-", 1)
    data_config.target = JarvisTarget[target] if dataset == "Jarvis" else MPTarget[target]
    data_config.batch_size = args.batch_size
    data_config.use_cif_csv = args.use_cif_csv
    data_config.cif_dir = args.cif_dir or ""
    data_config.text_shuffle_seed = args.text_shuffle_seed

    if args.use_cif_csv:
        if not args.cif_dir or not os.path.isdir(args.cif_dir):
            raise ValueError("--use_cif_csv requires an existing --cif_dir")

    model_config.num_epoch = args.num_epoch
    model_config.random_seed = args.seed
    model_config.lr = args.lr
    model_config.max_lr = args.max_lr
    model_config.weight_decay = args.weight_decay

    if args.model == "graphonly":
        model_config.model_name = "SFTGNN"
        model_config.message = "SFTGNN graph-only model"
        data_config.use_multimodal = bool(args.share_text_cache)
        if args.share_text_cache:
            if not args.text_emb_path:
                raise ValueError("--share_text_cache requires --text_emb_path")
            data_config.text_emb_path = args.text_emb_path
            model_config.text_emb_path = args.text_emb_path
        return

    if not args.text_emb_path or not os.path.isfile(args.text_emb_path):
        raise ValueError("concat and textresidual require an existing --text_emb_path")
    data_config.use_multimodal = True
    data_config.text_emb_path = args.text_emb_path
    model_config.text_emb_path = args.text_emb_path

    if args.model == "concat":
        model_config.model_name = "SFTGNNMultimodal"
        model_config.property_fusion = "concat"
        model_config.use_contrastive = False
        model_config.message = "Direct graph-text concatenation"
        return

    if args.model == "textresidual":
        model_config.model_name = "TextResidualSHFMat"
        model_config.text_residual_alpha_init = args.alpha_init if args.alpha_init is not None else 0.03
        model_config.text_residual_alpha_max = args.alpha_max
        model_config.text_residual_lambda_delta = args.lambda_delta
        model_config.message = "Graph-anchored bounded-gain textual residual"
        return

    if args.model == "smm":
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

    # --- GPSR: atom-level semantic residual injection (graph-preserving) ---
    model_config.gpsr_enabled = True
    model_config.gpsr_alpha_init = args.alpha_init if args.alpha_init is not None else 1e-3
    model_config.gpsr_injection_position = args.inject_position
    model_config.gpsr_injection_after_block = args.inject_after
    model_config.gpsr_max_residual_ratio = getattr(args, "max_residual_ratio", float("inf"))
    model_config.gpsr_lambda_res = getattr(args, "lambda_res", 0.0)
    model_config.gpsr_rho_target = getattr(args, "rho_target", 0.10)
    model_config.gpsr_use_sample_gate = False
    model_config.gpsr_sample_gate_bias = getattr(args, "sample_gate_bias", 3.0)
    if args.model == "gpsrsimple":
        model_config.model_name = "GPSRSimple"
        model_config.gpsr_use_node_gate = False
        model_config.message = "GPSR-Simple: h_i' = h_i + alpha * z_t"
    elif args.model == "gpsrnodegate":
        model_config.model_name = "GPSRNodeGate"
        model_config.gpsr_use_node_gate = True
        model_config.message = "GPSR-NodeGate: h_i' = h_i + alpha * g_i * z_t"
    elif args.model == "gpsrnodegate-early":
        model_config.model_name = "GPSRNodeGateEarly"
        model_config.gpsr_use_node_gate = True
        model_config.gpsr_injection_position = "early"
        model_config.message = "GPSR-NodeGate injected after ~25% of the graph depth"
    elif args.model in ("gpsrnodegate-early-soft", "gpsrnodegate-early-soft-srg"):
        if args.lambda_res <= 0:
            raise ValueError("gpsrnodegate-early-soft requires --lambda_res > 0")
        if args.rho_target <= 0:
            raise ValueError("--rho_target must be positive")
        tag = f"{args.lambda_res:.0e}".replace("-", "m").replace("+", "p")
        is_srg = args.model.endswith("-srg")
        model_config.model_name = f"GPSRNodeGateEarlySoftLambda{tag}" + ("SRG" if is_srg else "")
        model_config.gpsr_use_node_gate = True
        model_config.gpsr_use_sample_gate = is_srg
        model_config.gpsr_injection_position = "early"
        model_config.gpsr_max_residual_ratio = float("inf")
        model_config.message = (
            "GPSR-NodeGate Early with sample-wise soft excess-residual penalty "
            f"lambda={args.lambda_res:g}, rho_target={args.rho_target:g}, sample_gate={is_srg}"
        )
    elif args.model in ("gpsrnodegate-early-bounded", "gpsrnodegate-early-bounded-srg"):
        if not (args.max_residual_ratio > 0 and args.max_residual_ratio < float("inf")):
            raise ValueError("gpsrnodegate-early-bounded requires a finite positive --max_residual_ratio")
        tau_tag = f"{args.max_residual_ratio:.2f}".replace(".", "")
        is_srg = args.model.endswith("-srg")
        model_config.model_name = f"GPSRNodeGateEarlyBoundedTau{tau_tag}" + ("SRG" if is_srg else "")
        model_config.gpsr_use_node_gate = True
        model_config.gpsr_use_sample_gate = is_srg
        model_config.gpsr_injection_position = "early"
        model_config.message = (
            "GPSR-NodeGate Early with differentiable per-crystal residual cap "
            f"tau={args.max_residual_ratio:g}, sample_gate={is_srg}"
        )
    elif args.model == "gpsrnodegate-late":
        model_config.model_name = "GPSRNodeGateLate"
        model_config.gpsr_use_node_gate = True
        model_config.gpsr_injection_position = "late"
        model_config.message = "GPSR-NodeGate injected after ~75% of the graph depth"
    else:
        model_config.model_name = "GPSRNodeGateShuffled"
        model_config.gpsr_use_node_gate = True
        if data_config.text_shuffle_seed < 0:
            data_config.text_shuffle_seed = model_config.random_seed
        model_config.message = "GPSR-NodeGate, shuffled crystal<->text pairing (control)"


def main() -> None:
    args = build_parser().parse_args()
    configure(args)
    with TrainConfigManager(None):
        if args.task == "train_and_test":
            train_and_test()
        else:
            test_with_pretrained_model()


if __name__ == "__main__":
    main()
