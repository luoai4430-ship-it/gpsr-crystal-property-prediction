from typing import Literal


class Config:
    message: str = "SFTGNN graph-only model"
    model_name: Literal[
        "SFTGNN", "SFTGNNMultimodal", "TextResidualSHFMat",
        "GPSRSimple", "GPSRNodeGate", "GPSRNodeGateShuffled",
    ] = "SFTGNN"
    random_seed: int = 123

    node_feature: int = 128
    num_layers: int = 5
    dist_dim: int = 64
    sh_dim: int = 64
    use_sh: bool = True
    use_dist: bool = True

    lr: float = 1e-3
    max_lr: float = 1e-3
    weight_decay: float = 1e-4
    begin_epoch: int = 0
    num_epoch: int = 300

    latent_dim: int = 128
    text_emb_dim: int = 768
    text_hidden_dim: int = 128
    text_out_dim: int = 64
    use_contrastive: bool = False
    contrastive_temperature: float = 0.07
    contrastive_loss_weight: float = 0.1
    property_fusion: Literal["crystal_only", "concat"] = "concat"
    text_emb_path: str = ""

    text_residual_hidden_dim: int = 128
    text_residual_dropout: float = 0.1
    text_residual_alpha_init: float = 0.03
    text_residual_alpha_max: float = 0.2
    text_residual_lambda_delta: float = 0.0

    # --- GPSR: graph-preserving semantic residual injection (atom level) ---
    gpsr_enabled: bool = False
    gpsr_use_node_gate: bool = True
    gpsr_alpha_init: float = 1e-3
    gpsr_injection_after_block: int = 0  # 0 -> resolved from gpsr_injection_position
    gpsr_injection_position: str = "middle"
    gpsr_dropout: float = 0.0
    gpsr_max_residual_ratio: float = float("inf")
    gpsr_lambda_res: float = 0.0
    gpsr_rho_target: float = 0.10

    @staticmethod
    def info() -> str:
        attrs = [
            "model_name", "random_seed", "node_feature", "num_layers",
            "dist_dim", "sh_dim", "use_sh", "lr", "max_lr",
            "weight_decay",
        ]
        if Config.model_name != "SFTGNN":
            attrs += [
                "text_emb_dim", "text_hidden_dim", "text_out_dim",
                "property_fusion", "use_contrastive",
            ]
        if Config.model_name == "TextResidualSHFMat":
            attrs += [
                "text_residual_hidden_dim", "text_residual_dropout",
                "text_residual_alpha_init", "text_residual_alpha_max",
                "text_residual_lambda_delta",
            ]
        if Config.gpsr_enabled:
            attrs += [
                "gpsr_use_node_gate", "gpsr_alpha_init", "gpsr_injection_position",
                "gpsr_injection_after_block", "gpsr_dropout",
                "gpsr_max_residual_ratio", "gpsr_lambda_res", "gpsr_rho_target",
            ]
        lines = ["model_config:", "{"]
        lines.extend(f"    {attr}={getattr(Config, attr)}" for attr in attrs)
        lines.append("}")
        return "\n".join(lines)
