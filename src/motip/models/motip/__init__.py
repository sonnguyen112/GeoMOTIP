# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Modified for Trail-MOTIP V0.

from .motip import MOTIP

from structures.args import Args

from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder
from models.motip.trail import VisualTrail, trail_mode
from models.motip.geometry import GEOMETRY_FEATURE_DIM
from pathlib import Path
import torch


def build(config: dict):
    mode = trail_mode(config)
    # Generate DETR args:
    detr_args = Args()

    # 1. backbone:
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]

    # 2. transformer:
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]

    detr_framework = config["DETR_FRAMEWORK"].lower()
    match detr_framework:
        case "deformable_detr":
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
        case _:
            raise NotImplementedError(
                f"DETR framework {config['DETR_FRAMEWORK']} is not supported."
            )

    # Build each component:
    # 1. trajectory modeling + optional Trail-MOTIP V0 branch:
    _trajectory_modeling = TrajectoryModeling(
        detr_dim=config["DETR_HIDDEN_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        feature_dim=config["FEATURE_DIM"],
        trail_enabled=mode == "v0",
        trail_grid_h=config.get("TRAIL_GRID_H", 12),
        trail_grid_w=config.get("TRAIL_GRID_W", 20),
        trail_alpha=config.get("TRAIL_ALPHA", 0.8),
        trail_encoder_hidden_dim=config.get("TRAIL_ENCODER_HIDDEN_DIM", 128),
        trail_raster_sharpness=config.get("TRAIL_RASTER_SHARPNESS", 80.0),
        trail_fusion_scale=config.get("TRAIL_FUSION_SCALE", 1.0),
        preserve_trail_rng="USE_TRAIL" in config,
    ) if config["ONLY_DETR"] is False else None

    geometry_input_stats = None
    geometry_stats_path = config.get("GEOMETRY_INPUT_STATS_PATH")
    if config.get("USE_GEOMETRY", False) and geometry_stats_path:
        stats_path = Path(geometry_stats_path)
        if not stats_path.is_absolute():
            stats_path = Path.cwd() / stats_path
        stats_blob = torch.load(stats_path, map_location="cpu", weights_only=True)
        try:
            geometry_input_stats = stats_blob["G4_full_geometry"]["stats"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "GEOMETRY_INPUT_STATS_PATH must contain "
                "G4_full_geometry/stats"
            ) from exc
        if tuple(geometry_input_stats.shape) != (2, GEOMETRY_FEATURE_DIM):
            raise ValueError(
                "G4 geometry stats must have shape [2, 66], got "
                f"{tuple(geometry_input_stats.shape)}"
            )

    # 2. ID decoder:
    _id_decoder = IDDecoder(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
        use_trail=mode == "visual",
        use_token_type=config.get("TRAIL_USE_TOKEN_TYPE", False),
        trail_fusion_mode=config.get("TRAIL_FUSION_MODE", "shared_attention"),
        trail_freeze_baseline=config.get("TRAIL_FREEZE_BASELINE", False),
        trail_residual_init=config.get("TRAIL_RESIDUAL_INIT", 0.0),
        use_geometry=config.get("USE_GEOMETRY", False),
        geometry_dim=config.get("GEOMETRY_DIM", GEOMETRY_FEATURE_DIM),
        geometry_hidden_dim=config.get("GEOMETRY_HIDDEN_DIM", 128),
        geometry_lags=tuple(config.get("GEOMETRY_LAGS", [1, 2, 4, 8])),
        geometry_fixed_lambda=config.get("GEOMETRY_FIXED_LAMBDA", 0.01),
        geometry_base_temperature=config.get("GEOMETRY_BASE_TEMPERATURE", 1.0),
        geometry_temperature=config.get("GEOMETRY_TEMPERATURE", 1.0),
        geometry_aux_weight=config.get("GEOMETRY_AUX_WEIGHT", 1.0),
        geometry_fused_weight=config.get("GEOMETRY_FUSED_WEIGHT", 1.0),
        geometry_learned_lambda=config.get("GEOMETRY_LEARNED_LAMBDA", False),
        geometry_freeze_baseline=config.get("FREEZE_MOTIP", False),
        geometry_validate_alignment=config.get("GEOMETRY_VALIDATE_ALIGNMENT", True),
        geometry_input_clip=config.get("GEOMETRY_INPUT_CLIP", 50.0),
        geometry_input_normalize=config.get("GEOMETRY_INPUT_NORMALIZE", True),
        geometry_input_stats=geometry_input_stats,
        geometry_candidate_diagnostics=config.get("GEOMETRY_CANDIDATE_DIAGNOSTICS", False),
        use_spatial_anchor=config.get("USE_SPATIAL_ANCHOR", False),
        use_motip2_spatial_anchor=config.get("USE_MOTIP2_SPATIAL_ANCHOR", False),
    ) if config["ONLY_DETR"] is False else None

    # Construct MOTIP model:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
    )

    # Create optional modules after all original modules. Preserve the global
    # RNG stream so enabling V1 does not change subsequent data augmentation.
    if mode == "visual":
        with torch.random.fork_rng(devices=[]):
            motip_model.visual_trail = VisualTrail(config)
        detr.trail_source_size = tuple(config.get("TRAIL_SOURCE_SIZE", [48, 80]))
    if config.get("TRAIL_FREEZE_BASELINE", False):
        if _trajectory_modeling is not None:
            _trajectory_modeling.requires_grad_(False)
        if _id_decoder is not None:
            _id_decoder.freeze_baseline_branch()
    if config.get("FREEZE_MOTIP", False):
        if _trajectory_modeling is not None:
            _trajectory_modeling.requires_grad_(False)
        if _id_decoder is not None:
            _id_decoder.freeze_geometry_baseline()
    motip_model.freeze_detr = config.get("FREEZE_DETR", False)
    tuning = config.get("DETR_TRAIN_MODE", "frozen" if motip_model.freeze_detr else "full")
    if tuning not in ("frozen", "partial", "full"):
        raise ValueError("DETR_TRAIN_MODE must be frozen, partial, or full")
    if motip_model.freeze_detr or tuning == "frozen":
        motip_model.freeze_detr = True
        detr.requires_grad_(False)
    elif tuning == "partial":
        names = config.get("DETR_TRAINABLE_NAMES", ["transformer.decoder", "class_embed", "bbox_embed"])
        for name, param in detr.named_parameters():
            param.requires_grad_(any(fragment in name for fragment in names))

    return motip_model, detr_criterion
