import re
import copy
import importlib
import ml_collections as mlc


def set_inf(config, inf):
    # 递归设置inf参数
    for k, v in config.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            config[k] = inf


def enforce_config_constraints(config):
    # region: 检查互斥的参数
    def string_to_setting(s):
        path = s.split('.')
        setting = config
        for p in path:
            setting = setting.get(p)

        return setting

    mutually_exclusive_bools = [
        (
            "globals.use_lma",
            "globals.use_flash",
            "globals.use_deepspeed_evo_attention"
        ),
    ]

    for options in mutually_exclusive_bools:
        option_settings = [string_to_setting(o) for o in options]
        if sum(option_settings) > 1:
            raise ValueError(f"Only one of {', '.join(options)} may be set at a time")
    # endregion
    

    fa_is_installed = importlib.util.find_spec("flash_attn") is not None
    if config.globals.use_flash and not fa_is_installed:
        raise ValueError("use_flash requires that FlashAttention is installed")

    deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
    ds4s_is_installed = deepspeed_is_installed and importlib.util.find_spec(
        "deepspeed.ops.deepspeed4science") is not None
    if config.globals.use_deepspeed_evo_attention and not ds4s_is_installed:
        raise ValueError(
            "use_deepspeed_evo_attention requires that DeepSpeed be installed "
            "and that the deepspeed.ops.deepspeed4science package exists"
        )


def model_config(
    name: str, # 训练和推理模式
    train: bool = False, 
    low_prec: bool = False, 
    long_sequence_inference: bool = False
):
    c = copy.deepcopy(config)
    # TRAINING PRESETS
    if name == "e2e":
        c.globals.is_e2e = True
        c.model.evoformer_stack.is_e2e = True
    elif name == "geom":
        c.globals.is_e2e = False
        c.model.evoformer_stack.is_e2e = False
    # INFERENCE PRESETS
    else:
        raise ValueError("Invalid model name")
    
    if long_sequence_inference:
        assert not train
        c.globals.offload_inference = True
        # Default to DeepSpeed memory-efficient attention kernel unless use_lma is explicitly set
        c.globals.use_deepspeed_evo_attention = True if not c.globals.use_lma else False
        c.globals.use_flash = False
        c.model.evoformer_stack.tune_chunk_size = False
    
    if train:
        c.globals.blocks_per_ckpt = 1
        c.globals.chunk_size = None
        c.globals.use_lma = False
        c.globals.offload_inference = False
    
    if low_prec:
        c.globals.eps = 1e-4
        # If we want exact numerical parity with the original, inf can't be
        # a global constant
        set_inf(c, 1e4)
    
    enforce_config_constraints(c)

    return c


# 这些预先定义的值需要修改
c_z = mlc.FieldReference(64, field_type=int)
c_m = mlc.FieldReference(64, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(64, field_type=int)

# For seqemb mode, dimension size of the per-residue sequence embedding passed to the model
# In current model, the dimension size is the ESM-1b dimension size i.e. 1280.
# preemb_dim_size = mlc.FieldReference(1280, field_type=int)

blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(36+2, field_type=int)
eps = mlc.FieldReference(1e-8, field_type=float)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)
loss_unit_distance = mlc.FieldReference(10.0, field_type=float)# trans_scale_factor


NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"

config = mlc.ConfigDict({
    "data": {
        "common": {
            "feat": {
                "restype": [NUM_RES],
                "all_atom_mask": [NUM_RES, None],
                "all_atom_positions": [NUM_RES, None, None],
                "backbone_rigid_mask": [NUM_RES],
                "backbone_rigid_tensor": [NUM_RES, None, None],
                "bert_mask": [NUM_MSA_SEQ, NUM_RES],
                "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
                "msa_mask": [NUM_MSA_SEQ, NUM_RES],
                # "msa_row_mask": [NUM_MSA_SEQ],
                "no_recycling_iters": [],
                "glycos_N": [NUM_RES, None],
                "glycos_N_mask": [NUM_RES],
                "C4_prime": [NUM_RES, None],
                "C4_prime_mask": [NUM_RES],
                "atom_P": [NUM_RES, None],
                "atom_P_mask": [NUM_RES],
                # "residue_index": [NUM_RES],
                "resolution": [],
                "seq_length": [],
                "seq_mask": [NUM_RES],
                "target_feat": [NUM_RES, None],
                "true_msa": [NUM_MSA_SEQ, NUM_RES],
                "use_clamped_fape": [],
                "ss": [NUM_RES, NUM_RES, None]
            },
            "masked_msa": {
                "same_prob": 0.5, # todo 我的msa只有一条, 担心bert mask学习太难, 调整了一些比率
                "uniform_prob": 0.1, # todo 我的msa只有一条, 担心bert mask学习太难, 调整了一些比率
            },
            "max_recycling_iters": 3,
            "unsupervised_features": [
                "restype",
                "msa",
                "ss",
                "num_alignments",
                "seq_length",
                "between_segment_residues",
                "no_recycling_iters",
            ],

        },
        "seqemb_mode": { # Configuration for sequence embedding mode
            "enabled": False, # If True, use seq emb instead of MSA
        },
        "supervised": {
            "clamp_prob": 0.9,
            "supervised_features": [
                "all_atom_mask",
                "all_atom_positions",
                "resolution",
                "use_clamped_fape",
            ],
        },
        "predict": {
            "fixed_size": True,
            "masked_msa_replace_fraction": 0.1, # todo 我的msa只有一条, 担心bert mask学习太难, 调整了一些比率
            "crop": False,
            "crop_size": None,
            "spatial_crop_prob": None,
            # "interface_threshold": None,
            "supervised": False,
            "uniform_recycling": False,
        },
        "eval": {
            "fixed_size": True,
            "masked_msa_replace_fraction": 0.15,
            "crop": False,
            "crop_size": None,
            "spatial_crop_prob": None,
            # "interface_threshold": None,
            "supervised": True,
            "uniform_recycling": False,
        },
        "train": {
            "fixed_size": True,
            "masked_msa_replace_fraction": 0.15,
            "crop": True,
            "crop_size": 200,
            "spatial_crop_prob": 0.,
            # "interface_threshold": None,
            "supervised": True,
            "clamp_prob": 0.9,
            "uniform_recycling": True,
        },
        "data_module": {
            "data_loaders": {
                "batch_size": 1,    # todo
                "num_workers": 16,
                "pin_memory": True,
            },
        },
    },
    # Recurring FieldReferences that can be changed globally here
    "globals": {
        "is_e2e": True, # 采用e2e模型还是geometry模型
        "blocks_per_ckpt": blocks_per_ckpt,
        "chunk_size": chunk_size,
        # Use DeepSpeed memory-efficient attention kernel. Mutually
        # exclusive with use_lma and use_flash.
        "use_deepspeed_evo_attention": False,
        # Use Staats & Rabe's low-memory attention algorithm. Mutually
        # exclusive with use_deepspeed_evo_attention and use_flash.
        "use_lma": False,
        # Use FlashAttention in selected modules. Mutually exclusive with 
        # use_deepspeed_evo_attention and use_lma. Doesn't work that well
        # on long sequences (>1000 residues).
        "use_flash": False,
        "offload_inference": False,
        "c_z": c_z,
        "c_m": c_m,
        "c_t": c_t,
        "c_e": c_e,
        "c_s": c_s,
        "eps": eps,
        "seqemb_mode_enabled": False, # Global flag for enabling seq emb mode
    },
    "model": {
        "_mask_trans": False,
        "input_embedder": {
            "tf_dim": 6, # todo 论文里是21, openfold多了一个between_segment_residues
            "msa_dim": 7, # todo
            "c_m": c_m,
            "c_z": c_z,
            "max_len_seq": 800,
            "no_pos_bins_1d": 14,
            "pos_wsize_2d": 64,
        },
        "ss_embedder": {
            "ss_dim": 4, 
            "c_z": c_z,
        },
        "recycling_embedder": {
            "c_m": c_m,
            "c_z": c_z,
            "dis_encoding_dim": 64,
        },
        "evoformer_stack": {
            "c_m": c_m,
            "c_z": c_z,
            "c_hidden_msa_att": 16,
            "c_hidden_opm": 16,
            "c_hidden_mul": c_z,
            "c_hidden_pair_att": 16,
            "c_s": c_s,
            "no_heads_msa": 8,
            "no_heads_pair": 8,
            "no_blocks": 48,
            "transition_n": 2,
            "msa_dropout": 0.15,
            "pair_dropout": 0.25,
            "no_column_attention": True,
            "opm_first": False,
            "fuse_projection_weights": False,
            "blocks_per_ckpt": blocks_per_ckpt,
            "clear_cache_between_blocks": False,
            "tune_chunk_size": tune_chunk_size,
            "inf": 1e9,
            "eps": eps,  # 1e-10,
            "is_e2e": True,
        },
        "structure_module": {
            "c_s": c_s,
            "c_z": c_z,
            "c_ipa": 16,
            "no_heads_ipa": 8,
            "no_qk_points": 4,
            "no_v_points": 8,
            "dropout_rate": 0.1,
            "no_blocks": 4,
            "no_transition_layers": 1,
            "trans_scale_factor": loss_unit_distance,
            "epsilon": eps,  # 1e-12,
            "inf": 1e5,
        },
        "aux_heads": {
            "distogram": {
                "c_z": c_z,
                "no_bins": aux_distogram_bins,
            },
            "masked_msa": {
                "c_m": c_m,
                "c_out": 4+3, # todo
            },
        },
        "geom_heads": {
            "PP": { # P(i)-P(j) distance
                "c_z": c_z,
                "c_hidden": c_z,
                "no_blocks": 1,
                "no_bins": 56+2,
                "symmetrize": True,
            },
            "CC": { # C4'(i)-C4'(j) distance
                "c_z": c_z,
                "c_hidden": c_z,
                "no_blocks": 1,
                "no_bins": 44+2,
                "symmetrize": True,
            },
            "NN": { # N(i)-N(j) distance
                "c_z": c_z,
                "c_hidden": c_z,
                "no_blocks": 1,
                "no_bins": 32+2,
                "symmetrize": True,
            },
            "PCCP": { # P(i)-C4'(i)-C4'(j)-P(j) distance
                "c_z": c_z,
                "c_hidden": c_z,
                "no_blocks": 1,
                "no_bins": 36+1,
                "symmetrize": True,
            },
            "CNNC": { # C4'(i)-N(i)-N(j)-C4'(j) distance
                "c_z": c_z,
                "c_hidden": c_z,
                "no_blocks": 1,
                "no_bins": 36+1,
                "symmetrize": True,
            },
            "PNNP": { # P(i)-N(i)-N(j)-P(j) distance
                "c_z": c_z,
                "c_hidden": c_z,
                "no_blocks": 1,
                "no_bins": 36+1,
                "symmetrize": True,
            },
            "masked_msa": {
                "c_m": c_m,
                "c_out": 4+3, # todo
            },
        },
        # A negative value indicates that no early stopping will occur, i.e.
        # the model will always run `max_recycling_iters` number of recycling
        # iterations. A positive value will enable early stopping if the
        # difference in pairwise distances is less than the tolerance between
        # recycling steps.
        "recycle_early_stop_tolerance": -1.
    },
    "struct_loss": {
        "distogram": {
            "min_bin": 2.0,
            "max_bin": 40.0,
            "no_bins": aux_distogram_bins,
            "reduce": True, # 是否要除以N_res ^ 2
            "eps": eps,  # 1e-6,
            "weight": 0.6,
        },
        "fape": {
            "backbone": {
                "clamp_distance": 30.0,
                "loss_unit_distance": loss_unit_distance,
                "reduce": True,
                "eps": 1e-3,
            },
            "weight": 1.5,
        },
        "masked_msa": {
            "num_classes": 4+3, 
            "eps": eps,  # 1e-8,
            "weight": 1.0,
        },
        # "eps": eps,
    },
    "geom_loss": {
        "PP": {
            "min_bin": 2.0,
            "max_bin": 30.0,
            "no_bins": 56+2,
            "reduce": True, # 是否要除以N_res ^ 2
            "eps": eps,  # 1e-6,
            "weight": 1.0,
        },
        "CC": {
            "min_bin": 2.0,
            "max_bin": 24.0,
            "no_bins": 44+2,
            "reduce": True, # 是否要除以N_res ^ 2
            "eps": eps,  # 1e-6,
            "weight": 1.0,
        },
        "NN": {
            "min_bin": 2.0,
            "max_bin": 18.0,
            "no_bins": 32+2,
            "reduce": True, # 是否要除以N_res ^ 2
            "eps": eps,  # 1e-6,
            "weight": 1.0,
        },
        "PCCP": {
            "min_bin": -180,
            "max_bin": 180,
            "no_bins": 36+1,
            "max_dist": 24,
            "reduce": True, # 是否要除以N_res ^ 2
            "eps": eps,  # 1e-6,
            "weight": 0.5,
        },
        "PNNP": {
            "min_bin": -180,
            "max_bin": 180,
            "no_bins": 36+1,
            "max_dist": 18,
            "reduce": True, # 是否要除以N_res ^ 2
            "eps": eps,  # 1e-6,
            "weight": 0.5,
        },
        "CNNC": {
            "min_bin": -180,
            "max_bin": 180,
            "no_bins": 36+1,
            "max_dist": 18,
            "reduce": True, # 是否要除以N_res ^ 2
            "eps": eps,  # 1e-6,
            "weight": 0.5,
        },
        "masked_msa": {
            "num_classes": 4+3, 
            "eps": eps,  # 1e-8,
            "weight": 1.0,
        },
    },
    "ema": {"decay": 0.999},
})
