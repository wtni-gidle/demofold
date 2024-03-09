import re
import copy
import importlib
import ml_collections as mlc


c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(384, field_type=int)

# For seqemb mode, dimension size of the per-residue sequence embedding passed to the model
# In current model, the dimension size is the ESM-1b dimension size i.e. 1280.
# preemb_dim_size = mlc.FieldReference(1280, field_type=int)

blocks_per_ckpt = mlc.FieldReference(None, field_type=int)
chunk_size = mlc.FieldReference(4, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
eps = mlc.FieldReference(1e-8, field_type=float)
tune_chunk_size = mlc.FieldReference(True, field_type=bool)

NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"
NUM_EXTRA_SEQ = "extra msa placeholder"
NUM_TEMPLATES = "num templates placeholder"

config = mlc.ConfigDict(
    {
        "data": {
            "common": {
                "feat": {
                    "restype": [NUM_RES],
                    # "all_atom_mask": [NUM_RES, None],
                    # "all_atom_positions": [NUM_RES, None, None],
                    "backbone_rigid_mask": [NUM_RES],
                    "backbone_rigid_tensor": [NUM_RES, None, None],
                    "bert_mask": [NUM_MSA_SEQ, NUM_RES],
                    "is_distillation": [],
                    "msa_feat": [NUM_MSA_SEQ, NUM_RES, None],
                    "msa_mask": [NUM_MSA_SEQ, NUM_RES],
                    # "msa_row_mask": [NUM_MSA_SEQ],
                    # "no_recycling_iters": [],
                    # "pseudo_beta": [NUM_RES, None],
                    # "pseudo_beta_mask": [NUM_RES],
                    # "residue_index": [NUM_RES],
                    # "resolution": [],
                    "seq_length": [],
                    "seq_mask": [NUM_RES],
                    "target_feat": [NUM_RES, None],
                    "true_msa": [NUM_MSA_SEQ, NUM_RES],
                    "use_clamped_fape": [],
                },
                "masked_msa": {
                    "profile_prob": 0.1,
                    "same_prob": 0.1,
                    "uniform_prob": 0.1,
                },
                "max_recycling_iters": 3,
                # "unsupervised_features": [
                #     "aatype",
                #     "residue_index",
                #     "msa",
                #     "num_alignments",
                #     "seq_length",
                #     "between_segment_residues",
                #     "deletion_matrix",
                #     "no_recycling_iters",
                # ],
            },
            "seqemb_mode": { # Configuration for sequence embedding mode
                "enabled": False, # If True, use seq emb instead of MSA
            },
            "supervised": {
                # "clamp_prob": 0.9,
                "supervised_features": [
                    "all_atom_mask",
                    "all_atom_positions",
                    "resolution",
                    "use_clamped_fape",
                    "is_distillation",
                ],
            },
            "predict": {
                "fixed_size": True,
                "masked_msa_replace_fraction": 0.15,
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
                "crop_size": 256,
                "spatial_crop_prob": 0.,
                # "interface_threshold": None,
                "supervised": True,
                "clamp_prob": 0.9,
                "max_distillation_msa_clusters": 1000,
                "uniform_recycling": True,
                "distillation_prob": 0.75,
            },
            "data_module": {
                "data_loaders": {
                    "batch_size": 1,
                    "num_workers": 16,
                    "pin_memory": True,
                },
            },
        },
        # Recurring FieldReferences that can be changed globally here
        "globals": {
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
                "tf_dim": 22, # todo
                "msa_dim": 49, # todo
                "c_m": c_m,
                "c_z": c_z,
                "max_len_seq": 800,
                "no_pos_bins_1d": 14,
                "pos_wsize_2d": 64,
            },
            "recycling_embedder": {
                "c_m": c_m,
                "c_z": c_z,
                "dis_encoding_dim": 64,
            },
            "evoformer_stack": {
                "c_m": c_m,
                "c_z": c_z,
                "c_hidden_msa_att": 32,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "c_s": c_s,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "no_blocks": 48,
                "transition_n": 4,
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "no_column_attention": False,
                "opm_first": False,
                "fuse_projection_weights": False,
                "blocks_per_ckpt": blocks_per_ckpt,
                "clear_cache_between_blocks": False,
                "tune_chunk_size": tune_chunk_size,
                "inf": 1e9,
                "eps": eps,  # 1e-10,
            }
        }
    }
)