from typing import Dict, Mapping, List
import itertools
from functools import reduce, wraps
from operator import add

import numpy as np
import torch

from ..config import NUM_RES, NUM_MSA_SEQ
from ..np import residue_constants as rc
from ..utils.rigid_utils import Rigid
from ..utils.tensor_utils import (
    tree_map,
    tensor_tree_map,
    batched_gather,
)

TensorDict = Dict[str, torch.Tensor]

def curry1(f):
    """Supply all arguments but the first."""
    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def cast_to_64bit_ints(protein: TensorDict) -> TensorDict:
    # We keep all ints as int64
    for k, v in protein.items():
        if v.dtype == torch.int32:
            protein[k] = v.type(torch.int64)

    return protein


def squeeze_features(protein: TensorDict):
    """Remove singleton and repeated dimensions in protein features."""
    """
    restype: [N_res,]
    seq_length: [N_res]
    num_alignments: [1]
    """
    protein["restype"] = torch.argmax(protein["restype"], dim=-1)
    for k in [
        "domain_name",
        "msa",
        "num_alignments",
        "seq_length",
        "sequence",
        "resolution",
        "between_segment_residues",
    ]:
        if k in protein:
            final_dim = protein[k].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                if torch.is_tensor(protein[k]):
                    protein[k] = torch.squeeze(protein[k], dim=-1)
                else:
                    protein[k] = np.squeeze(protein[k], axis=-1)

    for k in ["seq_length", "num_alignments"]:
        if k in protein:
            protein[k] = protein[k][0]

    return protein


@curry1
# todo
def randomly_replace_msa_with_unknown(
    protein: TensorDict, 
    replace_proportion: float
):
    """Replace a portion of the MSA with 'X'."""
    """不mask gap, 另外restype肯定没有gap"""
    """
    太奇怪了, mask token就是X,但是alphafold原文是额外的mask token, 得查一查
    msa_mask和restype_mask代表是否被mask, 1表示被mask,
    按理说原来的msa和restype可能就有X, 如果被mask了, 预测的时候应当要预测成X,
    但是为什么openfold预测的类别是23个?
    还有这里是直接替换msa和restype, 没有返回mask
    """
    x_idx = 4
    gap_idx = 5
    # torch.rand是0-1随机数
    msa_mask = torch.rand(protein["msa"].shape) < replace_proportion
    msa_mask = torch.logical_and(msa_mask, protein["msa"] != gap_idx)
    protein["msa"] = torch.where(
        msa_mask,
        torch.ones_like(protein["msa"]) * x_idx,
        protein["msa"]
    )

    restype_mask = torch.rand(protein["restype"].shape) < replace_proportion
    protein["restype"] = torch.where(
        restype_mask,
        torch.ones_like(protein["restype"]) * x_idx,
        protein["restype"],
    )
    return protein


def make_seq_mask(protein: TensorDict):
    """
    seq_mask: [N_res]"""
    protein["seq_mask"] = torch.ones(
        protein["restype"].shape, dtype=torch.float32
    )
    return protein


def make_msa_mask(protein: TensorDict):
    """Mask features are all ones, but will later be zero-padded."""
    """
    msa_mask: [N_seq, N_res]全1
    # msa_row_mask[N_seq,]全1
    """
    protein["msa_mask"] = torch.ones(protein["msa"].shape, dtype=torch.float32)
    # protein["msa_row_mask"] = torch.ones(
    #     (protein["msa"].shape[0]), dtype=torch.float32
    # )
    return protein


def get_backbone_frames(protein: TensorDict, eps=1e-8):
    restype = protein["restype"]
    all_atom_positions = protein["all_atom_positions"]
    all_atom_mask = protein["all_atom_mask"]
    batch_dims = len(restype.shape[:-1])

    # todo 这里有个问题, openfold
    restype_bb_base_atom_names = np.full([5, 3], "", dtype=object)
    for res_idx, resname in enumerate(rc.restypes):
        restype_bb_base_atom_names[res_idx] = rc.tmp_mapping[resname]
    
    
    lookuptable = rc.bb_atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_bb_base_atom4_idx = lookup(
        restype_bb_base_atom_names
    )
    restype_bb_base_atom4_idx = restype.new_tensor(
        restype_bb_base_atom4_idx
    )
    # [*, 5, 3]
    restype_bb_base_atom4_idx = (
        restype_bb_base_atom4_idx.view(
            *((1,) * batch_dims), *restype_bb_base_atom4_idx.shape
        )
    )
    residx_bb_base_atom4_idx = batched_gather(
        restype_bb_base_atom4_idx,
        restype,
        dim=-2, # 这里与openfold不同, 因为我们只有一个组
        no_batch_dims=batch_dims,
    )
    # [*, N_res, 3, 3]  C, P, N
    base_atom_pos = batched_gather(
        all_atom_positions,
        residx_bb_base_atom4_idx,
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )
    # group_exists [N_res, 8] 在这里都是1

    restype_bb_mask = all_atom_mask.new_ones(
        (*restype.shape[:-1], 5),
    )
    # ! 我在这里把X视为不存在backbone
    restype_bb_mask[..., -1] = 0
    
    bb_exists = batched_gather(
        restype_bb_mask,
        restype,
        dim=-1,
        no_batch_dims=batch_dims,
    )
    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_bb_base_atom4_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * bb_exists
    gt_bb = Rigid.from_3_points_svd(
        atom_P=base_atom_pos[..., 1, :],  # P
        C4_prime=base_atom_pos[..., 0, :],  # C
        glycos_N=base_atom_pos[..., 2, :],  # N
    )
    gt_bb_tensor = gt_bb.to_tensor_4x4()
    
    protein["backbone_rigid_tensor"] = gt_bb_tensor
    protein["backbone_rigid_mask"] = gt_exists

    return protein


def glycos_N_fn(
    restype: torch.Tensor, 
    all_atom_positions: torch.Tensor, 
    all_atom_mask: torch.Tensor
):
    """Create glycos_N features."""
    is_purine = (restype == rc.restype_order["A"]) | (restype == rc.restype_order["G"])
    n1_idx = rc.bb_atom_order["N1"]
    n9_idx = rc.bb_atom_order["N9"]
    glycos_N = torch.where(
        torch.tile(is_purine[..., None], [1] * len(is_purine.shape) + [3]),
        all_atom_positions[..., n9_idx, :],
        all_atom_positions[..., n1_idx, :],
    )

    if all_atom_mask is not None:
        # ! 如果是X, 则mask相应位置, 因为不知道是N1还是N9, 其他原子的提取不影响
        is_unknown = restype == rc.restype_order_with_x["X"]
        glycos_N_mask = torch.where(
            is_purine, 
            all_atom_mask[..., n9_idx], 
            all_atom_mask[..., n1_idx]
        )
        glycos_N_mask[is_unknown] = 0.0
        return glycos_N, glycos_N_mask
    else:
        return glycos_N

# ! 添加了其他原子
def make_glycos_N(protein: TensorDict):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    (
        protein["glycos_N"],
        protein["glycos_N_mask"],
    ) = glycos_N_fn(
        protein["restype"],
        protein["all_atom_positions"],
        protein["all_atom_mask"],
    )
    (
        protein["C4_prime"], 
        protein["C4_prime_mask"],
    ) = make_atom(
        "C4'",
        protein["all_atom_positions"],
        protein["all_atom_mask"],
    )
    (
        protein["atom_P"], 
        protein["atom_P_mask"],
    ) = make_atom(
        "P",
        protein["all_atom_positions"],
        protein["all_atom_mask"],
    )
    
    return protein

def make_atom(
    atom_name: str,
    all_atom_positions: torch.Tensor, 
    all_atom_mask: torch.Tensor
):
    atom_idx = rc.bb_atom_order[atom_name]
    # [..., N_res, 4, 3]
    position = all_atom_positions[..., atom_idx, :]

    if all_atom_mask is not None:
        mask = all_atom_mask[..., atom_idx]
        return position, mask
    else:
        return position


def make_one_hot(
    x: torch.Tensor, 
    num_classes: int
) -> torch.Tensor:
    """
    return [..., num_classes]
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


def shaped_categorical(
    probs: torch.Tensor, 
    epsilon: float = 1e-10
):
    ds = probs.shape
    num_classes = ds[-1]
    distribution = torch.distributions.categorical.Categorical(
        torch.reshape(probs + epsilon, [-1, num_classes])
    )
    counts = distribution.sample()
    return torch.reshape(counts, ds[:-1])


@curry1
def make_masked_msa(
    protein: torch.Tensor, 
    config, 
    replace_fraction: float, 
    seed: int
):
    """Create data for BERT on raw MSA."""
    """
    [msa]是额外的一个token, 不是X
    bert_mask: [..., N_seq, N_res] 1为被replace
    true_msa: [..., N_seq, N_res] 原来的msa
    msa: [..., N_seq, N_res] mask之后的msa
    """
    device = protein["msa"].device

    # Add a random amino acid uniformly.
    # [ACGUX-]
    random_aa = torch.tensor(
        [1 / rc.restype_num] * rc.restype_num + [0.0, 0.0], 
        dtype=torch.float32, 
        device=device
    )
    # [..., N_seq, N_res, num_classes]
    categorical_probs = (
        config.uniform_prob * random_aa
        + config.same_prob * make_one_hot(
            protein["msa"], 
            rc.restype_num + 2
        )
    )

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = list(
        reduce(add, [(0, 0) for _ in range(len(categorical_probs.shape))])
    )
    pad_shapes[1] = 1
    mask_prob = (
        1.0 - config.same_prob - config.uniform_prob
    )
    assert mask_prob >= 0.0
    # 在[mask]的地方(第7列)填充mask_prob
    # 现在categorical_probs就是每个位置都是一个概率，
    # 表示原来的残基被替换这些位置的概率（如果要被替换的话）
    categorical_probs = torch.nn.functional.pad(
        categorical_probs, pad_shapes, value=mask_prob,
    )

    g = None
    if seed is not None:
        g = torch.Generator(device=protein["msa"].device)
        g.manual_seed(seed)

    # mask_position: 1为被替换
    sample = torch.rand(protein["msa"].shape, device=device, generator=g)
    mask_position = sample < replace_fraction
    # [..., N_seq, N_res]
    bert_msa = shaped_categorical(categorical_probs)
    bert_msa = torch.where(mask_position, bert_msa, protein["msa"])

    # Mix real and masked MSA
    protein["bert_mask"] = mask_position.to(torch.float32)
    protein["true_msa"] = protein["msa"]
    protein["msa"] = bert_msa

    return protein


@curry1
def make_msa_feat(protein: TensorDict):
    """Create and concatenate MSA features."""
    """
    msa_feat: [..., N_seq, N_res, 7]
    target_feat: [..., N_res, 6]
    """
    has_break = torch.clip(
        protein["between_segment_residues"].to(torch.float32), 0, 1
    )
    # AUCGX
    aatype_1hot = make_one_hot(protein["restype"], 4+1)
    # 这样target_feat的最后一维是4+1+1=6
    target_feat = [
        torch.unsqueeze(has_break, dim=-1),
        aatype_1hot,  # Everyone gets the original sequence.
    ]
    # AUCGX-[mask]
    msa_1hot = make_one_hot(protein["msa"], 4+3)

    msa_feat = [msa_1hot]

    protein["msa_feat"] = torch.cat(msa_feat, dim=-1)
    protein["target_feat"] = torch.cat(target_feat, dim=-1)
    return protein


@curry1
def select_feat(
    protein: TensorDict, 
    feature_list: List
):
    return {k: v for k, v in protein.items() if k in feature_list}


@curry1
def random_crop_to_size(
    protein: TensorDict,
    crop_size: int,
    shape_schema: Dict,
    seed: int = None,
):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    # We want each ensemble to be cropped the same way
    """
    对能做crop的执行crop, seq_length更改
    """

    g = None
    if seed is not None:
        g = torch.Generator(device=protein["seq_length"].device)
        g.manual_seed(seed)

    seq_length = protein["seq_length"]

    num_res_crop_size = min(int(seq_length), crop_size)

    def _randint(lower, upper):
        return int(torch.randint(
                lower,
                upper + 1,
                (1,),
                device=protein["seq_length"].device,
                generator=g,
        )[0])

    n = seq_length - num_res_crop_size
    if "use_clamped_fape" in protein and protein["use_clamped_fape"] == 1.:
        right_anchor = n
    else:
        x = _randint(0, n)
        right_anchor = n - x

    num_res_crop_start = _randint(0, right_anchor)

    for k, v in protein.items():
        if NUM_RES not in shape_schema[k]:
            continue

        slices = []
        for dim_size, dim in zip(shape_schema[k], v.shape):
            is_num_res = dim_size == NUM_RES
            crop_start = num_res_crop_start if is_num_res else 0
            crop_size = num_res_crop_size if is_num_res else dim
            slices.append(slice(crop_start, crop_start + crop_size))
        protein[k] = v[slices]

    protein["seq_length"] = protein["seq_length"].new_tensor(num_res_crop_size)
    
    return protein


@curry1
def make_fixed_size(
    protein: TensorDict,
    shape_schema: Dict,
    num_res: int = 0,
):
    """Guess at the MSA and sequence dimension to make fixed size."""
    """0填充, 保证所有样本的shape都一样"""
    """有趣, 前面的seq_mask和msa_mask都是全1, 在这里零填充后就自动实现mask了"""
    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: 1,
    }

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [
            pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
        ]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)
    
    return protein