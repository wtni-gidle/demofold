import torch
import torch.nn as nn

from .rigid_utils import Rigid

from ..np import residue_constants as rc


def glycos_N_fn(
    restype: torch.Tensor, 
    all_atom_positions: torch.Tensor, 
    all_atom_mask: torch.Tensor
):
    """Create glycos_N features."""
    """
    目前只需要用到C4', P, N1/N9三个原子的坐标, 因此先只用四个原子代替全原子
    all_atom_positions [..., N_res, bb_atom_type_num, 3] 只含四个原子, 顺序为bb_atom_order
    """
    is_purine = (restype == rc.restype_order["A"]) | (restype == rc.restype_order["G"])
    n1_idx = rc.bb_atom_order["N1"]
    n9_idx = rc.bb_atom_order["N9"]
    glycos_N = torch.where(
        torch.tile(is_purine[..., None], [1] * len(is_purine.shape) + [3]),
        all_atom_positions[..., n9_idx, :],
        all_atom_positions[..., n1_idx, :],
    )

    if all_atom_mask is not None:
        glycos_N_mask = torch.where(
            is_purine, 
            all_atom_mask[..., n9_idx], 
            all_atom_mask[..., n1_idx]
        )
        return glycos_N, glycos_N_mask
    else:
        return glycos_N


# 先写一个frame即bockbone的函数
def frame_and_literature_positions_to_atom3_pos(
    backb_to_global: Rigid,
    restype: torch.Tensor,
    lit_positions: torch.Tensor,
):
    """
    backb_to_global: [*, N_res] rigid
    restype: [*, N_res]
    lit_positions: [5, 3, 3], 在backbone frame下的局部坐标
    return: [*, N_res, 3, 3]
    """
    # [*, N_res, 3, 3]
    lit_positions = lit_positions[restype, ...]
    # [*, N_res, 1]
    backb_to_global = backb_to_global[..., None]
    # [*, N_res, 3, 3]
    pred_positions = backb_to_global.apply(lit_positions)

    return pred_positions