import torch
import torch.nn as nn

from .rigid_utils import Rigid

from ..np import residue_constants as rc



def atom_position_fn(
    atom: str, 
    all_atom_positions: torch.Tensor, 
    all_atom_masks: torch.Tensor
):
    """
    目前只需要用到C4', P, N1/N9三个原子的坐标, 因此先只用三个原子代替全原子
    all_atom_positions [..., N_res, 3, 3] 只含三个原子, 顺序为C4', P, N1/N9
    atom: "N" or "C" or "P"
    """
    atom_mapping = {atom: idx for idx, atom in enumerate(["C", "P", "N"])}
    atom_idx = atom_mapping[atom]
    atom_position = all_atom_positions[..., atom_idx, :]

    if all_atom_masks is not None:
        return atom_position, all_atom_masks[..., atom_idx]
    else:
        return atom_position


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