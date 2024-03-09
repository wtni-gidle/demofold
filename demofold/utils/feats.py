import torch
import torch.nn as nn

from .rigid_utils import Rigid


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