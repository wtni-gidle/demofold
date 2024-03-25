import ml_collections
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

# from ..np import residue_constants
from .rigid_utils import Rotation, Rigid
from .tensor_utils import (
    tree_map,
    masked_mean,
    permute_final_dims,
    tensor_tree_map
)
from .geometry_utils import calc_dihedral
from ..np import residue_constants as rc
from ..data.data_transforms import make_atom, glycos_N_fn

import logging

logger = logging.getLogger(__name__)


def softmax_cross_entropy(
    logits: torch.Tensor, 
    labels: torch.Tensor
) -> torch.Tensor:
    """
    logits不是概率值, 而是linear的输出
    """
    loss = -1 * torch.sum(
        labels * nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def distogram_loss(
    logits: torch.Tensor,
    atom_position: torch.Tensor,
    atom_mask: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    reduce: bool = True,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    <2: 1
    >40: 1
    2-40: 36
    logits: [*, N_res, N_res, no_bins]
    atom_position: [*, N_res, 3], 应该是Cβ的坐标
    atom_mask: [*, N_res]
    reduce: 是否要除以N_res ^ 2

    return: scalar
    """
    # [no_bins -1,]
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2

    # [*, N_res, N_res, 1] 点间距的平方
    dists = torch.sum(
        (atom_position[..., None, :] - atom_position[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )
    # [*, N_res, N_res] 值为0-37
    true_bins = torch.sum(dists > boundaries, dim=-1)
    # [*, N_res, N_res]
    errors = softmax_cross_entropy(
        logits,
        nn.functional.one_hot(true_bins, no_bins),
    )
    # [*, N_res, N_res]
    square_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    if reduce:
        mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean

# region: fape
def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    pair_mask: Optional[torch.Tensor] = None,
    l1_clamp_distance: Optional[float] = None,
    reduce: bool = True,    # DRfold原文只是求和，没有求均值
    eps: float = 1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            pair_mask: 应该是multimer的内容
                [*,  N_frames, N_pts] mask to use for
                separating intra- from inter-chain losses.
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # x_ij [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    # x_ij true
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )
    # [*, N_frames, N_pts]
    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    if pair_mask is not None:
        normed_error = normed_error * pair_mask
        normed_error = torch.sum(normed_error, dim=(-1, -2))

        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
        norm_factor = torch.sum(mask, dim=(-2, -1))

        normed_error = normed_error / (eps + norm_factor)
    else:
        # FP16-friendly averaging. Roughly equivalent to:
        #
        # norm_factor = (
        #     torch.sum(frames_mask, dim=-1) *
        #     torch.sum(positions_mask, dim=-1)
        # )
        # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
        #
        # ("roughly" because eps is necessarily duplicated in the latter)
        normed_error = torch.sum(normed_error, dim=-1)
        if reduce: 
            normed_error = (
                normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
            )
        normed_error = torch.sum(normed_error, dim=-1)
        if reduce:
            normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


#todo reduce, 以及点的坐标，还有batch
def backbone_loss(
    restype: torch.Tensor,
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    pred_atom_positions: torch.Tensor,
    pair_mask: Optional[torch.Tensor] = None,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    reduce: bool = True,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    """
    Parameters
    ----------
    restype: torch.Tensor
        [*, N_res]
    backbone_rigid_tensor : torch.Tensor
        [*, N_res, 4, 4]
    backbone_rigid_mask : torch.Tensor
        [*, N_res, ]
    traj : torch.Tensor
        [no_blocks, *, N_res, 4, 4]
    pair_mask : Optional[torch.Tensor], optional
        _description_, by default None
    use_clamped_fape : Optional[torch.Tensor], optional
        _description_, by default None
    clamp_distance : float, optional
        _description_, by default 10.0
    loss_unit_distance : float, optional
        _description_, by default 10.0
    reduce : bool, optional
        _description_, by default True
    eps : float, optional
        _description_, by default 1e-4

    Returns
    -------
    torch.Tensor
        scalar
    """
    ### need to check if the traj belongs to 4*4 matrix or a tensor_7
    if traj.shape[-1] == 7:
        pred_frame = Rigid.from_tensor_7(traj)
    elif traj.shape[-1] == 4:
        pred_frame = Rigid.from_tensor_4x4(traj)

    pred_frame = Rigid(
        Rotation(rot_mats=pred_frame.get_rots().get_rot_mats(), quats=None),
        pred_frame.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_frame = Rigid.from_tensor_4x4(backbone_rigid_tensor)
    # [...]
    batch_dims = gt_frame.shape[:-1]
    
    # ! 凑合着用吧...以后再改
    C4_prime, C4_prime_mask = make_atom("C4'", all_atom_positions, all_atom_mask)
    atom_P, atom_P_mask = make_atom("P", all_atom_positions, all_atom_mask)
    glycos_N, glycos_N_mask = glycos_N_fn(restype, all_atom_positions, all_atom_mask)
    # [..., N_res * 3, 3]
    bb_atom_pos_gt = torch.stack((C4_prime, atom_P, glycos_N), dim=-2).reshape(*batch_dims, -1, 3)
    # [..., N_res * 3]
    bb_atom_mask = torch.stack((C4_prime_mask, atom_P_mask, glycos_N_mask), dim=-1).reshape(*batch_dims, -1)

    C4_prime_pred = make_atom("C4'", pred_atom_positions)
    atom_P_pred = make_atom("P", pred_atom_positions)
    restype = torch.tile(restype[None], [pred_atom_positions.shape[0]] + [1] * len(restype.shape))
    glycos_N_pred = glycos_N_fn(restype, pred_atom_positions)
    # [no_blocks, ..., N_res * 3, 3]
    bb_atom_pos_pred = torch.stack((C4_prime_pred, atom_P_pred, glycos_N_pred), dim=-2).reshape(
        *pred_atom_positions.shape[:-3], -1, 3
    )
    fape_loss = compute_fape(
        pred_frame,
        gt_frame[None],
        backbone_rigid_mask[None],
        bb_atom_pos_pred,
        bb_atom_pos_gt[None],
        bb_atom_mask[None],
        pair_mask=pair_mask,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        reduce=reduce,
        eps=eps,
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_frame,
            gt_frame[None],
            backbone_rigid_mask[None],
            bb_atom_pos_pred,
            bb_atom_pos_gt[None],
            bb_atom_mask[None],
            pair_mask=pair_mask,
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            reduce=reduce,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss


def fape_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    traj = out["sm"]["frames"]
    bb_loss = backbone_loss(
            traj=traj,
            pred_atom_positions=out["sm"]["positions"],
            **{**batch, **config.backbone},
        )

    return bb_loss
# endregion

def masked_msa_loss(
    logits: torch.Tensor, 
    true_msa: torch.Tensor, 
    bert_mask: torch.Tensor, 
    num_classes: int, 
    eps: float = 1e-8, 
    **kwargs
) -> torch.Tensor:
    """
    Computes BERT-style masked MSA loss. Implements subsection 1.9.9.
    在DRfold中, msa只含一个序列, 即N_seq = 1

    Args:
        logits: [*, N_seq, N_res, 23] predicted residue distribution
        true_msa: [*, N_seq, N_res] true MSA
        bert_mask: [*, N_seq, N_res] MSA mask, 1代表一开始被mask
    Returns:
        Masked MSA loss scalar
    """
    # [*, N_seq, N_res]
    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_msa, num_classes=num_classes)
    )

    # FP16-friendly averaging. Equivalent to:
    # loss = (
    #     torch.sum(errors * bert_mask, dim=(-1, -2)) /
    #     (eps + torch.sum(bert_mask, dim=(-1, -2)))
    # )
    loss = errors * bert_mask
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = torch.mean(loss)

    return loss


class StructureLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def loss(
        self, 
        out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor], 
        _return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Rename previous forward() as loss()
        so that can be reused in the subclass 
        """
        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                atom_position=batch["glycos_N"],
                atom_mask=batch["glycos_N_mask"],
                **{**batch, **self.config.distogram},
            ),
            # todo
            "fape": lambda: fape_loss(
                out,
                batch,
                self.config.fape,
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                **{**batch, **self.config.masked_msa},
            ),
        }

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                # for k,v in batch.items():
                #    if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                #        logging.warning(f"{k}: is nan")
                # logging.warning(f"{loss_name}: {loss}")
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        # seq_len是完整长度，crop_len是裁剪长度。seq_len可能比crop_len小
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["restype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
    
    def forward(
        self, 
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor], 
        _return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not _return_breakdown:
            cum_loss = self.loss(out, batch, _return_breakdown)
            return cum_loss
        else:
            cum_loss, losses = self.loss(out, batch, _return_breakdown)
            return cum_loss, losses


# mode = mode.split("_")
# assert len(mode) != 8

# atoms = mode[::2]
# indicators = mode[1::2]
# assert set(indicators) == {"i", "j"}

# atom_positions = []
# for i, (indic, atom) in enumerate(zip(indicators, atoms)):
#     if indic == "i":
#         atom_positions[i] = atoms_map[atom][..., None, :]
#     else:
#         atom_positions[i] = atoms_map[atom][..., None, :, :]
# # [*, N_res, N_res]
# dihedrals = calc_dihedral(*atom_positions)



def PCCP_dihedral_loss(
    logits: torch.Tensor,
    atom_P: torch.Tensor,
    C4_prime: torch.Tensor,
    atom_P_mask: torch.Tensor,
    C4_prime_mask: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    max_dist: float,
    reduce: bool = True,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    <2: 1
    >40: 1
    2-40: 36
    logits: [*, N_res, N_res, no_bins]
    atom_P: [*, N_res, 3], 应该是Cβ的坐标
    atom_C: [*, N_res, 3], 应该是Cβ的坐标
    atom_mask: [*, N_res]
    min_bin = -180
    max_bin = 180
    no_bins = 36 + 1, 最后一维为no contact
    max_dist: C4’(i)-C4’(j) and N(i)-N(j), is larger than their corresponding
    maximum distance values M

    reduce: 是否要除以N_res ^ 2

    return: scalar
    """
    # [no_bins,]
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins,
        device=logits.device,
    )
    # [*, N_res, N_res, 1]
    dihedrals = calc_dihedral(
        atom_P[..., None, :],
        C4_prime[..., None, :],
        C4_prime[..., None, :, :],
        atom_P[..., None, :, :],
        degree=True
    )[..., None]

    # [*, N_res, N_res] 值为1-36
    true_bins = torch.sum(dihedrals > boundaries, dim=-1)
    # [*, N_res, N_res] 值为0-35
    true_bins -= 1

    # [*, N_res, N_res]
    C_dists = torch.sum(
        (C4_prime[..., None, :] - C4_prime[..., None, :, :]) ** 2,
        dim=-1,
    )
    contact = C_dists > (max_dist ** 2)
    true_bins[contact] = no_bins - 1

    # [*, N_res, N_res]
    errors = softmax_cross_entropy(
        logits,
        nn.functional.one_hot(true_bins, no_bins),
    )
    # [*, N_res, N_res]
    # todo
    atom_mask = atom_P_mask * C4_prime_mask
    square_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    if reduce:
        mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


def PNNP_dihedral_loss(
    logits: torch.Tensor,
    atom_P: torch.Tensor,
    glycos_N: torch.Tensor,
    atom_P_mask: torch.Tensor,
    glycos_N_mask: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    max_dist: float,
    reduce: bool = True,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    <2: 1
    >40: 1
    2-40: 36
    logits: [*, N_res, N_res, no_bins]
    atom_P: [*, N_res, 3], 应该是Cβ的坐标
    atom_C: [*, N_res, 3], 应该是Cβ的坐标
    atom_mask: [*, N_res]
    min_bin = -180
    max_bin = 180
    no_bins = 36 + 1, 最后一维为no contact
    max_dist: C4’(i)-C4’(j) and N(i)-N(j), is larger than their corresponding
    maximum distance values M

    reduce: 是否要除以N_res ^ 2

    return: scalar
    """
    # [no_bins,]
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins,
        device=logits.device,
    )
    # [*, N_res, N_res, 1]
    dihedrals = calc_dihedral(
        atom_P[..., None, :],
        glycos_N[..., None, :],
        glycos_N[..., None, :, :],
        atom_P[..., None, :, :],
        degree=True
    )[..., None]

    # [*, N_res, N_res] 值为1-36
    true_bins = torch.sum(dihedrals > boundaries, dim=-1)
    # [*, N_res, N_res] 值为0-35
    true_bins -= 1

    # [*, N_res, N_res]
    N_dists = torch.sum(
        (glycos_N[..., None, :] - glycos_N[..., None, :, :]) ** 2,
        dim=-1,
    )
    contact = N_dists > (max_dist ** 2)
    true_bins[contact] = no_bins - 1

    # [*, N_res, N_res]
    errors = softmax_cross_entropy(
        logits,
        nn.functional.one_hot(true_bins, no_bins),
    )
    # [*, N_res, N_res]
    atom_mask = atom_P_mask * glycos_N_mask
    square_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    if reduce:
        mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


def CNNC_dihedral_loss(
    logits: torch.Tensor,
    C4_prime: torch.Tensor,
    glycos_N: torch.Tensor,
    C4_prime_mask: torch.Tensor,
    glycos_N_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    max_dist: float,
    reduce: bool = True,
    eps: float = 1e-6,
    **kwargs
) -> torch.Tensor:
    """
    <2: 1
    >40: 1
    2-40: 36
    logits: [*, N_res, N_res, no_bins]
    atom_P: [*, N_res, 3], 应该是Cβ的坐标
    atom_C: [*, N_res, 3], 应该是Cβ的坐标
    atom_mask: [*, N_res]
    min_bin = -180
    max_bin = 180
    no_bins = 36 + 1, 最后一维为no contact
    max_dist: C4’(i)-C4’(j) and N(i)-N(j), is larger than their corresponding
    maximum distance values M

    reduce: 是否要除以N_res ^ 2

    return: scalar
    """
    # [no_bins,]
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins,
        device=logits.device,
    )
    # [*, N_res, N_res, 1]
    dihedrals = calc_dihedral(
        C4_prime[..., None, :],
        glycos_N[..., None, :],
        glycos_N[..., None, :, :],
        C4_prime[..., None, :, :],
        degree=True
    )[..., None]

    # [*, N_res, N_res] 值为1-36
    true_bins = torch.sum(dihedrals > boundaries, dim=-1)
    # [*, N_res, N_res] 值为0-35
    true_bins -= 1

    # [*, N_res, N_res]
    N_dists = torch.sum(
        (glycos_N[..., None, :] - glycos_N[..., None, :, :]) ** 2,
        dim=-1,
    )
    contact = N_dists > (max_dist ** 2)
    true_bins[contact] = no_bins - 1

    # [*, N_res, N_res]
    errors = softmax_cross_entropy(
        logits,
        nn.functional.one_hot(true_bins, no_bins),
    )
    # [*, N_res, N_res]
    atom_mask = C4_prime_mask * glycos_N_mask
    square_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    if reduce:
        mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


class GeometryLoss(nn.Module):
    """C4'怎么命名需要解决"""
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def loss(
        self, 
        out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor], 
        _return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Rename previous forward() as loss()
        so that can be reused in the subclass 
        """
        loss_fns = {
            "PP": lambda: distogram_loss(
                logits=out["PP_logits"],
                atom_position=batch["atom_P"],
                atom_mask=batch["atom_P_mask"],
                **{**batch, **self.config.PP},
            ),
            "CC": lambda: distogram_loss(
                logits=out["CC_logits"],
                atom_position=batch["C4_prime"],
                atom_mask=batch["C4_prime_mask"],
                **{**batch, **self.config.CC},
            ),
            "NN": lambda: distogram_loss(
                logits=out["NN_logits"],
                atom_position=batch["glycos_N"],
                atom_mask=batch["glycos_N_mask"],
                **{**batch, **self.config.NN},
            ),
            "PCCP": lambda: PCCP_dihedral_loss(
                logits=out["PCCP_logits"],
                **{**batch, **self.config.PCCP},
            ),
            "PNNP": lambda: PNNP_dihedral_loss(
                logits=out["PNNP_logits"],
                **{**batch, **self.config.PNNP},
            ),
            "CNNC": lambda: CNNC_dihedral_loss(
                logits=out["CNNC_logits"],
                **{**batch, **self.config.CNNC},
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                **{**batch, **self.config.masked_msa},
            ),
        }

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                # for k,v in batch.items():
                #    if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                #        logging.warning(f"{k}: is nan")
                # logging.warning(f"{loss_name}: {loss}")
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        # seq_len是完整长度，crop_len是裁剪长度。seq_len可能比crop_len小
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["restype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
    
    def forward(
        self, 
        out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor], 
        _return_breakdown: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not _return_breakdown:
            cum_loss = self.loss(out, batch, _return_breakdown)
            return cum_loss
        else:
            cum_loss, losses = self.loss(out, batch, _return_breakdown)
            return cum_loss, losses
