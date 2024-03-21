import sys

import torch
import torch.nn as nn
from typing import Optional, Tuple, Sequence, Union, Dict, Mapping

from .primitives import LayerNorm, Linear
from .ipa import InvariantPointAttention
from ..utils.rigid_utils import Rotation, Rigid
from ..utils.tensor_utils import dict_multimap
from ..utils.feats import frame_and_literature_positions_to_atom3_pos
from ..np import residue_constants as rc
from ..np.residue_constants import restype_atom3_bb_positions


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s: int):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super().__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        update = self.linear(s)

        return update



class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c: int):
        super().__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s
    


class StructureModuleTransition(nn.Module):
    def __init__(
        self, 
        c: int, 
        num_layers: int, 
        dropout_rate: float
    ):
        super().__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            s = l(s)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s
    

# todo 看一下要不要所有迭代的输出
class StructureModule(nn.Module):
    """
    修改自openfold。执行Algorithm 20 Structure module到第10行。更新backbone之后没有预测侧链
    输出更新后的seq和backbone。
    DRfold输出更新后的backbone以及backbone作用后的[L, 3, 3]三原子坐标
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_ipa: int,
        no_heads_ipa: int,
        no_qk_points: int,
        no_v_points: int,
        dropout_rate: float,
        no_blocks: int,
        no_transition_layers: int,
        trans_scale_factor: float,
        epsilon: float,
        inf: float,
    ):
        # trans_scale_factor不知道干嘛用的
        # Gram-Schmidt正交化体现在哪里也不知道
        # 它是给定三个点，得到rigid。感觉是从ground truth得到的
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = BackboneUpdate(self.c_s)

        # self.angle_resnet = AngleResnet(
        #     self.c_s,
        #     self.c_resnet,
        #     self.no_resnet_blocks,
        #     self.no_angles,
        #     self.epsilon,
        # )

    def forward(
        self,
        evoformer_output_dict: Mapping[str, torch.Tensor],
        restype: torch.Tensor,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ) -> Mapping[str, torch.Tensor]:
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            restype:
                [*, N_res] amino acid indices, 注意对应np.residue_constants
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
                frames: 
                    [no_blocks, *, N_res, 7]
                positions: C, P, N
                    [no_blocks, *, N_res, 3, 3]
                states:
                    [no_blocks, *, N_res, C_s]
                single:
                    [*, N_res, C_s]                
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N_res]
            mask = s.new_ones(s.shape[:-1])

        # [*, N_res, C_s]
        s = self.layer_norm_s(s)

        # [*, N_res, N_res, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict["pair"]) == 2
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N_res, C_s]
        # s_initial = seq
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(
            s.shape[:-1], 
            s.dtype, 
            s.device, 
            self.training,
            fmt="quat",
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N_res, C_s]
            s = s + self.ipa(
                s, 
                z, 
                rigids, 
                mask, 
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference, 
                _z_reference_list=z_reference_list
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N_res]
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(
                    rot_mats=rigids.get_rots().get_rot_mats(), 
                    quats=None
                ),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(
                self.trans_scale_factor
            )
            # [*, N_res, bb_atom_type_num, 3]
            pred_xyz = self.frame_and_literature_positions_to_atom4_pos(
                backb_to_global,
                restype
            )

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        if _offload_inference:
            evoformer_output_dict["pair"] = (
                evoformer_output_dict["pair"].to(s.device)
            )

        # 注意是torch.stack，会在开头添加一个维度
        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs
    
    def _init_residue_constants(
        self, 
        float_dtype: torch.dtype, 
        device: torch.device
    ):
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    rc.restype_atom4_bb_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def frame_and_literature_positions_to_atom4_pos(
        self, 
        backb_to_global: Rigid,
        restype: torch.Tensor
    ):
        """
        backb_to_global: [*, N_res] rigid
        restype: [*, N_res]
        """
        self._init_residue_constants(backb_to_global.dtype, backb_to_global.device)
        return frame_and_literature_positions_to_atom3_pos(
            backb_to_global,
            restype,
            self.lit_positions,
        )

