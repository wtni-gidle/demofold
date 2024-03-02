import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional

from ..utils.tensor_utils import expand_first_dims
from .primitives import Linear, LayerNorm



class InputEmbedder(nn.Module):
    def __init__(
        self,
        seq_dim: int, 
        msa_dim: int,
        c_m: int,
        c_z: int,
        max_len_seq: int,
        no_pos_bins_1d: int,
        pos_wsize_2d: int,
    ):
        """
        Args:
            seq_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_m:
                MSA embedding dimension
            c_z:
                Pair embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super().__init__()
        self.seq_dim = seq_dim
        self.msa_dim = msa_dim
        self.c_m = c_m
        self.c_z = c_z

        # RPE stuff
        self.max_seq_len = max_len_seq
        self.no_pos_bins_1d = no_pos_bins_1d
        self.pos_wsize_2d = pos_wsize_2d
        self.no_pos_bins_2d = 2 * self.pos_wsize_2d + 1

        self.linear_seq_m = Linear(self.seq_dim, self.c_m)
        self.linear_msa_m = Linear(self.msa_dim, self.c_m)
        self.linear_pos_1d = Linear(self.no_pos_bins_1d, self.c_m)
        self.linear_seq_z_i = Linear(self.seq_dim, self.c_z)
        self.linear_seq_z_j = Linear(self.seq_dim, self.c_z)
        self.linear_pos_2d = Linear(self.no_pos_bins_2d, self.c_z)

        self.pos_1d = self.compute_pos_1d()
        self.pos_2d = self.compute_pos_2d()

    def compute_pos_1d(self) -> torch.Tensor:
        """
        return: [max_seq_len, no_pos_bins_1d]的二进制矩阵。即每一个位置用二进制表示
        e.g. 
        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        """
        pos = torch.arange(self.max_seq_len)
        rel_pos = ((pos[:,None] & (1 << torch.arange(self.no_pos_bins_1d)))) > 0
        
        return rel_pos.float()

    def compute_pos_2d(self) -> torch.Tensor:
        """
        return: [max_seq_len, max_seq_len, no_pos_bins_2d]
        """
        pos = torch.arange(self.max_seq_len)
        rel_pos = (pos[None, :] - pos[:, None]).clamp(-self.pos_wsize_2d, self.pos_wsize_2d)
        rel_pos_enc = F.one_hot(rel_pos + self.pos_wsize_2d, self.no_pos_bins_2d)
        
        return rel_pos_enc.float()

    def forward(
        self,
        seq: torch.Tensor,
        msa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: [*, N_res, seq_dim]
        msa: [*, N_seq, N_res, msa_dim]

        return:
        msa_emb: [*, N_seq, N_res, c_m]
        pair_emb: [*, N_res, N_res, c_z]
        """
        n_res = seq.shape[-2]
        # device
        self.pos_1d = self.pos_1d.to(seq.device)
        self.pos_2d = self.pos_2d.to(seq.device)
        # [*, N_res, c_m]
        seq_emb_m = self.linear_seq_m(seq)
        # [*, N_seq, N_res, c_m]
        msa_emb_m = self.linear_msa_m(msa)
        # [N_res, c_m] -> [*, N_seq, N_res, c_m]
        pos_enc_1d = self.linear_pos_1d(self.pos_1d[:n_res])
        pos_enc_1d = expand_first_dims(pos_enc_1d, len(msa.shape) - 2)
        # [*, N_seq, N_res, c_m]
        msa_emb = msa_emb_m + seq_emb_m[..., None, :, :] + pos_enc_1d

        # [*, N_res, c_z]
        seq_emb_i = self.linear_seq_z_i(seq)
        seq_emb_j = self.linear_seq_z_j(seq)
        # [N_res, N_res, c_z] -> [*, N_res, N_res, c_z]
        pos_enc_2d = self.linear_pos_2d(self.pos_2d[:n_res, :n_res])
        pos_enc_2d = expand_first_dims(pos_enc_2d, len(seq.shape) - 2)
        # [*, N_res, N_res, c_z]
        pair_emb = seq_emb_i[..., None, :, :] + seq_emb_j[..., None, :] + pos_enc_2d

        return msa_emb, pair_emb


class SSEmbedder(nn.Module):
    def __init__(
        self, 
        ss_dim: int, 
        c_z: int,
    ):
        """
        Args:
            ss_dim:
                Final dimension of the ss features
            c_z:
                Pair embedding dimension
        """
        super().__init__()
        self.ss_dim = ss_dim
        self.ss_linear = Linear(ss_dim, c_z)

    def forward(self, ss: torch.Tensor) -> torch.Tensor:
        """
        ss: [*, N_res, N_res, ss_dim]

        return:
        pair_emb_ss: [*, N_res, N_res, c_z]
        """
        pair_emb_ss = self.ss_linear(ss)
        
        return pair_emb_ss


