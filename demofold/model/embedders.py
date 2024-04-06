import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple

from ..utils.tensor_utils import expand_first_dims, add
from .primitives import Linear, LayerNorm


# 如果是soloseq，msa就是seq[..., None, :, :]
class InputEmbedder(nn.Module):
    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_m: int,
        c_z: int,
        max_len_seq: int,
        no_pos_bins_1d: int,
        pos_wsize_2d: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_m:
                MSA embedding dimension
            c_z:
                Pair embedding dimension
            max_len_seq:
                序列长度不能超过这个值
            no_pos_bins_1d:
                一维位置编码的维度
            pos_wsize_2d:
                二维位置编码的窗口大小
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim
        self.c_m = c_m
        self.c_z = c_z

        # RPE stuff
        self.max_seq_len = max_len_seq
        self.no_pos_bins_1d = no_pos_bins_1d
        self.pos_wsize_2d = pos_wsize_2d
        self.no_pos_bins_2d = 2 * self.pos_wsize_2d + 1

        self.linear_tf_m = Linear(self.tf_dim, self.c_m)
        self.linear_msa_m = Linear(self.msa_dim, self.c_m)
        self.linear_pos_1d = Linear(self.no_pos_bins_1d, self.c_m)
        self.linear_tf_z_i = Linear(self.tf_dim, self.c_z)
        self.linear_tf_z_j = Linear(self.tf_dim, self.c_z)
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
        tf: torch.Tensor,
        msa: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tf: [*, N_res, tf_dim]
        msa: [*, N_seq, N_res, msa_dim]

        return:
        msa_emb: [*, N_seq, N_res, c_m]
        pair_emb: [*, N_res, N_res, c_z]
        """
        n_res = tf.shape[-2]
        # device
        self.pos_1d = self.pos_1d.to(tf.device, tf.dtype)
        self.pos_2d = self.pos_2d.to(tf.device, tf.dtype)

        # [*, N_res, c_m]
        tf_emb_m = self.linear_tf_m(tf)
        # [*, N_seq, N_res, c_m]
        msa_emb_m = self.linear_msa_m(msa)
        # [N_res, c_m] -> [*, N_seq, N_res, c_m]
        pos_enc_1d = self.linear_pos_1d(self.pos_1d[:n_res])
        pos_enc_1d = expand_first_dims(pos_enc_1d, len(msa.shape) - 2)
        # [*, N_seq, N_res, c_m]
        msa_emb = msa_emb_m + tf_emb_m[..., None, :, :] + pos_enc_1d

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        # [N_res, N_res, c_z] -> [*, N_res, N_res, c_z]
        pair_emb = self.linear_pos_2d(self.pos_2d[:n_res, :n_res])
        pair_emb = expand_first_dims(pair_emb, len(tf.shape) - 2)
        # [*, N_res, N_res, c_z]
        pair_emb = add(
            pair_emb, 
            tf_emb_i[..., None, :], 
            inplace=inplace_safe
        )
        pair_emb = add(
            pair_emb, 
            tf_emb_j[..., None, :, :], 
            inplace=inplace_safe
        )

        return msa_emb, pair_emb



class SSEmbedder(nn.Module):
    def __init__(
        self, 
        ss_dim: int, 
        c_z: int,
        **kwargs,
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
        self.c_z = c_z
        self.ss_linear = Linear(self.ss_dim, self.c_z)

    def forward(self, ss: torch.Tensor) -> torch.Tensor:
        """
        ss: [*, N_res, N_res, ss_dim]

        return:
        pair_emb_ss: [*, N_res, N_res, c_z]
        """
        pair_emb_ss = self.ss_linear(ss)
        
        return pair_emb_ss



def fourier_encode_dist(
    x: torch.Tensor, 
    num_encodings: int = 20, 
    include_self: bool = True
):
    # from https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x



class RecyclingEmbedder(nn.Module):
    def __init__(
        self, 
        c_m: int, 
        c_z: int, 
        dis_encoding_dim: int
    ) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.dis_encoding_dim = dis_encoding_dim
        
        self.linear = Linear(self.dis_encoding_dim * 2 + 1, c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)
    
    def forward(
        self, 
        m: torch.Tensor, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        m: 
            [*, N_res, c_m] First row of the MSA embedding.
        z: 
            [*, N_res, N_res, c_z] Pair embedding.
        x: 
            [*, N_res, 3] predicted N coordinates

        return:
            m_update: [*, N_res, c_m]
            z_update: [*, N_res, N_res, c_z]
        """
        m_update = self.layer_norm_m(m)
        z_update = self.layer_norm_z(z)
        # dismap: [*, N_res, N_res]
        dismap = (x[..., None, :] - x[..., None, :, :]).norm(dim = -1)
        d_enc = fourier_encode_dist(dismap, self.dis_encoding_dim)

        z_update = add(
            z_update, 
            self.linear(d_enc), 
            inplace_safe
        )

        return m_update, z_update
