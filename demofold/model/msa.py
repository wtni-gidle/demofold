from functools import partial
import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from ..model.primitives import (
    Linear, 
    LayerNorm,
    Attention, 
    GlobalAttention, 
    _attention_chunked_trainable,
)
from ..utils.checkpointing import get_checkpoint_fn
from ..utils.chunk_utils import chunk_layer
from ..utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


class MSAAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        pair_bias: bool = False,
        c_z: Optional[int] = None,
        inf: float = 1e9,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            pair_bias:
                Whether to use pair embedding bias
            c_z:
                Pair embedding channel dimension. Ignored unless pair_bias
                is true
            inf:
                A large number to be used in computing the attention mask
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.pair_bias = pair_bias
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_m = LayerNorm(self.c_in)

        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(self.c_z)
            self.linear_z = Linear(
                self.c_z, self.no_heads, bias=False, init="normal"
            )
        
        self.mha = Attention(
            self.c_in, 
            self.c_in, 
            self.c_in, 
            self.c_hidden, 
            self.no_heads,
        )

    def _chunk(
        self, 
        m: torch.Tensor,
        biases: Optional[List[torch.Tensor]],
        chunk_size: int,
        use_memory_efficient_kernel: bool,
        use_deepspeed_evo_attention: bool,
        use_lma: bool,
        use_flash: bool,
        flash_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        def fn(m, biases, flash_mask):
            m = self.layer_norm_m(m)
            return self.mha(
                q_x=m, 
                kv_x=m, 
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=flash_mask,
            )

        inputs = {"m": m}
        if biases is not None:
            inputs["biases"] = biases
        else:
            fn = partial(fn, biases=None)
        if use_flash and flash_mask is not None:
            inputs["flash_mask"] = flash_mask
        else:
            fn = partial(fn, flash_mask=None)

        return chunk_layer(
            fn,
            inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def _prep_inputs(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        n_seq, n_res = m.shape[-3:-1]
        if mask is None:
            # [*, N_seq, N_res]
            mask = m.new_ones(
                m.shape[:-3] + (n_seq, n_res),
            )

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        if (self.pair_bias and 
            z is not None and                       # For the 
            self.layer_norm_z is not None and       # benefit of
            self.linear_z is not None               # TorchScript
        ):
            chunks = []

            for i in range(0, z.shape[-3], 256):
                z_chunk = z[..., i: i + 256, :, :]

                # [*, N_res, N_res, C_z]
                z_chunk = self.layer_norm_z(z_chunk)
            
                # [*, N_res, N_res, no_heads]
                z_chunk = self.linear_z(z_chunk)

                chunks.append(z_chunk)
            
            z = torch.cat(chunks, dim=-3)
            
            # [*, 1, no_heads, N_res, N_res]
            z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return m, mask_bias, z

    def _chunked_msa_attn(
            self,
        m: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_logits: int,
        checkpoint: bool,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """ 
        MSA attention with training-time chunking of the softmax computation.
        Saves memory in the extra MSA stack. Probably obviated by our fused 
        attention kernel, which is now used by default.
        """
        MSA_DIM = -4

        def _get_qkv(m, z):
            m, mask_bias, z = self._prep_inputs(
                m, z, mask, inplace_safe=inplace_safe
            )
            m = self.layer_norm_m(m)
            q, k, v = self.mha._prep_qkv(m, m)
            return m, q, k, v, mask_bias, z

        checkpoint_fn = get_checkpoint_fn()

        if torch.is_grad_enabled() and checkpoint:
            m, q, k, v, mask_bias, z = checkpoint_fn(_get_qkv, m, z)
        else:
            m, q, k, v, mask_bias, z = _get_qkv(m, z)
       
        o = _attention_chunked_trainable(
            query=q, 
            key=k, 
            value=v, 
            biases=[mask_bias, z], 
            chunk_size=chunk_logits, 
            chunk_dim=MSA_DIM,
            checkpoint=checkpoint,
        )

        if torch.is_grad_enabled() and checkpoint:
            # Storing an additional m here is far from ideal
            m = checkpoint_fn(self.mha._wrap_up, o, m)
        else:
            m = self.mha._wrap_up(o, m)

        return m

    def forward(
        self, 
        msa: torch.Tensor, 
        pair: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _chunk_logits: Optional[int] = None,
        _checkpoint_chunks: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        什么都不使用则使用手动实现的attention
        Args:
            msa:
                [*, N_seq, N_res, C_m] MSA embedding
            pair:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
                
        """
        if _chunk_logits is not None:
            return self._chunked_msa_attn(
                m=msa, z=pair, mask=mask, 
                chunk_logits=_chunk_logits, 
                checkpoint=_checkpoint_chunks,
                inplace_safe=inplace_safe,
            )
       
        if use_flash:
            assert pair is None
            biases = None
        else:    
            msa, mask_bias, pair = self._prep_inputs(
                msa, pair, mask, inplace_safe=inplace_safe
            )
    
            biases = [mask_bias]
            if pair is not None:
                biases.append(pair)

        if chunk_size is not None:
            msa = self._chunk(
                msa, 
                biases, 
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=mask,
            )
        else:
            msa = self.layer_norm_m(msa)
            msa = self.mha(
                q_x=msa, 
                kv_x=msa, 
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                flash_mask=mask,
            )

        return msa


class MSARowAttentionWithPairBias(MSAAttention):
    """
    Implements Algorithm 7.
    """

    def __init__(
        self, 
        c_m: int, 
        c_z: int, 
        c_hidden: int, 
        no_heads: int, 
        inf: float = 1e9,
    ):
        """
        Args:
            c_m:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super().__init__(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=True,
            c_z=c_z,
            inf=inf,
        )


class MSAColumnAttention(nn.Module):
    """
    Implements Algorithm 8.

    By rights, this should also be a subclass of MSAAttention. Alas,
    most inheritance isn't supported by TorchScript.
    """

    def __init__(
        self, 
        c_m: int, 
        c_hidden: int, 
        no_heads: int, 
        inf: float = 1e9
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super().__init__()
        
        self.c_m = c_m
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self._msa_att = MSAAttention(
            c_m,
            c_hidden,
            no_heads,
            pair_bias=False,
            c_z=None,
            inf=inf,
        )

    def forward(self, 
        msa: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            chunk_size:
                Size of chunks into which the inputs are split along their
                batch dimensions. A low value decreases memory overhead at the 
                cost of slower execution. Chunking is not performed by default.
        """ 
        # [*, N_res, N_seq, C_in]
        msa = msa.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        msa = self._msa_att(
            msa, 
            mask=mask, 
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
        )

        # [*, N_seq, N_res, C_in]
        msa = msa.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        return msa


class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """
    def __init__(
        self, 
        c_m: int, 
        n: int,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super().__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    def _chunk(self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
         return chunk_layer(
             self._transition,
             {"m": m, "mask": mask},
             chunk_size=chunk_size,
             no_batch_dims=len(m.shape[:-2]),
         )

    def forward(
        self,
        msa: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            msa:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = msa.new_ones(msa.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            msa = self._chunk(msa, mask, chunk_size)
        else:
            msa = self._transition(msa, mask)

        return msa
    
    