import sys
import torch
import torch.nn as nn
from typing import Tuple, Sequence, Optional
from functools import partial
from abc import ABC, abstractmethod

from .primitives import Linear, LayerNorm
from .dropout import DropoutRowwise, DropoutColumnwise
from .msa import (
    MSARowAttentionWithPairBias,
    MSAColumnAttention,
    # MSAColumnGlobalAttention,
    MSATransition,
)
from .outer_product_mean import OuterProductMean
from .pair_transition import PairTransition
from .triangular_attention import TriangleAttention
from .triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing,
)
from ..utils.checkpointing import checkpoint_blocks, get_checkpoint_fn
from ..utils.chunk_utils import chunk_layer, ChunkSizeTuner
from ..utils.tensor_utils import add


class PairStack(nn.Module):
    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        fuse_projection_weights: bool,
        inf: float,
    ):
        super().__init__()

        if fuse_projection_weights:
            self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = FusedTriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        pair_trans_mask = pair_mask if _mask_trans else None

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        # region: tri_mul_out
        tmu_update = self.tri_mul_out(
            pair,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if not inplace_safe:
            pair = pair + self.ps_dropout_row_layer(tmu_update)
        else:
            pair = tmu_update

        del tmu_update
        # endregion
        # region: tri_mul_in
        tmu_update = self.tri_mul_in(
            pair,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if not inplace_safe:
            pair = pair + self.ps_dropout_row_layer(tmu_update)
        else:
            pair = tmu_update

        del tmu_update
        # endregion
        # region: tri_att_start
        pair = add(
            pair,
            self.ps_dropout_row_layer(
                self.tri_att_start(
                    pair,
                    mask=pair_mask,
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )
        # endregion
        # region: tri_att_end
        pair = pair.transpose(-2, -3)
        if inplace_safe:
            pair = pair.contiguous()

        pair = add(
            pair,
            self.ps_dropout_row_layer(
                self.tri_att_end(
                    pair,
                    mask=pair_mask.transpose(-1, -2),
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )

        pair = pair.transpose(-2, -3)
        if inplace_safe:
            pair = pair.contiguous()
        # endregion
        # region: pair_transition
        pair = add(
            pair,
            self.pair_transition(
                pair, mask=pair_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )
        # endregion

        return pair


class MSAPairBlock(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
    ):
        super().__init__()

        self.opm_first = opm_first

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.msa_transition = MSATransition(
            c_m=c_m,
            n=transition_n,
        )

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
        )

    def _compute_opm(
        self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        _offload_inference: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        msa, pair = input_tensors

        if _offload_inference and inplace_safe:
            # msa: GPU, pair: CPU
            del msa, pair
            assert (sys.getrefcount(input_tensors[1]) == 2)
            input_tensors[1] = input_tensors[1].cpu()
            msa, pair = input_tensors

        opm = self.outer_product_mean(
            msa, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
        )

        if _offload_inference and inplace_safe:
            # msa: GPU, pair: GPU
            del msa, pair
            assert (sys.getrefcount(input_tensors[0]) == 2)
            input_tensors[1] = input_tensors[1].to(opm.device)
            msa, pair = input_tensors

        pair = add(pair, opm, inplace=inplace_safe)
        del opm

        return msa, pair

    @abstractmethod
    def forward(
        self,
        msa: Optional[torch.Tensor],
        pair: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class EvoformerBlock(MSAPairBlock):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        no_column_attention: bool,
        opm_first: bool,
        fuse_projection_weights: bool,
        inf: float,
    ):
        super().__init__(
            c_m=c_m,
            c_z=c_z,
            c_hidden_msa_att=c_hidden_msa_att,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            msa_dropout=msa_dropout,
            pair_dropout=pair_dropout,
            opm_first=opm_first,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf
        )

        # Specifically, seqemb mode does not use column attention
        self.no_column_attention = no_column_attention

        if not self.no_column_attention:
            self.msa_att_col = MSAColumnAttention(
                c_m,
                c_hidden_msa_att,
                no_heads_msa,
                inf=inf,
            )

    def forward(
        self,
        msa: Optional[torch.Tensor],
        pair: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        msa_trans_mask = msa_mask if _mask_trans else None

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        if _offload_inference and inplace_safe:
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [msa, pair]

        msa, pair = input_tensors

        if self.opm_first:
            del msa, pair

            msa, pair = self._compute_opm(
                input_tensors=input_tensors,
                msa_mask=msa_mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
            )

        msa = add(
            msa,
            self.msa_dropout_layer(
                self.msa_att_row(
                    msa,
                    pair=pair,
                    mask=msa_mask,
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                )
            ),
            inplace=inplace_safe,
        )

        if _offload_inference and inplace_safe:
            # msa: GPU, pair: CPU
            del msa, pair
            assert (sys.getrefcount(input_tensors[1]) == 2)
            input_tensors[1] = input_tensors[1].cpu()
            torch.cuda.empty_cache()
            msa, pair = input_tensors

        # Specifically, column attention is not used in seqemb mode.
        if not self.no_column_attention:
            msa = add(
                msa,
                self.msa_att_col(
                    msa,
                    mask=msa_mask,
                    chunk_size=chunk_size,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    use_flash=use_flash,
                ),
                inplace=inplace_safe,
            )

        msa = add(
            msa,
            self.msa_transition(
                msa, mask=msa_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        if not self.opm_first:
            if not inplace_safe:
                input_tensors = [msa, pair]

            del msa, pair

            msa, pair = self._compute_opm(
                input_tensors=input_tensors,
                msa_mask=msa_mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference
            )

        if _offload_inference and inplace_safe:
            # msa: CPU, pair: GPU
            del msa, pair
            assert (sys.getrefcount(input_tensors[0]) == 2)
            device = input_tensors[0].device
            input_tensors[0] = input_tensors[0].cpu()
            input_tensors[1] = input_tensors[1].to(device)
            msa, pair = input_tensors

        if not inplace_safe:
            input_tensors = [msa, pair]

        del msa, pair

        pair = self.pair_stack(
            pair=input_tensors[1],
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
        )

        if _offload_inference and inplace_safe:
            # msa: GPU, pair: GPU
            device = pair.device
            assert (sys.getrefcount(input_tensors[0]) == 2)
            input_tensors[0] = input_tensors[0].to(device)
            msa, _ = input_tensors
        else:
            msa = input_tensors[0]

        return msa, pair


class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        no_column_attention: bool,
        opm_first: bool,
        fuse_projection_weights: bool,
        blocks_per_ckpt: int,
        inf: float,
        eps: float,
        clear_cache_between_blocks: bool = False, 
        tune_chunk_size: bool = False,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            no_column_attention:
                When True, doesn't use column attention. Required for running
                sequence embedding mode
            opm_first:
                When True, Outer Product Mean is performed at the beginning of
                the Evoformer block instead of after the MSA Stack.
                Used in Multimer pipeline.
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super().__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = EvoformerBlock(
                c_m=c_m,
                c_z=c_z,
                c_hidden_msa_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_msa=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                no_column_attention=no_column_attention,
                opm_first=opm_first,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
                eps=eps,
            )
            self.blocks.append(block)

        self.linear = Linear(c_m, c_s)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def _prep_blocks(
        self, 
        m: torch.Tensor, 
        z: torch.Tensor, 
        chunk_size: int,
        use_deepspeed_evo_attention: bool,
        use_lma: bool,
        use_flash: bool,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_flash=use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if self.clear_cache_between_blocks:
            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                # We don't want to write in-place during chunk tuning runs
                args=(m.clone(), z.clone(),),
                min_chunk_size=chunk_size,
            )
            blocks = [
                partial(b, 
                    chunk_size=tuned_chunk_size,
                    # A temporary measure to address torch's occasional
                    # inability to allocate large tensors
                    _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4),
                ) for b in blocks
            ]

        return blocks

    def _forward_offload(
        self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert not (self.training or torch.is_grad_enabled())
        blocks = self._prep_blocks(
            # We are very careful not to create references to these tensors in
            # this function
            m=input_tensors[0],
            z=input_tensors[1],
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=True,
            _mask_trans=_mask_trans,
        )

        for b in blocks:
            msa, pair = b(
                None, 
                None, 
                _offload_inference=True,
                _offloadable_inputs=input_tensors,
            )
            input_tensors[0] = msa
            input_tensors[1] = pair
            del msa, pair
        
        msa, pair = input_tensors
        
        seq = self.linear(msa[..., 0, :, :])
        
        return msa, pair, seq

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            msa:
                [*, N_seq, N_res, C_m] MSA embedding
            pair:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
            use_flash: 
                Whether to use FlashAttention where possible. Mutually 
                exclusive with use_lma and use_deepspeed_evo_attention.
        Returns:
            msa:
                [*, N_seq, N_res, C_m] MSA embedding
            pair:
                [*, N_res, N_res, C_z] pair embedding
            seq:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """ 
        blocks = self._prep_blocks(
            m=msa,
            z=pair,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        
        msa, pair = checkpoint_blocks(
            blocks,
            args=(msa, pair),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        seq = self.linear(msa[..., 0, :, :])

        return msa, pair, seq

