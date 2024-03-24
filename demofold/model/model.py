from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..utils.feats import glycos_N_fn
from ..utils.tensor_utils import (
    add,
    tensor_tree_map,
)
from .embedders import (
    InputEmbedder,
    SSEmbedder,
    RecyclingEmbedder,
)
from .evoformer import EvoformerStack
from .heads import AuxiliaryHeads, GeometryHeads
from .structure_module import StructureModule
from ..np import residue_constants as rc


class DemoFold(nn.Module):
    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super().__init__()

        self.globals = config.globals
        self.config = config.model

        # Main trunk + structure module
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"]
        )
        self.ss_embedder = SSEmbedder(
            **self.config["ss_embedder"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )
        if self.globals.is_e2e:
            self.structure_module = StructureModule(
                **self.config["structure_module"],
            )
        
        if self.globals.is_e2e:
            self.heads = AuxiliaryHeads(
                self.config["aux_heads"],
            )
        else:
            self.heads = GeometryHeads(
                self.config["geom_heads"]
            )
    
    def iteration(
        self, 
        feats: Dict[str, torch.Tensor], 
        prevs: List[torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        prevs: [m_1_prev, z_prev, x_prev]
        """
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)
        
        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        n_res = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        msa_mask = feats["msa_mask"]
        # 这里可以看出来和drfold不同，drfold是1代表pad，这里是0代表pad
        # 在这里构建pair_mask, 并没有像seq_mask和msa_mask一样
        # 在数据预处理时设置ss_mask, 考虑这些mask的作用不需要设置ss_mask
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

        # Initialize the MSA and pair representations
        # m: [*, N_seq, N_res, C_m]
        # z: [*, N_res, N_res, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
        )

        ss_emb = self.ss_embedder(
            feats["ss"],
        )
        # [*, N_res, N_res, C_z]
        z = add(z, ss_emb, inplace=inplace_safe)

        del ss_emb

        # Unpack the recycling embeddings. Removing them from the list allows 
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

        # Initialize the recycling embeddings, if needs be 
        # todo 这里做了偷懒修改
        if m_1_prev is None:
            # [*, N_res, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n_res, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N_res, N_res, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n_res, n_res, self.config.input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N_res, atom_type_num, 3]
            # 这里不同于openfold的全原子, 只使用三个原子
            x_prev = z.new_zeros(
                (*batch_dims, n_res, rc.bb_atom_type_num, 3),
                requires_grad=False,
            )
        
        # DRfold geometry
        if not self.globals.is_e2e:
            lit_positions = torch.tensor(
                rc.restype_atom4_bb_positions, 
                dtype=z.dtype,
                device=z.device,
                requires_grad=False,
            )
            x_prev = lit_positions[feats["restype"], ...]

        # [*, N_res, 3] predicted N coordinates
        glycos_N_x_prev = glycos_N_fn(
            feats["restype"], x_prev, None
        ).to(dtype=z.dtype)

        del x_prev

        # The recycling embedder is memory-intensive, so we offload first
        # 其实很简单，就是暂时把在内存密集型计算任务用不到的参数先移到cpu上，
        # 计算完之后再将参数移回gpu上
        if self.globals.offload_inference and inplace_safe:
            m = m.cpu()
            z = z.cpu()
        
        # m_1_prev_emb: [*, N_res, C_m]
        # z_prev_emb: [*, N_res, N_res, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            glycos_N_x_prev,
            inplace_safe=inplace_safe,
        )

        del glycos_N_x_prev

        if self.globals.offload_inference and inplace_safe:
            m = m.to(m_1_prev_emb.device)
            z = z.to(z_prev.device)

        # [*, N_seq, N_res, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N_res, N_res, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, N_seq, N_res, C_m]
        # z: [*, N_res, N_res, C_z]
        # s: [*, N_res, C_s]
        if self.globals.offload_inference:
            input_tensors = [m, z]
            del m, z
            m, z, s = self.evoformer._forward_offload(
                input_tensors,
                msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                _mask_trans=self.config._mask_trans,
            )

            del input_tensors
        else:
            m, z, s = self.evoformer(
                m,
                z,
                msa_mask=msa_mask.to(dtype=m.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        # openfold这里选择前n_seq个，是因为之前有个template模块，m在序列个数维度上有所增加
        # outputs["msa"] = m[..., :n_seq, :, :]
        outputs["msa"] = m
        outputs["pair"] = z
        outputs["single"] = s

        del z

        # e2e or geometry
        # Predict 3D structure
        if self.globals.is_e2e:
            outputs["sm"] = self.structure_module(
                outputs,
                feats["restype"],
                mask=feats["seq_mask"].to(dtype=s.dtype),
                inplace_safe=inplace_safe,
                _offload_inference=self.globals.offload_inference,
            )
            # [*, N_res, bb_atom_type_num, 3]
            outputs["final_atom_positions"] = outputs["sm"]["positions"][-1]
            outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N_res, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N_res, N_res, C_z]
        z_prev = outputs["pair"]

        # [*, N_res, bb_atom_type_num, 3]
        x_prev = None
        if self.globals.is_e2e:
            x_prev = outputs["final_atom_positions"]

        return outputs, m_1_prev, z_prev, x_prev



    def _disable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        self.evoformer.blocks_per_ckpt = (
            self.config.evoformer_stack.blocks_per_ckpt
        )



    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "restype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["restype"].shape[-1]
        # openfold有early stop机制，但貌似只用在multimer模式
        # early_stop = False
        num_recycles = 0
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats,
                    prevs
                )

                num_recycles += 1

                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev
                else:
                    break

        outputs["num_recycles"] = torch.tensor(num_recycles, device=feats["restype"].device)

        # Run auxiliary heads
        # todo
        outputs.update(self.heads(outputs))

        return outputs


# todo 最后检查一下outputs是否都用到了，可以del不必要的变量。再次检查两种模式
# todo 修改transform


