import copy
from functools import partial
import json
import logging
import os
import pickle
from typing import Optional, Sequence, Any, Union

import ml_collections as mlc
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from . import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
)
from ..utils.tensor_utils import dict_multimap
from ..utils.tensor_utils import (
    tensor_tree_map,
)

class DemoFoldSingleDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        ss_dir: str,
        config: mlc.ConfigDict,
        mode: str = "train",
        _output_raw: bool = False,
        _structure_index: Optional[Any] = None,
        filter_path: Optional[str] = None
    ):
        """
        alignment_dir是放置所有file_id的alignment的dir, 即alignment_dir/file_id_chain_id/
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                alignment_dir:
                    A path to a directory containing only data in the format 
                    output by an AlignmentRunner 
                    (defined in openfold.features.alignment_runner).
                    I.e. a directory of directories named {PDB_ID}_{CHAIN_ID}
                    or simply {PDB_ID}, each containing .a3m, .sto, and .hhr
                    files.
                template_mmcif_dir:
                    Path to a directory containing template mmCIF files.
                config:
                    A dataset config object. See openfold.config
                chain_data_cache_path:
                    Path to cache of data_dir generated by
                    scripts/generate_chain_data_cache.py
                kalign_binary_path:
                    Path to kalign binary.
                max_template_hits:
                    An upper bound on how many templates are considered. During
                    training, the templates ultimately used are subsampled
                    from this total quantity.
                template_release_dates_cache_path:
                    Path to the output of scripts/generate_mmcif_cache.
                obsolete_pdbs_file_path:
                    Path to the file containing replacements for obsolete PDBs.
                shuffle_top_k_prefiltered:
                    Whether to uniformly shuffle the top k template hits before
                    parsing max_template_hits of them. Can be used to
                    approximate DeepMind's training-time template subsampling
                    scheme much more performantly.
                treat_pdb_as_distillation:
                    Whether to assume that .pdb files in the data_dir are from
                    the self-distillation set (and should be subjected to
                    special distillation set preprocessing steps).
                mode:
                    "train", "val", or "predict"
        """
        super().__init__()
        self.data_dir = data_dir
        self.ss_dir = ss_dir
        self.config = config
        
        self.mode = mode
        valid_modes = ["train", "eval", "predict"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')
        
        self._output_raw = _output_raw
        # todo这个还不知道有没有用
        self._structure_index = _structure_index

        self.supported_exts = [".cif"]

        self._chain_ids = list(os.listdir(ss_dir))
        if filter_path is not None:
            with open(filter_path, "r") as f:
                chains_to_include = set([l.strip() for l in f.readlines()])

            self._chain_ids = [
                c for c in self._chain_ids if c in chains_to_include
            ]

        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }

        self.data_pipeline = data_pipeline.DataPipeline()

        if not self._output_raw:
            self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

    def _parse_mmcif(
        self, 
        path: str, 
        file_id: str, 
        chain_id: str, 
        ss_dir: str
    ):
        with open(path, 'r') as f:
            mmcif_string = f.read()

        mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string
        )

        mmcif_object = mmcif_object.mmcif_object

        data = self.data_pipeline.process_mmcif(
            mmcif=mmcif_object,
            ss_dir=ss_dir,
            chain_id=chain_id,
        )

        return data

    def chain_id_to_idx(self, chain_id):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx: int):
        return self._chain_ids[idx]

    def __getitem__(self, idx):
        """
        本质上是获取文件路径后传入data_pipeline分
        mmcif和fasta分析获取FeatureDict"""
        # {file_id}_{chain_id}
        name = self.idx_to_chain_id(idx)
        ss_dir = os.path.join(self.ss_dir, name)

        if self.mode == 'train' or self.mode == 'eval':
            spl = name.rsplit('_', 1)
            if len(spl) == 2:
                file_id, chain_id = spl
            else:
                file_id, = spl
                chain_id = None

            # mmcif_dir/4enc
            path = os.path.join(self.data_dir, file_id)
            # todo 这个structure_index是啥
            if self._structure_index is not None:
                structure_index_entry = self._structure_index[name]
                assert (len(structure_index_entry["files"]) == 1)
                filename, _, _ = structure_index_entry["files"][0]
                ext = os.path.splitext(filename)[1]
            else:
                ext = None
                for e in self.supported_exts:
                    if os.path.exists(path + e):
                        ext = e
                        break

                if ext is None:
                    raise ValueError("Invalid file type")

            # mmcif_dir/4enc.cif
            path += ext
            if ext == ".cif":
                data = self._parse_mmcif(
                    path, file_id, chain_id, ss_dir, 
                )
        else:
            # 如果是predict模式, path不用和data_dir合并
            path = os.path.join(name, name + ".fasta")
            data = self.data_pipeline.process_fasta(
                fasta_path=path,
                ss_dir=ss_dir,
            )

        if self._output_raw:
            return data

        feats = self.feature_pipeline.process_features(
            data, self.mode
        )

        feats["batch_idx"] = torch.tensor(
            [idx for _ in range(feats["restype"].shape[-1])],
            dtype=torch.int64,
            device=feats["restype"].device)

        return feats

    def __len__(self):
        return len(self._chain_ids)


"""
raw_data: Dict[str, np.ndarray]
mmcif:
    restype: [N_res, 4+1]
    msa: shape of [num_alignments=1, N_res] 
        每个值是ACGUX-到012345, 但是我们是单序列, 不可能有5
    ss: [N_res, N_res, 4]
    num_alignments: [num_alignments=1, ..., num_alignments]
    seq_length: [num_res] * num_res
    between_segment_residues: [N_res,]全0

    domain_name: {file_id}_{chain_id}
    sequence: ACGUCG

    all_atom_positions: [N_res, bb_atom_type_num, 3]原始坐标
    all_atom_mask: [N_res, bb_atom_type_num]
    resolution

    release_date
fasta:
    restype: [N_res, 4+1]
    msa: shape of [num_alignments=1, N_res] 
        每个值是ACGUX-到012345, 但是我们是单序列, 不可能有5
    ss: [N_res, N_res, 4]
    num_alignments: [num_alignments=1, ..., num_alignments]
    seq_length: [num_res] * num_res
    between_segment_residues: [N_res,]
    
    domain_name: {file_id}_{chain_id}
    sequence: ACGUCG
"""
"""
mmcif: N_res为实际的crop_size, 
    Features (without the recycling dimension): 不同的recycle之间的唯一区别是bert msa不一样, 
    但是每次都有对应的gt
    batch_idx: [idx] 样本序号
    restype: [N_res]ACGUX
    msa: shape of [num_alignments=1, N_res] 
        被mask之后的msa, 每个元素是0-6, 4+"X"+"-"+[mask]
    true_msa: [1, N_res] 原来的msa
        每个值是ACGUX-到012345, 但是我们是单序列, 不可能有5
    bert_mask: [1, N_res] 1为被replace
    msa_feat: [1, N_res, 7]
    target_feat: [N_res, 6]第一个是between_segment_residues
    ss: [N_res, N_res, 4]
    # num_alignments: [1]
    seq_length: [N_res]
    # between_segment_residues: [N_res,]全0
    seq_mask: [N_res]用于记录pad(用0补全), 所有样本需要相同形状, 补全到crop_size
    msa_mask: [N_seq, N_res]用于记录pad(用0补全), 所有样本需要相同形状, 补全到crop_size

    all_atom_positions: [N_res, bb_atom_type_num, 3]原始坐标
    all_atom_mask: [N_res, bb_atom_type_num]
    resolution
    glycos_N: [N_res, 3]
    glycos_N_mask: [N_res]
    atom_P
    C4_prime
    backbone_rigid_tensor: [N_res, 4, 4]
    backbone_rigid_mask: [N_res]
    use_clamped_fape: 是否使用clamped_fape
    gt_features: None
    no_recycling_iters
fasta:
    batch_idx: [idx] 样本序号
    restype: [N_res]ACGUX
    msa: shape of [num_alignments=1, N_res] 
        被mask之后的msa, 每个元素是0-6, 4+"X"+"-"+[mask]
    true_msa: [1, N_res] 原来的msa
        每个值是ACGUX-到012345, 但是我们是单序列, 不可能有5
    bert_mask: [1, N_res] 1为被replace
    msa_feat: [N_res, 7]
    target_feat: [N_res, 6]第一个是between_segment_residues
    ss: [N_res, N_res, 4]
    # num_alignments: [1]
    seq_length: [num_res]
    # between_segment_residues: [N_res,]全0
    seq_mask: [N_res]
    msa_mask: [N_seq, N_res]
    use_clamped_fape: 是否使用clamped_fape, fasta这个值为0
    gt_features: None
    no_recycling_iters
"""


class DemoFoldDataset(Dataset):
    """
        Implements the stochastic filters applied during AlphaFold's training.
        Because samples are selected from constituent datasets randomly, the
        length of an OpenFoldFilteredDataset is arbitrary. Samples are selected
        and filtered once at initialization.
    """

    def __init__(
        self,
        datasets: Sequence[DemoFoldSingleDataset],
        probabilities: Sequence[float],
        epoch_len: int,
        generator: torch.Generator = None,
        _roll_at_init: bool = True,
    ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.epoch_len = epoch_len
        self.generator = generator

        self._samples = [self.looped_samples(i) for i in range(len(self.datasets))]
        if _roll_at_init:
            self.reroll()

    def looped_shuffled_dataset_idx(self, dataset_len):
        while True:
            # Uniformly shuffle each dataset's indices
            weights = [1. for _ in range(dataset_len)]
            shuf = torch.multinomial(
                torch.tensor(weights),
                num_samples=dataset_len,
                replacement=False,
                generator=self.generator,
            )
            for idx in shuf:
                yield idx

    def looped_samples(self, dataset_idx):
        max_cache_len = int(self.epoch_len * self.probabilities[dataset_idx])
        dataset = self.datasets[dataset_idx]
        idx_iter = self.looped_shuffled_dataset_idx(len(dataset))
        while True:
            weights = []
            idx = []
            for _ in range(max_cache_len):
                candidate_idx = next(idx_iter)

                weights.append([0.0, 1.0])
                idx.append(candidate_idx)

            samples = torch.multinomial(
                torch.tensor(weights),
                num_samples=1,
                generator=self.generator,
            )
            samples = samples.squeeze()

            cache = [i for i, s in zip(idx, samples) if s]

            for datapoint_idx in cache:
                yield datapoint_idx

    def __getitem__(self, idx):
        # self.datapoints每个元素是(dataset_idx, datapoint_idx),
        # 即第几个数据集的第几个样本
        dataset_idx, datapoint_idx = self.datapoints[idx]
        return self.datasets[dataset_idx][datapoint_idx]

    def __len__(self):
        return self.epoch_len

    def reroll(self):
        dataset_choices = torch.multinomial(
            torch.tensor(self.probabilities),
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )
        self.datapoints = []
        for dataset_idx in dataset_choices:
            samples = self._samples[dataset_idx]
            datapoint_idx = next(samples)
            self.datapoints.append((dataset_idx, datapoint_idx))


class DemoFoldBatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots)


class DemoFoldDataLoader(DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters

        if stage_cfg.uniform_recycling:
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.

        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs]

        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        """这玩意就是为了实现recycle次数的随机性"""
        gt_features = batch.pop('gt_features', None)
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1,  # 1 per row
            replacement=True,
            generator=self.generator
        )

        aatype = batch["restype"]
        batch_dims = aatype.shape[:-2]
        recycling_dim = aatype.shape[-1]
        no_recycling = recycling_dim
        for i, key in enumerate(self.prop_keys):
            sample = int(samples[i][0])
            sample_tensor = torch.tensor(
                sample,
                device=aatype.device,
                requires_grad=False
            )
            orig_shape = sample_tensor.shape
            sample_tensor = sample_tensor.view(
                (1,) * len(batch_dims) + sample_tensor.shape + (1,)
            )
            sample_tensor = sample_tensor.expand(
                batch_dims + orig_shape + (recycling_dim,)
            )
            batch[key] = sample_tensor

            if key == "no_recycling_iters":
                no_recycling = sample

        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)
        batch['gt_features'] = gt_features

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


class DemoFoldDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: mlc.ConfigDict,
        train_data_dir: Optional[str] = None,
        train_ss_dir: Optional[str] = None,
        val_data_dir: Optional[str] = None,
        val_ss_dir: Optional[str] = None,
        predict_data_dir: Optional[str] = None,
        predict_ss_dir: Optional[str] = None,
        batch_seed: Optional[int] = None,
        train_epoch_len: int = 50000,
        train_filter_path: str = None,
        val_filter_path: str = None,
        **kwargs
    ):
        super().__init__()

        self.config = config
        self.train_data_dir = train_data_dir
        self.train_ss_dir = train_ss_dir
        self.val_data_dir = val_data_dir
        self.val_ss_dir = val_ss_dir
        self.predict_data_dir = predict_data_dir
        self.predict_ss_dir = predict_ss_dir
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len
        self.train_filter_path = train_filter_path
        self.val_filter_path = val_filter_path

        if self.train_data_dir is None and self.predict_data_dir is None:
            raise ValueError(
                'At least one of train_data_dir or predict_data_dir must be '
                'specified'
            )

        self.training_mode = self.train_data_dir is not None

        if self.training_mode and train_ss_dir is None:
            raise ValueError(
                'In training mode, train_alignment_dir must be specified'
            )
        elif not self.training_mode and predict_ss_dir is None:
            raise ValueError(
                'In inference mode, predict_alignment_dir must be specified'
            )
        elif val_data_dir is not None and val_ss_dir is None:
            raise ValueError(
                'If val_data_dir is specified, val_alignment_dir must '
                'be specified as well'
            )

    def setup(self):
        # Most of the arguments are the same for the three datasets 
        dataset_gen = partial(
            DemoFoldSingleDataset,
            config=self.config,
        )

        if self.training_mode:
            train_dataset = dataset_gen(
                data_dir=self.train_data_dir,
                ss_dir=self.train_ss_dir,
                mode="train",
                filter_path=self.train_filter_path
            )

            datasets = [train_dataset]
            probabilities = [1.]

            generator = None
            if self.batch_seed is not None:
                generator = torch.Generator()
                generator = generator.manual_seed(self.batch_seed + 1)

            self.train_dataset = DemoFoldDataset(
                datasets=datasets,
                probabilities=probabilities,
                epoch_len=self.train_epoch_len,
                generator=generator,
                _roll_at_init=False,
            )

            if self.val_data_dir is not None:
                self.eval_dataset = dataset_gen(
                    data_dir=self.val_data_dir,
                    ss_dir=self.val_ss_dir,
                    mode="eval",
                    filter_path=self.val_filter_path
                )
            else:
                self.eval_dataset = None
        else:
            self.predict_dataset = dataset_gen(
                data_dir=self.predict_data_dir,
                ss_dir=self.predict_ss_dir,
                mode="predict",
            )

    def _gen_dataloader(self, stage):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)

        if stage == "train":
            dataset = self.train_dataset
            # Filter the dataset, if necessary
            dataset.reroll()
        elif stage == "eval":
            dataset = self.eval_dataset
        elif stage == "predict":
            dataset = self.predict_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = DemoFoldBatchCollator()

        dl = DemoFoldDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=batch_collator,
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train")

    def val_dataloader(self):
        if self.eval_dataset is not None:
            return self._gen_dataloader("eval")
        return None

    def predict_dataloader(self):
        return self._gen_dataloader("predict")

