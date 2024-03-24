import os
from typing import Mapping, Optional, Sequence, Any, MutableMapping
import numpy as np
from . import parsers, mmcif_parsing
from .tools import rnafold
from ..np import residue_constants as rc


FeatureDict = MutableMapping[str, np.ndarray]


# region: mmcif feats
def make_sequence_features(
    sequence: str, 
    description: str, 
    num_res: int
) -> FeatureDict:
    """
    Construct a feature dict of sequence features.
    sequence: ACGUCG
    description: {file_id}_{chain_id}
    return:
        restype: [N_res, 4+1]
        between_segment_residues: [N_res,]
        domain_name: {file_id}_{chain_id}
        seq_length: [num_res] * num_res
        sequence: ACGUCG
    """
    features = {}
    # 带X的，即[N_res, 4+1]
    features["restype"] = rc.sequence_to_onehot(
        sequence=sequence,
        mapping=rc.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=object
    )
    # features["residue_index"] = np.arange(num_res, dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=object
    )
    return features

def make_mmcif_features(
    mmcif_object: mmcif_parsing.MmcifObject, 
    chain_id: str, 
    ptype: str = "rna"
) -> FeatureDict:
    """
    restype: [N_res, 4+1]
    between_segment_residues: [N_res,]
    domain_name: {file_id}_{chain_id}
    seq_length: [num_res] * num_res
    sequence: ACGUCG
    all_atom_positions: [N_res, bb_atom_type_num, 3]原始坐标
    all_atom_mask: [N_res, bb_atom_type_num]
    resolution
    release_date
    """
    # 本来应该是使用转化为X的序列, 但是rc.sequence_to_onehot包含了这一步, 
    # 因此可以直接使用这里的序列
    input_sequence = mmcif_object.chain_to_seqres[ptype][chain_id]
    # {file_id}_{chain_id}
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )
    # [N_res, bb_atom_type_num, 3]
    # [N_res, bb_atom_type_num]
    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id, ptype=ptype
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["resolution"] = np.array(
        [mmcif_object.header["resolution"]], dtype=np.float32
    )

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=object
    )

    return mmcif_feats
# endregion

# region: msa feats
def make_msa_features(msas: Sequence[parsers.MSA]) -> FeatureDict:
    """
    Constructs a feature dict of MSA features.
    openfold整合了
        deletion_matrix_int, 
        msa, 
        num_alignments, 
        msa_species_identifiers
    我们只用到msa和num_alignments, 其中Msa对象的description属性没有被用到
        msa: shape of [num_alignments, N_res]
        num_alignments: [num_alignments, ..., num_alignments]
    """
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(
                f"MSA {msa_index} must contain at least one sequence."
            )
        for sequence in msa.sequences:
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [rc.restype_order_with_x_and_gap.get(res, rc.restype_order_with_x_and_gap["X"]) for res in sequence]
            )

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)
    features = {}
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array(
        [num_alignments] * num_res, dtype=np.int32
    )
    return features

def make_dummy_msa_obj(input_sequence: str) -> parsers.MSA:
    return parsers.MSA(
        sequences=[input_sequence],
        descriptions=['dummy']
    )

# Generate 1-sequence MSA features having only the input sequence
def make_dummy_msa_feats(input_sequence: str) -> FeatureDict:
    msa_data_obj = make_dummy_msa_obj(input_sequence)
    return make_msa_features([msa_data_obj])
# endregion

# region: ss feats
def make_ss_features(ss_data: parsers.SS) -> FeatureDict:
    """
    写不下去了...先写一个应急版本
    ss: [N_res, N_res, 4]
    """
    ss = np.concatenate(
        (ss_data.contact[..., None], 
         ss_data.prob[..., None]), 
        axis=-1,
        dtype=np.float32,
    )
    ss = np.concatenate(
        (ss,
         ss), 
        axis=-1,
        dtype=np.float32,
    )

    return {"ss": ss}

# endregion
    
def run_ss_tool(
    ss_runner: rnafold.RNAfold, # todo建立一个基类
    fasta_path: str,
    ss_out_prefix: str
) -> Mapping[str, str]:
    """
    {ss_out_prefix}.dbn
    {ss_out_prefix}.prob
    """
    result = ss_runner.predict(fasta_path)
    for fmt in result:
        with open(ss_out_prefix + "." + fmt, "w") as f:
            f.write(result[fmt])
    
    return result

class DataPipeline:
    """Assembles input features."""

    def _parse_ss_data(
        self,
        ss_dir: str
    ):
        # !写不下去了...先写一个应急版本
        for f in os.listdir(ss_dir):
            path = os.path.join(ss_dir, f)

            if f == "rnafold.dbn":
                with open(path, "r") as fp:
                    dbn_string = fp.read()
            elif f == "rnafold.prob":
                with open(path, "r") as fp:
                    prob_string = fp.read()
            
        ss_data = parsers.parse_rnafold(dbn_string, prob_string)
        
        return ss_data
    
    def _process_ss_feats(
        self, 
        ss_dir: str
    ):
        ss_data = self._parse_ss_data(ss_dir)
        ss_features = make_ss_features(ss_data)

        return ss_features

    def process_fasta(
        self,
        fasta_path: str,
        ss_dir: str,
    ) -> FeatureDict:
        """
        Assembles features for a single sequence in a FASTA file
        mmcif_feats: 
            restype: [N_res, 4+1]
            between_segment_residues: [N_res,]
            domain_name: {file_id}_{chain_id}
            seq_length: [num_res] * num_res
            sequence: ACGUCG
        msa_features: 
            msa: shape of [num_alignments=1, N_res] 
                每个值是ACGUX-到012345, 但是我们是单序列, 不可能有5
            num_alignments: [num_alignments=1, ..., num_alignments]
        ss_feats:
            ss: [N_res, N_res, 4]
        """
        with open(fasta_path) as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f"More than one input sequence found in {fasta_path}."
            )
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
        )

        # If using seqemb mode, generate a dummy MSA features using just the sequence
        msa_features = make_dummy_msa_feats(input_sequence)

        # ss_feats
        ss_feats = self._process_ss_feats(ss_dir)
        assert len(input_sequence) == len(list(ss_feats.values())[0])
        
        return {
            **sequence_features,
            **msa_features, 
            **ss_feats
        }

    def process_mmcif(
        self,
        mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
        ss_dir: str,
        chain_id: Optional[str] = None,
        ptype: bool = "rna"
    ) -> FeatureDict:
        """
            Assembles features for a specific chain in an mmCIF object.

            If chain_id is None, it is assumed that there is only one chain
            in the object. Otherwise, a ValueError is thrown.

            返回的是一个字典.
            openfold返回
                mmcif_feats, 
                template_features, 
                msa_features, 
                sequence_embedding_features(正常模式为空, seqemb模式下为ESM的embedding)

            demofold返回
                mmcif_feats: 
                    restype: [N_res, 4+1]
                    between_segment_residues: [N_res,]
                    domain_name: {file_id}_{chain_id}
                    seq_length: [num_res] * num_res
                    sequence: ACGUCG
                    all_atom_positions: [N_res, bb_atom_type_num, 3]原始坐标
                    all_atom_mask: [N_res, bb_atom_type_num]
                    resolution
                    release_date
                msa_features: 
                    msa: shape of [num_alignments=1, N_res] 
                        每个值是ACGUX-到012345, 但是我们是单序列, 不可能有5
                    num_alignments: [num_alignments=1, ..., num_alignments]
                ss_feats:
                    ss: [N_res, N_res, 4]
        """
        if chain_id is None:
            chains = mmcif.structure.get_chains()
            chain = next(chains, None)
            if chain is None:
                raise ValueError("No chains in mmCIF file")
            chain_id = chain.id

        # mmcif_feats
        mmcif_feats = make_mmcif_features(mmcif, chain_id, ptype)

        input_sequence = mmcif.chain_to_seqres[ptype][chain_id]
        
        # msa_feats
        # If using seqemb mode, generate a dummy MSA features using just the sequence
        msa_features = make_dummy_msa_feats(input_sequence)

        # ss_feats
        ss_feats = self._process_ss_feats(ss_dir)
        assert len(input_sequence) == len(list(ss_feats.values())[0])

        return {**mmcif_feats, **msa_features, **ss_feats}

class SSRunner:
    def __init__(
        self,
        rnafold_binary_path: Optional[str] = None,
    ):
        self.rnafold_runner = None
        if rnafold_binary_path is not None:
            self.rnafold_runner = rnafold.RNAfold(
                binary_path=rnafold_binary_path
            )
        
    def run(
        self,
        fasta_path: str,
        output_dir: str,
    ):
        """
        Run SS Prediction tools on one sequence.
        可能有多个工具, 比如rnafold和petfold, 结果都保存在output_dir
        """
        if self.rnafold_runner is not None:
            rnafold_out_prefix = os.path.join(output_dir, "rnafold")
            run_ss_tool(
                ss_runner=self.rnafold_runner,
                fasta_path=fasta_path,
                ss_out_prefix=rnafold_out_prefix,
            )


        