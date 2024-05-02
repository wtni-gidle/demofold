import argparse
import logging
import numpy as np
import os
import random
import time
import tempfile

from demofold.data.data_pipeline import SSRunner
from demofold.model.model import DemoFold
import logging
import os
import argparse

import time

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import torch
torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if (
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from demofold.config import model_config
from demofold.data import feature_pipeline, data_pipeline
from demofold.utils.tensor_utils import tensor_tree_map

RNAFOLD_BINARY_PATH = '/expanse/projects/itasser/jlspzw/nwentao/ss-program/ViennaRNA/2.4.18/bin/RNAfold'


def parse_fasta(fasta_path: str):
    with open(fasta_path, "r") as fp:
        fasta_string = fp.readlines()
    desc = fasta_string[0][1:].strip()
    seq = fasta_string[1].strip()

    return desc, seq

def run_ss(desc_seq_pair, ss_runner: SSRunner, output_dir):
    """运行一个文件"""
    desc, seq = desc_seq_pair
    
    fd, fasta_path = tempfile.mkstemp(suffix=".fasta")
    with os.fdopen(fd, 'w') as fp:
        fp.write(f'>seq\n{seq}')

    try:
        ss_runner.run(
            fasta_path, output_dir
        )
    except Exception as e:
        logging.warning(e)
        logging.warning(f"Failed to run ss for {desc}. Skipping...")
        os.remove(fasta_path)
        return 
    
    os.remove(fasta_path)

def generate_feature_dict(
    tag,
    seq,
    ss_dir,
    data_processor: data_pipeline.DataPipeline,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")

    tag = tag
    seq = seq
    with open(tmp_fasta_path, "w") as fp:
        fp.write(f">{tag}\n{seq}")

    feature_dict = data_processor.process_fasta(
        fasta_path=tmp_fasta_path,
        ss_dir=ss_dir
    )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict

def run_model(model, batch, tag):
    with torch.no_grad():
        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")

    return out

def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    config_e2e = model_config("e2e", long_sequence_inference=args.long_sequence_inference)
    config_geom = model_config("geom", long_sequence_inference=args.long_sequence_inference)

    data_processor = data_pipeline.DataPipeline()
    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2 ** 32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config_e2e.data)
    ss_dir = output_dir_base

    desc, seq = parse_fasta(args.fasta_path)

    ss_runner = SSRunner(
        rnafold_binary_path=RNAFOLD_BINARY_PATH
    )
    run_ss((desc, seq), ss_runner, output_dir=ss_dir)

    feature_dict = generate_feature_dict(
        desc,
        seq,
        ss_dir,
        data_processor,
        args,
    )
    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode='predict'
    )
    processed_feature_dict = {
        k: torch.as_tensor(v, device=args.model_device)
        for k, v in processed_feature_dict.items()
    }

    for model_num in range(6):
        model_e2e = DemoFold(config_e2e)
        model_e2e.eval()
        path = getattr(args, f"e2e_path_{model_num}")
        sd = torch.load(path)
        model_e2e.load_state_dict(sd["ema"]["params"])
        model_e2e = model_e2e.to(args.model_device)
        out_e2e = run_model(model_e2e, processed_feature_dict, desc)

        # [n_res, 4, 3]
        out_e2e = out_e2e["final_atom_positions"]
        out_e2e = tensor_tree_map(lambda x: np.array(x.cpu()), out_e2e)
        # ["C4'", "P", "N1", "N9"]
        mapping = {
            "A": [1, 0, 3],
            "C": [1, 0, 2],
            "G": [1, 0, 3],
            "U": [1, 0, 2]
        }
        indices = [mapping[nucleotide] for nucleotide in seq]
        sub_arrays = [out_e2e[i, ind, :] for i, ind in enumerate(indices)]
        result_tensor = np.stack(sub_arrays, axis=0)
        #todo注意原子顺序
        np.save(os.path.join(args.output_dir, f"e2e_{model_num}"), result_tensor)


    model_geom = DemoFold(config_geom)
    model_geom.eval()
    sd = torch.load(args.geom_path)
    model_geom.load_state_dict(sd["ema"]["params"])
    model_geom = model_geom.to(args.model_device)
    out_geom = run_model(model_geom, processed_feature_dict, desc)
    
    keys_mapping = {
        "PP_logits": "pp",
        "CC_logits": "cc",
        "NN_logits": "nn",
        "PCCP_logits": "pccp",
        "PNNP_logits": "pnnp", 
        "CNNC_logits": "cnnc",
    }
    out_geom = {v: out_geom[k] for k, v in keys_mapping.items()}
    out_geom = tensor_tree_map(lambda x: torch.softmax(x, dim=-1), out_geom)
    out_geom = tensor_tree_map(lambda x: np.array(x.cpu()), out_geom)
    #todo注意原子顺序
    np.save(os.path.join(args.output_dir, "geo"), out_geom, allow_pickle=True)
    
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_path", type=str,
        help="Path to directory containing FASTA files, one sequence per file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--e2e_path_0", type=str
    )
    parser.add_argument(
        "--e2e_path_1", type=str
    )
    parser.add_argument(
        "--e2e_path_2", type=str
    )
    parser.add_argument(
        "--e2e_path_3", type=str
    )
    parser.add_argument(
        "--e2e_path_4", type=str
    )
    parser.add_argument(
        "--e2e_path_5", type=str
    )
    parser.add_argument(
        "--geom_path", type=str
    )
    parser.add_argument(
        "--data_random_seed", type=int, default=None
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    args = parser.parse_args()
    main(args)
