import argparse
import logging
import math
import numpy as np
import os
import pickle
import random
import time
import tempfile

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

def parse_fasta(fasta_path):
    with open(fasta_path, "r") as fp:
        fasta_string = fp.read()

    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            file_id, chain_info = line[1:].split("|")
            file_id = file_id.lower()
            chain_id = chain_info.split(",")[0].split()[-1]
            desc = "_".join([file_id, chain_id])
            descriptions.append(desc)  # Remove the '>' at the beginning.
            sequences.append("")
        else:
            sequences[index] += line
    
    desc_seq_map = dict(zip(descriptions, sequences))

    return desc_seq_map

def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    config = model_config(args.config_preset, long_sequence_inference=args.long_sequence_inference)
    data_processor = data_pipeline.DataPipeline()
    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2 ** 32)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if args.use_precomputed_ss is None:
        ss_dir = os.path.join(output_dir_base, "ss")
    else:
        ss_dir = args.use_precomputed_ss
    
    tag_list = []
    seq_list = []
    for fasta_file in list_files_with_extensions(args.fasta_dir, (".fasta", ".fa")):
        # Gather input sequences
        fasta_path = os.path.join(args.fasta_dir, fasta_file)
        with open(fasta_path, "r") as fp:
            data = fp.read()

        tags, seqs = parse_fasta(data)