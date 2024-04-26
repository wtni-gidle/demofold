"""
适用于少量的rna序列
一个fasta文件只包含一个RNA序列
"""
from demofold.data.data_pipeline import SSRunner
import logging
import tempfile
import os
import argparse
import re

import time

RNAFOLD_BINARY_PATH = '/expanse/projects/itasser/jlspzw/nwentao/ss-program/ViennaRNA/2.4.18/bin/RNAfold'


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]

def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [re.split('\W| \|', t)[0] for t in tags]

    return tags, seqs


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

def run_ss(desc_seq_pair, ss_runner: SSRunner, output_dir):
    """运行一个文件"""
    desc, seq = desc_seq_pair
    ss_dir = os.path.join(output_dir, desc)
    try:
        os.makedirs(ss_dir)
    except Exception as e:
        logging.warning(f"Failed to create directory for {desc} with exception {e}...")
        return 
    
    fd, fasta_path = tempfile.mkstemp(suffix=".fasta")
    with os.fdopen(fd, 'w') as fp:
        fp.write(f'>seq\n{seq}')

    try:
        ss_runner.run(
            fasta_path, ss_dir
        )
    except Exception as e:
        logging.warning(e)
        logging.warning(f"Failed to run ss for {desc}. Skipping...")
        os.remove(fasta_path)
        os.rmdir(ss_dir)
        return 
    
    os.remove(fasta_path)


def main(args):
    start = time.perf_counter()
    ss_runner = SSRunner(
        rnafold_binary_path=RNAFOLD_BINARY_PATH
    )
    for fasta_file in list_files_with_extensions(args.fasta_dir, (".fasta", ".fa")):
        fasta_path = os.path.join(args.fasta_dir, fasta_file)
        with open(fasta_path, "r") as fp:
            data = fp.read()

        tags, seqs = parse_fasta(data)
        assert len(tags) == 1
        assert len(tags) == len(set(tags))



    desc_seq_map = parse_fasta(args.fasta_path)
    desc_seq_pairs = list(desc_seq_map.items())
    assert len(desc_seq_pairs) == 1

    logging.warning("Start precompute ss...")
    
    run_ss(desc_seq_pairs[0], ss_runner, output_dir=args.output_dir)
    
    total_time = time.perf_counter() - start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.warning(f"Total time: {int(hours)} h {int(minutes)} min {seconds:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_path", type=str,
        help="Path for .fasta input"
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Directory in which to output ss"
    )

    args = parser.parse_args()

    main(args)