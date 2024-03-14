import argparse
import time
from functools import partial
import logging
import os
from multiprocessing import Pool
import tempfile

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from demofold.data.data_pipeline import SSRunner

RNAFOLD_BINARY_PATH = '/expanse/projects/itasser/jlspzw/nwentao/ss-program/ViennaRNA/2.4.18/bin/RNAfold'

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

    desc_seq_map = parse_fasta(args.fasta_path)
    desc_seq_pairs = list(desc_seq_map.items())
    ss_runner = SSRunner(
        rnafold_binary_path=RNAFOLD_BINARY_PATH
    )
    fn = partial(run_ss, ss_runner=ss_runner, output_dir=args.output_dir)

    # 使用进程池并行处理文件
    logging.warning("Start precompute ss...")
    with Pool(processes=args.no_workers) as p:
        # 使用 imap_unordered 并行处理文件
        for _ in p.imap_unordered(fn, desc_seq_pairs, chunksize=args.chunksize):
            pass
    
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
    parser.add_argument(
        "--no_workers", type=int, default=4,
        help="Number of workers to use for parsing"
    )
    parser.add_argument(
        "--chunksize", type=int, default=10,
        help="How many files should be distributed to each worker at a time"
    )

    args = parser.parse_args()

    main(args)