"""
直接执行, 不需要sbatch
fasta_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/pdb_RNA.fasta"
cache_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/mmcif_cache.json"
train_filter_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/train_filter.txt"
val_filter_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/val_filter.txt"
"""
from typing import Sequence
import argparse
import json
import logging

CUTOFF = "2022-05-01"

def parse_fasta(fasta_path) -> Sequence[str]:
    """{file_id}_{chain_id}"""
    with open(fasta_path, "r") as fp:
        fasta_string = fp.read()

    chain_ids = []
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            file_id, chain_info = line[1:].split("|")
            file_id = file_id.lower()
            chain_id = chain_info.split(",")[0].split()[-1]
            desc = "_".join([file_id, chain_id])
            chain_ids.append(desc)  # Remove the '>' at the beginning.

    return chain_ids


def main(args):
    chain_ids = parse_fasta(args.fasta_path)
    with open(args.cache_path, "r") as fp:
        mmcif_cache = json.load(fp)
    
    train_chain_ids = []
    val_chain_ids = []
    for each in chain_ids:
        file_id, _ = each.split("_")
        release_date = mmcif_cache[file_id]["header"]["release_date"]
        if release_date != "?":
            if release_date >= CUTOFF:
                val_chain_ids.append(each)
            else:
                train_chain_ids.append(each)
        else:
            train_chain_ids.append(each)

    logging.warning(f"Writing training set to {args.train_filter_path}")
    with open(args.train_filter_path, "w") as fp:
        fp.write("\n".join(train_chain_ids))
    logging.warning(f"The number of chains for training: {len(train_chain_ids)}")

    logging.warning(f"Writing validation set to {args.val_filter_path}")
    with open(args.val_filter_path, "w") as fp:
        fp.write("\n".join(val_chain_ids))
    logging.warning(f"The number of chains for validation: {len(val_chain_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_path", type=str,
        help="Path for .fasta output"
    )
    parser.add_argument(
        "cache_path", type=str,
        help="Path for .json mmcif cache"
    )
    parser.add_argument(
        "train_filter_path", type=str,
        help="Output path for .txt train filter path"
    )
    parser.add_argument(
        "val_filter_path", type=str,
        help="Output path for .txt val filter path"
    )
    args = parser.parse_args()

    main(args)