import argparse
import numpy as np
import json
import logging


def write_fasta_from_dict(fasta_path: str, seq_dict: dict):
    with open(fasta_path, 'w') as f:
        for key, seq in seq_dict.items():
            file_id, chain_ids = key.split("_")
            chain_ids = chain_ids.split(",")
            term = "Chains " if len(chain_ids) > 1 else "Chain "
            header = file_id.upper() + "|" + term + ", ".join(chain_ids)
            f.write('>' + header + '\n')
            f.write(seq + '\n')

def main(args):
    with open(args.cache_path, "r") as f:
        mmcif_cache = json.load(f)
    logging.warning("The number of candidate files: {}".format(len(mmcif_cache)))

    # remove overlap
    with open(args.log_path, "r") as f:
        mmcif_cache_log = json.load(f)
    for k in mmcif_cache_log["overlap"]:
        mmcif_cache.pop(k)
    logging.warning("Remove overlap chains.")
    logging.warning("The number of files containing RNA chains: {}".format(len(mmcif_cache)))

    has_rna = []
    for k, v in mmcif_cache.items():
        if v["no_chains"]["rna"]:
            has_rna.append(k)
    logging.warning("The number of files containing RNA chains: {}".format(len(has_rna)))

    # resolution
    valid_resolution = []
    for file_id in has_rna:
        if mmcif_cache[file_id]["header"]["resolution"] <= 4.5:
            valid_resolution.append(file_id)
    logging.warning("The number of files with valid resolution: {}".format(len(valid_resolution)))

    # modify format
    rna_chains = {}
    for file_id in valid_resolution:
        for chain_ids, chain in mmcif_cache[file_id]["rna"].items():
            rna_chains[file_id + "_" + chain_ids] = chain
    for k, v in rna_chains.items():
        v = ''.join(['X' if nucleotide not in 'AUCG' else nucleotide for nucleotide in v])
        rna_chains[k] = v

    # length
    minl = 15
    maxl = np.inf
    valid_length = {}
    for k, v in rna_chains.items():
        if len(v) <= maxl and len(v) >= minl:
            valid_length[k] = v
    logging.warning("The number of chains with valid length: {}".format(len(valid_length)))

    # Any single nucleotide accounts for more than 90%
    valid_most = {}
    for k,v in rna_chains.items():
        s = max(v, key=v.count)
        if v.count(s)/len(v) < 0.9:
            valid_most[k] = v
    logging.warning("Any single nucleotide: {}".format(len(valid_most)))

    # Unknown nucleotide “X” accounts for more than 50%
    valid_X = {}
    for k,v in rna_chains.items():
        if v.count("X")/len(v) < 0.5:
            valid_X[k] = v
    logging.warning("X: {}".format(len(valid_X)))

    keys = set(valid_X.keys()) & set(valid_most.keys()) & set(valid_length.keys())
    valid_total = {k: rna_chains[k] for k in rna_chains if k in keys}
    len(valid_total)
    logging.warning("total: {}".format(len(valid_total)))

    logging.warning("Writing fasta")
    write_fasta_from_dict(args.fasta_path, valid_total)



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
        "log_path", type=str,
        help="Path for .json mmcif cache log"
    )
    args = parser.parse_args()

    main(args)