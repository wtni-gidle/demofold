import argparse
import time
from functools import partial
import json
import logging
import os
from itertools import chain
from multiprocessing import Pool, Manager

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from demofold.data.mmcif_parsing import parse


def merge_dict(dictionary):
    merged_dict = {}
    for key, value in dictionary.items():
        merged_dict.setdefault(value, []).append(key)
    return {','.join(keys): value for value, keys in merged_dict.items()}

def parse_file(f, mmcif_dir, results, overlap_list, error_list):
    with open(os.path.join(mmcif_dir, f), "r") as fp:
        mmcif_string = fp.read()
    file_id = os.path.splitext(f)[0]
    mmcif = parse(file_id=file_id, mmcif_string=mmcif_string)
    
    if mmcif.mmcif_object is None:
        error = list(mmcif.errors.values())[0]
        logging.warning(f'Failed to parse {f}. Skipping...')
        logging.warning(f'{error.__class__.__name__}: {error}')
        error_list.append(file_id)
        return 
    
    mmcif = mmcif.mmcif_object

    chain_ids = list(chain(*[chains.keys() for chains in mmcif.chain_to_seqres.values()]))
    overlap = len(chain_ids) != len(set(chain_ids))
    if overlap:
        logging.warning(f'Overlapping chains appear in {f}')
        overlap_list.append(file_id)
    
    local_data = {}
    local_data["header"] = mmcif.header
    local_data["protein"] = merge_dict(mmcif.chain_to_seqres["protein"])
    local_data["rna"] = merge_dict(mmcif.chain_to_seqres["rna"])
    local_data["dna"] = merge_dict(mmcif.chain_to_seqres["dna"])
    local_data["no_chains"] = {k: len(v) for k, v in mmcif.chain_to_seqres.items()}

    # 将结果写入共享字典中
    results.update({file_id: local_data})

def main(args):
    start = time.perf_counter()
    files = os.listdir(args.mmcif_dir)

    # 创建共享字典用于存储结果
    results = Manager().dict()
    overlap_list = Manager().list()
    error_list = Manager().list()
    fn = partial(parse_file, mmcif_dir=args.mmcif_dir, results=results, 
                 overlap_list=overlap_list, error_list=error_list)

    # 使用进程池并行处理文件
    logging.warning("Start parsing mmcif files...")
    with Pool(processes=args.no_workers) as p:
        # 使用 imap_unordered 并行处理文件
        for _ in p.imap_unordered(fn, files, chunksize=args.chunksize):
            pass
        
    logs = {
        "overlap": list(overlap_list), 
        "error": list(error_list)
    }

    with open(args.output_path, "w") as fp:
        json.dump(dict(results), fp, indent=4)
    logging.warning(f"All results have been written to {args.output_path}")
    
    with open(args.log_path, "w") as fp:
        json.dump(logs, fp, indent=4)
    logging.warning(f"All exceptional results have been written to {args.log_path}")
    
    total_time = time.perf_counter() - start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.warning(f"Total time: {int(hours)} h {int(minutes)} min {seconds:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mmcif_dir", type=str,
        help="Path to directory containing mmCIF files"
    )
    parser.add_argument(
        "output_path", type=str,
        help="Path for .json output"
    )
    parser.add_argument(
        "log_path", type=str,
        help="Path for .json log"
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