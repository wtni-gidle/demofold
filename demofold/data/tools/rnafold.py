"""Library to run RNAfold from Python."""
from typing import List, Mapping, Any
import os
import logging
import subprocess

from . import utils
from .utils import SSPredictor

def _extract_DBN(ss_str: str) -> str:
    """extract dot-bracket notation"""
    ss_str = ss_str.split("\n")
    dbn = ss_str[2].strip().split()[0]

    return dbn

def _extract_prob_mat(dp_str: str) -> str:
    """
    extract probability matrix lines
    list of lines, 每一行格式如下：
    i j sqrt(p)
    """
    dp_str = dp_str.split("\n")
    dp_str = [line.strip() for line in dp_str]
    prob_mat_lines = []
    for line in dp_str:
        line = line.strip()
        if (line.endswith("ubox") and 
            len(line.split()) == 4 and 
            line[0].isdigit()):
            prob_mat_lines.append(" ".join(line.split()[:-1]))
    prob = "\n".join(prob_mat_lines)

    return prob

class RNAfold(SSPredictor):
    """Python wrapper of the RNAfold binary."""

    def __init__(self, binary_path: str):
        self.binary_path = binary_path
    
    def predict(self, input_fasta_path: str) -> Mapping[str, str]:
        """
        输出的key即format, value必须是字符串
        """
        input_fasta_path = os.path.abspath(input_fasta_path)
        ori_dir = os.getcwd()

        with utils.tmpdir_manager() as tmp_dir:
            os.chdir(tmp_dir)
            outfile_path = "ss.ViennaRNAss"

            cmd_flags = [
                "--outfile=" + outfile_path,
                "-p",
                "--noPS",
            ]

            cmd = [self.binary_path] + cmd_flags + [input_fasta_path]
            
            logging.info('Launching subprocess "%s"', " ".join(cmd))
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            with utils.timing(f"RNAfold calculate"):
                _, stderr = process.communicate()
                retcode = process.wait()
            if retcode:
                raise RuntimeError(
                    "RNAfold failed\nstderr:\n%s\n" % stderr.decode("utf-8")
                )
            
            # extract dot-bracket notation
            with open(outfile_path, "r") as f:
                ss_str = f.read()
            dbn = _extract_DBN(ss_str)

            # extract probability matrix lines
            dp_file = [fn for fn in os.listdir() if fn.endswith("dp.ps")]
            assert len(dp_file) == 1
            with open(dp_file[0], "r") as f:
                dp_str = f.read()
            prob = _extract_prob_mat(dp_str)
            
            os.chdir(ori_dir)

        output = {
            "dbn": dbn,
            "prob": prob,
        }

        return output

