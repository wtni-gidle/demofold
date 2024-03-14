"""Functions for parsing various file formats."""
import collections
import dataclasses
import itertools
import re
import string
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set
import numpy as np




@dataclasses.dataclass(frozen=True)
class MSA:
    """Class representing a parsed MSA file"""
    sequences: Sequence[str]
    descriptions: Optional[Sequence[str]]

    def __post_init__(self):
        if(not (
            len(self.sequences) == 
            len(self.descriptions)
        )):
            raise ValueError(
                "All fields for an MSA must have the same length"
            )

    def __len__(self):
        return len(self.sequences)

    def truncate(self, max_seqs: int):
        return MSA(
            sequences=self.sequences[:max_seqs],
            descriptions=self.descriptions[:max_seqs],
        )


@dataclasses.dataclass(frozen=True)
class SS:
    contact: Optional[np.ndarray]
    prob: Optional[np.ndarray]

    


def parse_dbn(dbn_string: str) -> np.ndarray:
    """
    返回0-1矩阵

    Example:
        >>> dbn_string = "(((.((..(((.))))))..))"
        >>> parse_dbn(dbn_string)
        array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=int32)
    """
    n_res = len(dbn_string)
    ss_matrix = np.zeros((n_res, n_res), dtype=np.int32)
    slist = []

    for i, char in enumerate(dbn_string):
        if char in "(<[{":
            slist.append(i)
        elif char in ")>]}":
            j = slist.pop()
            ss_matrix[i, j] = ss_matrix[j, i] = 1
        elif char not in  ".-":
            raise ValueError(
                f'Unknown secondary structure state: {char} at position {i}'
            )
        
    return ss_matrix


def parse_rnafold(dbn_string: str, prob_string: str) -> SS:
    ss_matrix = parse_dbn(dbn_string)

    n_res = len(dbn_string)
    prob_matrix = np.zeros((n_res, n_res), dtype=np.int32)
    prob_string = prob_string.split("\n")
    for line in prob_string:
        words = line.split()
        i = int(words[0]) - 1
        j = int(words[1]) - 1
        score = float(words[2])
        prob_matrix[i, j] = prob_matrix[j, i] = score
    
    return SS(
        contact=ss_matrix, 
        prob=prob_matrix
    )


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif line.startswith("#"):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions