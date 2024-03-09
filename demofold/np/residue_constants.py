from typing import Mapping

import numpy as np

# residue
restypes = [
    "A",
    "U",
    "C",
    "G",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 4.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}

restypes_with_x_and_gap = restypes + ["X", "-"]

def sequence_to_onehot(
    sequence: str, 
    mapping: Mapping[str, int], 
    map_unknown_to_x: bool = False
) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
      sequence: An amino acid sequence.
      mapping: A dictionary mapping amino acids to integers.
      map_unknown_to_x: If True, any amino acid that is not in the mapping will be
        mapped to the unknown amino acid 'X'. If the mapping doesn't contain
        amino acid 'X', an error will be thrown. If False, any amino acid not in
        the mapping will throw an error.

    Returns:
      A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of
      the sequence.

    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_aas - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_aas-1 "
            "without any gaps. Got: %s" % sorted(mapping.values())
        )

    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for aa_index, aa_type in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping["X"])
            else:
                raise ValueError(
                    f"Invalid character in the sequence: {aa_type}"
                )
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1

    return one_hot_arr

# atom
atom_types = [
    "C4'",
    "P",
    "N",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 3.

# todo 需要查阅资料
bb_atom3_positions = {
    "A": [
        ["C4'", (0.000, 0.000, 0.000)],
        ["P", (0.000, 0.000, 0.000)],
        ["N", (0.000, 0.000, 0.000)],
    ],
    "U": [
        ["C4'", (0.000, 0.000, 0.000)],
        ["P", (0.000, 0.000, 0.000)],
        ["N", (0.000, 0.000, 0.000)],
    ],
    "C": [
        ["C4'", (0.000, 0.000, 0.000)],
        ["P", (0.000, 0.000, 0.000)],
        ["N", (0.000, 0.000, 0.000)],
    ],
    "G": [
        ["C4'", (0.000, 0.000, 0.000)],
        ["P", (0.000, 0.000, 0.000)],
        ["N", (0.000, 0.000, 0.000)],
    ],
}

restype_atom3_bb_positions = np.zeros([5, 3, 3], dtype=np.float32)

def _make_bb_constants():
    """Fill the arrays above."""
    for res_idx, restype in enumerate(restypes):
        for atomname, atom_position in bb_atom3_positions[restype]:
            atom_idx = atom_order[atomname]
            
            restype_atom3_bb_positions[res_idx, atom_idx, :] = atom_position
        
_make_bb_constants()

