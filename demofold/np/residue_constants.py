from typing import Mapping

import numpy as np

# residue
restypes = [
    "A",
    "C",
    "G",
    "U",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 4.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.

restypes_with_x = restypes + ["X"]
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}

restypes_with_x_and_gap = restypes + ["X", "-"]
restype_order_with_x_and_gap =  {restype: i for i, restype in enumerate(restypes_with_x_and_gap)}

def sequence_to_onehot(
    sequence: str, 
    mapping: Mapping[str, int], 
    map_unknown_to_x: bool = False
) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
        sequence: 
            An amino acid sequence.

        mapping: 
            A dictionary mapping amino acids to integers.

        map_unknown_to_x: 
            If True, any amino acid that is not in the mapping will be 
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

    for idx, res_type in enumerate(sequence):
        if map_unknown_to_x:
            if res_type.isalpha() and res_type.isupper():
                res_id = mapping.get(res_type, mapping["X"])
            else:
                raise ValueError(
                    f"Invalid character in the sequence: {res_type}"
                )
        else:
            res_id = mapping[res_type]
        one_hot_arr[idx, res_id] = 1

    return one_hot_arr

# atom

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
              "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", 
              "N3", "C4", "O2", "N4", "O6", "N2", "O4"]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 28.



# residue and atom
# A list of atoms (excluding hydrogen) for each nucleotide type. PDB naming convention.
residue_atoms = {
    "A": ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
          "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", 
          "N3", "C4"],
    "C": ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
          "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "G": ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
          "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", 
          "N2", "N3", "C4"],
    "U": ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
          "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
}



# todo 需要查阅资料
# backbone上的三个原子的坐标
bb_atom3_positions = {
    "A": [
        ["C4'", (-0.8031747, 1.3708649, -0.56768906)],
        ["P", (2.3240912, -0.80317456, -1.5209152)],
        ["N9", (-1.520915, -0.5676891, 2.088603)],
    ],
    "C": [
        ["C4'", (-0.78803474, 1.3541472, -0.566115)],
        ["P", (2.35021, -0.7880346, -1.5621773)],
        ["N1", (-1.5621774, -0.56611556, 2.1282897)],
    ],
    "G": [
        ["C4'", (-0.8028024, 1.3702725, -0.5674678)],
        ["P", (2.3250363, -0.80280226, -1.5222337)],
        ["N9", (-1.5222336, -0.5674678, 2.0897026)],
    ],
    "U": [
        ["C4'", (-0.7881318, 1.3542684, -0.5661326)],
        ["P", (2.3493478, -0.7881313, -1.5612125)],
        ["N1", (-1.5612122, -0.5661328, 2.1273444)],
    ],
}
bb_atom_types = ["C4'", "P", "N1", "N9"]
bb_atom_order = {atom_type: i for i, atom_type in enumerate(bb_atom_types)}
bb_atom_type_num = len(bb_atom_types)  # := 4.

# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
restype_atom28_to_bb = np.zeros([5, 28], dtype=int)
restype_atom28_mask = np.zeros([5, 28], dtype=np.float32)
restype_atom28_bb_positions = np.zeros([5, 28, 3], dtype=np.float32)

# 新增只含三个原子的坐标
restype_atom3_bb_positions = np.zeros([5, 3, 3], dtype=np.float32)
tmp_mapping = {
    "A": ["C4'", "P", "N9"],
    "C": ["C4'", "P", "N1"],
    "G": ["C4'", "P", "N9"],
    "U": ["C4'", "P", "N1"],
}
for res_idx, restype in enumerate(restypes):
    for atomname, atom_position in bb_atom3_positions[restype]:
        atom_idx = tmp_mapping[restype].index(atomname)
        restype_atom3_bb_positions[res_idx, atom_idx, :] = atom_position

# 新增只含四个原子的坐标
restype_atom4_bb_positions = np.zeros([5, 4, 3], dtype=np.float32)
for res_idx, restype in enumerate(restypes):
    for atomname, atom_position in bb_atom3_positions[restype]:
        atom_idx = bb_atom_order[atomname]
        restype_atom4_bb_positions[res_idx, atom_idx, :] = atom_position


# fill restype_atom28_mask
for res_idx, restype in enumerate(restypes):
    for atomname in residue_atoms[restype]:
        atom_idx = atom_order[atomname]
        restype_atom28_mask[res_idx, atom_idx] = 1

def _make_bb_constants():
    """Fill the arrays above."""
    for res_idx, restype in enumerate(restypes):
        for atomname, atom_position in bb_atom3_positions[restype]:
            atom_idx = atom_order[atomname]
            # 只记录backbone，记为1
            restype_atom28_to_bb[res_idx, atom_idx] = 1
            restype_atom28_bb_positions[res_idx, atom_idx, :] = atom_position
        
_make_bb_constants()

