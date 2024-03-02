from collections import defaultdict
from dataclasses import dataclass
import io
import logging
from typing import Any, Mapping, Optional, Sequence, Tuple, Generator
from functools import partial
from copy import deepcopy

from Bio import PDB
from Bio.PDB.Structure import Structure
from Bio.Data import SCOPData



# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]
PType = str

PTYPE_LOOKUP: Mapping[PType, str] = {
    "protein": "peptide",
    "rna": "RNA",
    "dna": "DNA"
}

DNA_LOOKUP: Mapping[str, str] = {
    "DA": "A", "DT": "T", "DC": "C", "DG": "G"
}

RNA_LOOKUP: Mapping[str, str] = {
    "A": "A", "U": "U", "C": "C", "G": "G"
}

@dataclass(frozen=True)
class Monomer:
    id: str
    num: int


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclass(frozen=True)
class AtomSite:
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


# Used to map SEQRES index to a residue in the structure.
@dataclass(frozen=True)
class ResiduePosition:
    chain_id: str
    residue_number: int
    insertion_code: str


@dataclass(frozen=True)
class ResidueAtPosition:
    position: Optional[ResiduePosition]
    name: str
    is_missing: bool
    hetflag: str


@dataclass(frozen=True)
class MmcifObject:
    """Representation of a parsed mmCIF file.

    Contains:
      file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
        files being processed.
      header: Biopython header.
      structure: Biopython structure.
      chain_to_seqres: Dict mapping chain_id to 1 letter amino acid sequence. E.g.
        {'A': 'ABCDEFG'}
      seqres_to_structure: Dict; for each chain_id contains a mapping between
        SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                          1: ResidueAtPosition,
                                                          ...}}
      raw_string: The raw string used to construct the MmcifObject.
    """

    file_id: str
    header: PdbHeader
    structure: PdbStructure
    chain_to_seqres: Mapping[PType, Mapping[ChainId, SeqRes]]
    seqres_to_structure: Mapping[PType, Mapping[ChainId, Mapping[int, ResidueAtPosition]]]
    raw_string: Any


@dataclass(frozen=True)
class ParsingResult:
    """Returned by the parse function.

    Contains:
      mmcif_object: A MmcifObject, may be None if no chain could be successfully
        parsed.
      errors: A dict mapping (file_id, chain_id) to any exception generated.
    """

    mmcif_object: Optional[MmcifObject]
    errors: Mapping[Tuple[str, str], Any]


class ParseError(Exception):
    """An error indicating that an mmCIF file could not be parsed."""


# 虽然可以通过polymer_types选择想要的是蛋白质还是rna，但是输出的键依然包含所有polymer，即使有些value是空的
def parse(
    file_id: str, 
    mmcif_string: str, 
    polymer_types: Sequence[str] = PTYPE_LOOKUP.keys(), 
    catch_all_errors: bool = True
) -> ParsingResult:
    """Entry point, parses an mmcif_string.

    Args:
      file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
      mmcif_string: Contents of an mmCIF file.
      catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate.

    Returns:
      A ParsingResult.
    """
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        handle = io.StringIO(mmcif_string)
        structure = parser.get_structure("", handle)
        structure = _get_first_model(structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine all valid chains
        valid_chains = _get_all_valid_chains(parsed_info, polymer_types)
        if not any(valid_chains.values()):
            return ParsingResult(
                None, {(file_id, ""): ParseError("No chains found in this file.")}
            )
        
        mmcif_to_author_chain_id = dict(zip(
            parsed_info["_atom_site.label_asym_id"], 
            parsed_info["_atom_site.auth_asym_id"]
        ))

        # 新增的一个check
        valid_chains_copy = deepcopy(valid_chains)
        for ptype, chains in valid_chains_copy.items():
            for chain_id in chains.keys():
                if chain_id not in mmcif_to_author_chain_id.keys():
                    valid_chains[ptype].pop(chain_id)

        ptype_to_mmcif_chain_ids = {
            ptype: list(chains.keys()) 
            for ptype, chains in valid_chains.items()
        }
        
        # region: seq_start_num
        # Determine start numbers according to the internal mmCIF numbering scheme 
        # (likely but not guaranteed to be 1).
        flat_valid_chains: Mapping[ChainId, Sequence[Monomer]] = {}
        for chains in valid_chains.values():
            flat_valid_chains.update(chains)
        seq_start_num = {
            chain_id: min([monomer.num for monomer in seq])
            for chain_id, seq in flat_valid_chains.items()
        }
        # endregion

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # - mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # the authors / Biopython).
        # - seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
        seq_to_structure_mappings: MmcifObject.seqres_to_structure = {
            ptype: defaultdict(dict)
            for ptype in PTYPE_LOOKUP.keys()
        }
        for atom in _get_atom_site_generator(parsed_info):
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue

            valid_ptypes = [
                ptype 
                for ptype, chain_ids in ptype_to_mmcif_chain_ids.items() 
                if atom.mmcif_chain_id in chain_ids
            ]
            # 如果valid_ptypes非空，即atom的链存在于valid_chains中
            if valid_ptypes:
                # region: seq_idx
                seq_idx = (
                    int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                )
                # endregion
                # 如果seq_idx已经存在，即当前原子所属的残基已被记录，则跳过当前循环
                if seq_idx not in seq_to_structure_mappings[valid_ptypes[0]][atom.author_chain_id].keys():
                    # region: hetflag
                    hetflag = " "
                    if atom.hetatm_atom == "HETATM":
                        # Water atoms are assigned a special hetflag of W in Biopython. We
                        # need to do the same, so that this hetflag can be used to fetch
                        # a residue from the Biopython structure by id.
                        if atom.residue_name in ("HOH", "WAT"):
                            hetflag = "W"
                        else:
                            hetflag = "H_" + atom.residue_name
                    # endregion
                    # region: insertion_code
                    insertion_code = atom.insertion_code
                    if not _is_set(atom.insertion_code):
                        insertion_code = " "
                    # endregion
                    # region: position
                    position = ResiduePosition(
                        chain_id=atom.author_chain_id,
                        residue_number=int(atom.author_seq_num),
                        insertion_code=insertion_code,
                    )
                    # endregion
                    # region: current
                    for ptype in valid_ptypes:
                        current = seq_to_structure_mappings[ptype][atom.author_chain_id]
                        current[seq_idx] = ResidueAtPosition(
                            position=position,
                            name=atom.residue_name,
                            is_missing=False,
                            hetflag=hetflag,
                        )
                        seq_to_structure_mappings[ptype][atom.author_chain_id] = current
                    # endregion

        # Add missing residue information to seq_to_structure_mappings.
        for ptype, chains in valid_chains.items():
            for chain_id, seq_info in chains.items():
                author_chain_id = mmcif_to_author_chain_id[chain_id]
                current_mapping = seq_to_structure_mappings[ptype][author_chain_id]
                for idx, monomer in enumerate(seq_info):
                    if idx not in current_mapping:
                        current_mapping[idx] = ResidueAtPosition(
                            position=None,
                            name=monomer.id,
                            is_missing=True,
                            hetflag=" ",
                        )

        author_chain_to_sequence: MmcifObject.chain_to_seqres = {
            ptype: dict()
            for ptype in PTYPE_LOOKUP.keys()
        }
        # for ptype, chains in valid_chains.items():
        #     for chain_id, seq_info in chains.items():
        #         author_chain_id = mmcif_to_author_chain_id[chain_id]
        #         seq = []
        #         if ptype == "protein":
        #             for monomer in seq_info:
        #                 code = SCOPData.protein_letters_3to1.get(monomer.id, "X")
        #                 seq.append(code if len(code) == 1 else "X")
        #             seq = "".join(seq)
        #         if ptype == "rna":
        #             for monomer in seq_info:
        #                 code = RNA_LOOKUP.get(monomer.id, "X")
        #                 seq.append(code)
        #             seq = "".join(seq)
        #         if ptype == "dna":
        #             for monomer in seq_info:
        #                 code = DNA_LOOKUP.get(monomer.id, "X")
        #                 seq.append(code)
        #             seq = "".join(seq)
                    
        #         author_chain_to_sequence[ptype][author_chain_id] = seq
        entity_to_mmcif_chains = defaultdict(list)
        for chain_id, ent_id in zip(parsed_info["_struct_asym.id"], parsed_info["_struct_asym.entity_id"]):
            entity_to_mmcif_chains[ent_id].append(chain_id)

        entity_ids = parsed_info["_entity_poly.entity_id"]
        seqs = [seq.replace("\n", "") for seq in parsed_info["_entity_poly.pdbx_seq_one_letter_code_can"]]
        entity_id_to_seq = dict(zip(entity_ids, seqs))
            
        for ptype, chains in valid_chains.items():
            for chain_id in chains.keys():
                author_chain_id = mmcif_to_author_chain_id[chain_id]
                for entity_id, chain_ids in entity_to_mmcif_chains.items():
                    if chain_id in chain_ids:
                        seq = entity_id_to_seq[entity_id]
                        author_chain_to_sequence[ptype][author_chain_id] = seq
                

        # 这里structure和raw_string占据绝大部分内存，各占一半
        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=structure,
            chain_to_seqres=author_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            raw_string=parsed_info,
        )

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, "")] = e
        if not catch_all_errors:
            raise 
        return ParsingResult(mmcif_object=None, errors=errors)



def _get_first_model(structure: PdbStructure) -> PdbStructure:
    """Returns the first model in a Biopython structure."""
    return next(structure.get_models())

# region header

def _get_release_date(parsed_info: MmCIFDict) -> str:
    """Returns the oldest revision date."""
    release_date = "?"
    if "_pdbx_audit_revision_history.revision_date" in parsed_info.keys():
        release_date = min(parsed_info["_pdbx_audit_revision_history.revision_date"])
    else:
        logging.warning(
            "Could not determine release_date in %s", parsed_info["_entry.id"]
        )
    
    return release_date

def _get_resolution(parsed_info: MmCIFDict) -> float:
    resolution = 0.00
    for res_key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                resolution = float(raw_resolution)
            except ValueError:
                logging.info(
                    "Invalid resolution format in %s", parsed_info["_entry.id"]
                )

    return resolution

def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    # structure_method: 实验方法
    header["structure_method"] = ",".join(parsed_info["_exptl.method"]).lower()
    # release_date: 最早的revision日期
    # !Note: The release_date here corresponds to the oldest revision. We prefer to 
    # !use this for dataset filtering over the deposition_date.
    header["release_date"] = _get_release_date(parsed_info)
    # resolution: 分辨率, 三种数据项按顺序择其一; 默认为0.0
    header["resolution"] = _get_resolution(parsed_info)

    return header

# endregion

def _get_all_valid_chains(
    parsed_info: Mapping[str, Any],
    polymer_types: Sequence[str]
) -> Mapping[PType, Mapping[ChainId, Sequence[Monomer]]]:
    """Extracts polymer information for protein chains only.

    Args:
      parsed_info: _mmcif_dict produced by the Biopython parser.

    Returns:
      A dict mapping mmcif chain id to a list of Monomers.
    """
    # region: polymers  
    # Get polymer information for each entity in the structure.
    ent_ids = parsed_info["_entity_poly_seq.entity_id"]
    mon_ids = parsed_info["_entity_poly_seq.mon_id"]
    nums = parsed_info["_entity_poly_seq.num"]

    polymers = defaultdict(list)
    for ent_id, mon_id, num in zip(ent_ids, mon_ids, nums):
        polymers[ent_id].append(Monomer(id=mon_id, num=int(num)))
    # endregion

    # region: chem_comps  
    # Get chemical compositions. Will allow us to identify which of these polymers are proteins.
    chem_comps = dict(zip(parsed_info["_chem_comp.id"], parsed_info["_chem_comp.type"]))
    # endregion

    # region: entity_to_mmcif_chains
    # Get chains information for each entity. Necessary so that we can return a 
    # dict keyed on chain id rather than entity.
    entity_to_mmcif_chains = defaultdict(list)
    for chain_id, ent_id in zip(parsed_info["_struct_asym.id"], parsed_info["_struct_asym.entity_id"]):
        entity_to_mmcif_chains[ent_id].append(chain_id)
    # endregion

    # region: valid_chains
    # Identify and return the valid protein chains.

    get_valid_chains = partial(_get_valid_chains, polymers, chem_comps, entity_to_mmcif_chains)

    valid_chains = {}
    for ptype in polymer_types:
        valid_chains[ptype] = get_valid_chains(PTYPE_LOOKUP[ptype])
    # endregion

    return valid_chains

def _get_valid_chains(
    polymers: Mapping[str, Sequence[Monomer]], 
    chem_comps: Mapping[str, str], 
    entity_to_mmcif_chains: Mapping[str, list[ChainId]],
    term: str
) -> Mapping[ChainId, Sequence[Monomer]]:
    """
    term: "peptide", "RNA", "DNA"
    """
    valid_chains = {}
    for entity_id, seq_info in polymers.items():
        chain_ids = entity_to_mmcif_chains[entity_id]
        if any(
            [
                term in chem_comps[monomer.id]
                for monomer in seq_info
            ]
        ):
            for chain_id in chain_ids:
                valid_chains[chain_id] = seq_info
    return valid_chains

def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in (".", "?")

def _get_atom_site_generator(parsed_info: MmCIFDict) -> Generator[AtomSite, None, None]:
    """Returns list of atom sites; contains data not present in the structure."""
    return (
        AtomSite(*site)
        for site in zip(  # pylint:disable=g-complex-comprehension
            parsed_info["_atom_site.label_comp_id"],
            parsed_info["_atom_site.auth_asym_id"],
            parsed_info["_atom_site.label_asym_id"],
            parsed_info["_atom_site.auth_seq_id"],
            parsed_info["_atom_site.label_seq_id"],
            parsed_info["_atom_site.pdbx_PDB_ins_code"],
            parsed_info["_atom_site.group_PDB"],
            parsed_info["_atom_site.pdbx_PDB_model_num"]
        )
    )
