#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 00:14:15 2021

@author: maxjansen
"""
# Read all the vtk/npy files, Read all the pdb files

# Get first chains for each pdb stem and their pdb, get surface for opposing chains
# Do the inverse two double the dataset size


# End goal: unique list of AA's near the interface with the closest surface points

from tqdm import tqdm
from Bio.PDB import *
import glob
from pathlib import Path
from scipy.spatial import distance
import numpy as np
import pandas as pd

res2num = {"ALA": 0, "ASX": 1, "CYS": 2, "ASP": 3, "GLU": 4, "PHE": 5, "GLY": 6, "HIS": 7,
           "ILE": 8, "LYS": 9, "LEU": 10, "MET": 11, "ASN": 12, "PRO": 13, "GLN": 14,
           "ARG": 15, "SER": 16, "THR": 17, "SEC": 18, "VAL": 19, "TRP": 20, "XAA": 21,
           "TYR": 22, "GLX": 23}


def keep_inter_res(fname, inter_coords, center):
    """Loads a pdb and returns coord, atom type, AA type and AA resID
    Will only keep atoms within a distance threshold to interface surface points
    """
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    dist_thres = 2

    coords = []
    types = []
    atom_residues = []
    resseqs = []
    parent_list = []
    for atom in atoms:
        coord_list = atom.get_coord()
        dist_boolean = np.any(distance.cdist([coord_list],
                                             inter_coords, 'euclidean') < dist_thres)
        if dist_boolean == True:
            parent_list.append(atom.get_parent())
            # interesting to save how many atoms from given residue (AA parent),
            # were within distance metric, to assess ML model performance

    parent_list = set(parent_list)
    parent_list = list(parent_list)
    for parent in parent_list:
        atoms = parent.get_atoms()
        for atom in atoms:
            if str(atom.get_name()) == 'CA':
                coords.append(atom.get_coord())
                atom_residues.append(atom.get_parent().get_resname())
                resseqs.append(atom.get_parent().get_full_id()[3][1])
    if len(coords) != 0:
        coords = np.stack(coords)
    else:
        pass

    return {"xyz": coords, "AA_types": atom_residues,
            "SeqID": resseqs}


def get_close_points(ca_dict, surf_dict):
    """Returns the 100 closest points to interface C-alpha atoms
    # Loop goes through AA Ca and calculates distance to all points.
    # Picks closest 100 point coords and feats after sorting.
    Changed from original 20 points to 100"""

    types = []
    feats = []
    for i, ca_coord in enumerate(ca_dict['xyz']):
        dist_array = distance.cdist([ca_coord], surf_dict['xyz'],
                                    'euclidean')[0]
        # Consider adding distance as a feat!
        dist_index = np.argsort(dist_array)
        selected_dists = dist_array[dist_index][0:100]
        selected_coords = surf_dict['xyz'][dist_index][0:100]
        selected_feats = surf_dict['feats'][dist_index][0:100]
        selected_feats = np.column_stack((selected_feats, selected_dists))
        selected_feats = np.ravel(selected_feats)
        feats.append(selected_feats)

        types.append(res2num[ca_dict['AA_types'][i]])

    # =============================================================================
    # Uncomment this if you want to save point patch per AA, good in pymol
    #           seq = ca_dict['SeqID'][i]
    #           select_dir = Path("../dataset/aa_points")
    #           np.save(select_dir / (AA_type + str(seq) + "_predcoords.npy"), selected_coords)
    #           np.save(select_dir / (AA_type + str(seq) + "_predfeatures.npy"), selected_feats)
    #
    # =============================================================================
    if len(feats) != 0:
        feats_array = np.stack(feats)
        types_array = np.zeros((len(types), len(res2num)))
        for i, t in enumerate(types):
            types_array[i, t] = 1.0
    else:
        feats_array = []
        types_array = []

    return {"feat_data": feats_array, "target_type": types_array}


def load_surface_np(pdb_id_chain):
    surf_path = Path('../dataset/surfaces')
    coord_array = np.load(surf_path / str(pdb_id_chain +
                                          "_predcoords.npy"))
    feat_array = np.load(surf_path / str(pdb_id_chain +
                                         "_predfeatures.npy"))
    return {"xyz": coord_array, "feats": feat_array}


def surface_interface(surf_dict):
    """Takes dict of surface points coordinates and features and only returns
    features with interface feature == True"""

    interface_idx = np.where(surf_dict['feats'][:, 25] == 1)[0]
    interface_coords = surf_dict['xyz'][interface_idx]

    return interface_coords


def convert_pdbs(pdb_dir):  # , inter_coords):
    print("Converting PDBs")
    points_dict = 0
    counter = 0
    # Every pdb_id_chain in dir will be on the pdb side
    for p in tqdm(pdb_dir.glob("*.pdb")):
        atom_chain = p.stem
        pdb_id = atom_chain.split('_')[0]
        all_stem = pdb_dir.glob(pdb_id + "*.pdb")

        # Find the corresponding chain(s) for the same pdb id, get surface points
        for i in all_stem:
            if i.stem != atom_chain:
                surface_chain = i.stem
                surf_dict = load_surface_np(surface_chain)
                inter_coords = surface_interface(surf_dict)
                protein = keep_inter_res(p, inter_coords, center=False)
                if len(protein) == 0:
                    pass
                else:
                    close_points = get_close_points(protein, surf_dict)
                    if len(close_points) == 0:
                        pass
                    elif isinstance(close_points['feat_data'], np.ndarray):
                        if points_dict == 0:
                            points_dict = close_points
                        else:
                            points_dict["feat_data"] = np.concatenate(
                                (points_dict["feat_data"],
                                 close_points["feat_data"]), axis=0)
                            points_dict["target_type"] = np.concatenate(
                                (points_dict["target_type"],
                                 close_points["target_type"]), axis=0)
                    else:
                        pass

    return points_dict


data_dict = convert_pdbs(Path('../dataset/pdbs/'))

# Uncomment to save dataset, including distance

np.save("../dataset/dist_train_data.npy", data_dict['feat_data'])
np.save("../dataset/dist_train_target.npy", data_dict['target_type'])

