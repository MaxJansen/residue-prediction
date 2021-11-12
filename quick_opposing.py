#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04.11.21

@author: maxjansen
"""

# Same as many generate_AA_surface scripts, but I calculate full dihedrals, using Zander's scripts

# Get closest AA on target and Binder side and calculate these angles

# Not advanced, just a quick run.
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
from zander import *
import itertools

res2num = {"ALA": 0, "ASX": 1, "CYS": 2, "ASP": 3, "GLU": 4, "PHE": 5, "GLY": 6, "HIS": 7,
           "ILE": 8, "LYS": 9, "LEU": 10, "MET": 11, "ASN": 12, "PRO": 13, "GLN": 14,
           "ARG": 15, "SER": 16, "THR": 17, "SEC": 18, "VAL": 19, "TRP": 20, "XAA": 21,
           "TYR": 22, "GLX": 23}


def vectors_angle(v1, v2):
    """ Gives angle between two vectors. Gets unit vectors first.
    Max angle= 180 degrees (see cosine). Consider larger angles
    for side-chains that are flipped in."""
    v1_u= v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.dot(v1_u,v2_u))

def atom_loader(fname):
    """Loads pdb and returns atoms in a useful format."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()
    return atoms

def res_loader(fname):
    """Loads pdb and returns atoms in a useful format."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", fname)
    residues = structure.get_residues()
    return residues

def parent_finder(atoms, inter_coords, dist_thres, closest):
    """Takes atoms, a distance threshold and an interface surface.
    Returns a list of parent residues for atoms within distance threshold."""
    parent_list = []
    n_points = closest
    # print(inter_coords)
    # print(type(inter_coords))
    for atom in atoms:
        coord_list = atom.get_coord()
        # Keeps all within a given distance threshold, does not have a min or max number
        if closest == False:
            dist_boolean = np.any(distance.cdist([coord_list],
                                                 inter_coords, 'euclidean') < dist_thres)
            if dist_boolean == True:
                parent_list.append(atom.get_parent())
                # interesting to save how many atoms from given residue (AA parent),
                # were within distance metric, to assess ML model performance
        # Consider only keeping the closest N points
        else:
             print("When selecting residues/parents, 'closest' variable must be an integer or 'False'")

    return list(set(parent_list))

def get_res_coord(residues):
    """Returns array of Carbon alpha coordinates for all residues from a structure."""
    arr = np.empty((0,3), int)
    for i in residues:
        try:
            arr = np.append(arr, [i["CA"].get_coord()], axis=0)
        except KeyError:
            pass
    return(arr)

def keep_target_res(fname, c_coord):
    """Loads target pdb and binder carbon coord. Returns coord, atom type, AA type and AA resID
     Will only keep closest target residue to binder carbon coord."""
    residues = list(res_loader(fname))
    arr = get_res_coord(residues)
    sel_coord, sel_dist, dist_index = dist_selector(c_coord, arr, 1)
    return residues[dist_index[0]]

def get_ncac(residue):
    """"Takes residue class after loading structure via biopython. Returns N, Ca and C coordinates in numpy array"""
    ncac = np.stack([residue["N"].get_coord(), residue["CA"].get_coord(), residue["C"].get_coord()])
    return ncac


def keep_inter_res(fname, inter_coords, center):
    """Loads a pdb and a surface and returns coord, atom type, AA type and AA resID
    Will only keep atoms from an opposing interface chain within a distance
    threshold to interface surface points.
    """
    # Load atoms from pdb and assign variables
    atoms = atom_loader(fname)
    dist_thres = 2
    coords, residues, resseqs, vectors, backbone = [], [], [], [], []
    # Iterate over all atoms and keep those close to surface points
    parent_list = parent_finder(atoms, inter_coords, dist_thres, False)

    # Iterate through residues at interface
    for parent in parent_list:
        parent_type = parent.get_resname()
        # Determine whether parent is glycine or not, because Gly has no CB
        if parent_type == 'GLY':
            gly_bool = 1
        else:
            gly_bool = 0

        # For parent residue, function will append res and CB (or CA) traits to lists
        coords, residues, resseqs, vectors, backbone = residue_append(
            parent, coords, residues, resseqs, vectors, gly_bool, backbone)

    # Check whether there is a vector for each c_coord and vice versa
    if (len(coords) != 0) and (len(coords) == len(vectors)):
        coords = np.stack(coords)
        vectors = np.stack(vectors)
        backbone = np.stack(backbone)
    else:
        print("No vectors for all point coordinates")
        pass

    return {"xyz": coords, "AA_types": residues,
            "SeqID": resseqs, "res_vector": vectors, "coord_list": backbone}

def residue_append(parent, coords, residues, resseqs, vectors, gly_bool, backbone):
    """Gets called when iterating over residues. Returns name, res_id, vector between
    main carbon atoms (O for glycine), and coordinates of main carbons"""
    # Call function once for vector and coords
    vector_coord = res_vector_coord(parent, gly_bool)

    # These are the backbone atoms, check if you have three
    if len(vector_coord[2]) == 3:
        #Only append new information from parent residue if valid c atom
        if isinstance(vector_coord[1], np.ndarray):
            vectors.append(vector_coord[0])
            # Will give CB coordinates for all other than Gly, and CA for Gly
            coords.append(vector_coord[1])
            residues.append(parent.get_resname())
            resseqs.append(parent.get_full_id()[3][1])
            # Backbone means N, C alpha and C coord
            backbone.append(vector_coord[2])
    else:
        "Not enough backbone atoms"

    return coords, residues, resseqs, vectors, backbone

def res_vector_coord(parent, gly_bool):
    """Returns vector between main Carbon atoms (points at Cb) for all
    non-Gly input residues. Returns vector pointing at extended CB for Gly.
    Also returns coordinate of 'main' carbon atom (CB)."""

    atoms = parent.get_atoms()
    # backbone_ls is a list with coordinates, ncac = N, Ca, C.
    backbone_ls = []

    for atom in atoms:
        if str(atom.get_name()) == 'N':
            n_coord = atom.get_coord()
            backbone_ls.append(n_coord)
        elif str(atom.get_name()) == 'CA':
            ca_coord = atom.get_coord()
            backbone_ls.append(ca_coord)
            if str(gly_bool == 1):
                ncac = get_ncac(parent)
                N, CA, C = ncac[0], ncac[1], ncac[2]
                main_coord = extend(C, N, CA, 1.522, 1.927, -2.143)
        elif str(atom.get_name()) == 'C':
            cback_coord = atom.get_coord()
            backbone_ls.append(cback_coord)
        elif str(atom.get_name()) == 'CB':
            main_coord = atom.get_coord()
        else:
            pass
    # Checked len of vectors, all around 1.5 Angstrom
    vector = main_coord - ca_coord
    return vector, main_coord, backbone_ls

def dist_selector(c_coord, points, n_points):
    """Selects closest N points to a given c_coord. Returns the coordinates, distances and index of these points."""
    dist_array = distance.cdist([c_coord], points,
                                'euclidean')[0]
    dist_index = np.argsort(dist_array)
    selected_dists = dist_array[dist_index][0:n_points]
    selected_coords = points[dist_index][0:n_points]
    return selected_coords, selected_dists, dist_index

def get_close_points(c_dict, surf_dict, surface_chain):
    """Returns the 100 closest points to interface C-alpha atoms
    # Loop goes through AA main C and calculates distance to all points.
    # Picks closest 100 point coords and feats after sorting.
    Changed from original 20 points to 100"""

    types = []
    feats = []
    n_points = 100

    # Iterate through every main carbon for interface residues
    for i, c_coord in enumerate(c_dict['xyz']):
        # Get the closest residue on the opposite side (target side), use res to get target ncac coords
        close_t_res = keep_target_res(surface_chain, c_coord)
        t_ncac = get_ncac(close_t_res)
        tN, tCA, tC = t_ncac[0], t_ncac[1], t_ncac[2]
        tCB = extend(tC, tN, tCA, 1.522, 1.927, -2.143)

        # Get distance to all surface points and select closest points
        selected_coords, selected_dists, dist_index = dist_selector(c_coord, surf_dict['xyz'], n_points)

        # Get main carbon vector for residue and calculate angles to closest points
        c_vector = c_dict['res_vector'][i]
        point_c_vectors = selected_coords - c_coord
        # Angles here are most similar to phi
        angles = np.array([vectors_angle(c_vector, i) for i in point_c_vectors])
        # Get backbone atoms for additional angles
        ncac = c_dict["coord_list"][i]
        N, CA, C = ncac[0], ncac[1], ncac[2]
        CB = extend(C, N, CA, 1.522, 1.927, -2.143)
        # Get an angle similar to theta (backbone nitrogen) for each point
        thetas = np.array([to_dih(N, CA, c_coord, i) for i in point_c_vectors])
        dist_ref = to_len(tCB, CB)
        omega_ref = to_dih(tCA, tCB, CB, CA)
        theta_ref = to_dih(tN, tCA, tCB, CB)
        phi_ref = to_ang(tCA, tCB, CB)

        # Select loaded features for selected points, add distance and angle
        selected_feats = surf_dict['feats'][dist_index][0:n_points]
        # Get the carbon angles ready for appending
        res_angles = np.tile([dist_ref, omega_ref, theta_ref, phi_ref], n_points).reshape(n_points,4)
        selected_feats = np.column_stack((selected_feats, selected_dists, angles, thetas))
        selected_feats = np.column_stack((selected_feats, res_angles))
        selected_feats = np.ravel(selected_feats)
        feats.append(selected_feats)
        types.append(res2num[c_dict['AA_types'][i]])

    # =============================================================================
    #Uncomment this if you want to save point patch per AA, good in pymol
        # seq = c_dict['SeqID'][i]
        # AA_type = c_dict['AA_types'][i]
        # select_dir = Path("../dataset/aa_points/dense20")
        # selected_feats = selected_feats.reshape(n_points,28)
        # np.save(select_dir / (AA_type + str(seq) + "_predcoords.npy"), selected_coords)
        # np.save(select_dir / (AA_type + str(seq) + "_predfeatures.npy"), selected_feats)

    # =============================================================================
    if len(feats) != 0:
        feats_array = np.stack(feats)
        types_array = np.zeros((len(types), len(res2num)))
        for i, t in enumerate(types):
            types_array[i, t] = 1.0
    else:
        feats_array = []
        types_array = []
        print("No Features!")
    return {"feat_data": feats_array, "target_type": types_array}


def load_surface_np(pdb_id_chain):
    surf_path = Path('../dataset/denser_surfaces')
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
        print(p)
        # Find the corresponding chain(s) for the same pdb id, get surface points
        for i in all_stem:
            if i.stem != atom_chain:
                surface_chain = i.stem
                print(surface_chain)
                surf_dict = load_surface_np(surface_chain)
                inter_coords = surface_interface(surf_dict)
                protein = keep_inter_res(p, inter_coords, center=False)
                target = pdb_dir / str(i.stem + ".pdb")
                if len(protein) == 0:
                    pass
                else:
                    # Only get target res here, not earlier.
                    close_points = get_close_points(protein, surf_dict, target)
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
# from temp import *
# print(dist_selector.__doc__)
# Uncomment to save dataset, including distance
np.save("../dataset/dihedral_train_data.npy", data_dict['feat_data'])
np.save("../dataset/dihedral_train_target.npy", data_dict['target_type'])
