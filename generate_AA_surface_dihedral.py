"""
Created on 27.10.21 13:39

@author: maxjansen


Main parts:
1. Load pdbs of PPIS
2. Iterate and plit into target and binder side (can be vice versa to double dataset)
3. Get binder side interface
4. Get nearest AAs on binder side
5. Get nearest target AA to binder
 For a given binder side AA, get patch and close AA
4. Perform Zander calculations on
"""

from tqdm import tqdm
from Bio.PDB import *
import glob
from pathlib import Path
from scipy.spatial import distance
import numpy as np
import pandas as pd
import argparse
from aa_funcs import *
from angle_funcs import *
import logging

def make_ppi_list(pdb_dir):
    """Finds pdb's in a given directory. Returns a list relevant of PPI pairs. Does this based on splitting pdb_id and
    chain and finding non-redundant matches. Also works for complexes with more than two chains."""

    print("loading PDBs, selecting pairs")
    counter=0
    pair_list = []
    # Iterate to find first of pair
    for p in tqdm(pdb_dir.glob("*.pdb")):
        atom_chain = p.stem
        pdb_id = atom_chain.split('_')[0]
        all_stem = pdb_dir.glob(pdb_id + "*.pdb")
        # Iterate over all again to find second of pair with same pdb_id and different chain
        for i in all_stem:
            if i.stem != atom_chain:
                surface_chain = i.stem
                pair_list.append([atom_chain, surface_chain])

    return pair_list

def list_iterate(pair_list, surf_dir):
    """Iterate over list to perform separate functions on target (surface) and binder (pdb) chain."""
    for ppi in tqdm(pair_list):
        surface_chain = ppi[1]
        surf_dict = load_surface_np(surface_chain, surf_dir)
        inter_coord =








def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Arguments get parsed via --commands')
    parser.add_argument('-pdb_dir', metavar='protein_directory', type=str, default='../dataset/pdbs/',
                        help='Specify directory to load pdbs of PPIs. Default is ../dataset/pdbs')
    parser.add_argument('-path_out', metavar='output_directory', type=str, default='../dataset/',
                        help='Write where you want the output target and feature data to go.')
    parser.add_argument('-file_out', metavar='output_filenames', type=str, default='dihedral_train',
                        help='Write where you want the output target and feature data to go.')
    parser.add_argument('-vizpatch_out', metavar='see_patches', type=str, default='no',
                        help='Do you want all target surface patches per AA for the first PPI in the list? For pyMol.')
    parser.add_argument('-surf_dir', metavar='dmasif_surface_dir', type=str, default='../dataset/denser_surfaces',
                        help='Specify where dMaSIF surface predictions are to generate dataset.')
    parser.add_argument('-n_points', metavar='N_surf_points', type=int, default=100,
                        help='Number of surface points near AA')
    parser.add_argument('-n_augment', metavar='patch_augmentation', type=int, default=1,
                        help='N patches per AA. N=1 gives nearest n_points to AA Cbeta.')
    parser.add_argument('-aa_angles', metavar='')
    args = parser.parse_args()


    return args


def main():
    pass


if __name__ == '__main__':
    args = parse_arguments()
    # Read all pdb files to just get a list of constituent chains of all PPIs
    pair_list = make_ppi_list(Path(args.pdb_dir))
    pair_list.sort()
    # Iterate thgou
    list_iterate(pair_list, Path(args.surf_dir))
    main()


