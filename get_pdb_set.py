"""
Created on 24.11.21

@author: maxjansen
"""
from pathlib import Path
import re
import glob
from itertools import chain

data_dir= Path('../data/')
pdb_list = []
for filepath in data_dir.glob("*.slurm"):
    f = open(filepath, "r")
    for line in f.readlines():
        if line.startswith("pdb_id"):
            line = line.split("pdb_id=(")[1]
            line = line.split(")")[0]
            line = re.sub('"', '', line)
            pdb_list.append(list(line.split(" ")))



unnested_pdb = list(chain.from_iterable(pdb_list))
unique_pdb = (list(set(list(unnested_pdb))))


############
test_list = []
test_f = open(data_dir / "testing_ppi.txt", "r")
for line in test_f.readlines():
    line = line.split("\n")[0]
    test_list.append(line)


#######
unique_pdb_set = set(unique_pdb)
dmasif_test_set = set(test_list)


training_list = []
training_f = open(data_dir / "training_ppi.txt", "r")
for line in training_f.readlines():
    line = line.split("\n")[0]
    training_list.append(line)
dmasif_train_set = set(training_list)


print(list(sorted(dmasif_train_set.intersection(unique_pdb_set))))







