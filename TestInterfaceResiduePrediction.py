# This script runs the IF algorithm for outlier detection to remove false positive patches and maps the predicted LSPs on the underlying residues.
# The results are compared to other predictor software packages.
# Please remember to set the path variable to the current location of the test set.

import numpy as np
from sklearn.neighbors.kd_tree import KDTree
from glob import glob
from math import copysign
from sklearn.ensemble import IsolationForest
from os.path import basename
from os import path, makedirs

from Bio.PDB.PDBParser import PDBParser
p = PDBParser(QUIET=True, PERMISSIVE=True)

import re
_hydrogen = re.compile("[123 ]*H.*")

def isHydrogen(atm):
    return _hydrogen.match(atm.get_id())

def isHETATM(atm):
    return atm.get_parent().get_id()[0] != " "


#######################
# import pickle
#######################

outlier_fraction = {}

outlier_fraction[("AB", "l", "b")] = 0.00
outlier_fraction[("AB", "l", "u")] = 0.00
outlier_fraction[("AB", "r", "b")] = 0.44
outlier_fraction[("AB", "r", "u")] = 0.47

outlier_fraction[("A", "l", "b")] = 0.00
outlier_fraction[("A", "l", "u")] = 0.15
outlier_fraction[("A", "r", "b")] = 0.12
outlier_fraction[("A", "r", "u")] = 0.26

outlier_fraction[("EI", "l", "b")] = 0.34
outlier_fraction[("EI", "l", "u")] = 0.49
outlier_fraction[("EI", "r", "b")] = 0.49
outlier_fraction[("EI", "r", "u")] = 0.50

outlier_fraction[("ER", "l", "b")] = 0.31
outlier_fraction[("ER", "l", "u")] = 0.40
outlier_fraction[("ER", "r", "b")] = 0.50
outlier_fraction[("ER", "r", "u")] = 0.50

outlier_fraction[("ES", "l", "b")] = 0.23
outlier_fraction[("ES", "l", "u")] = 0.00
outlier_fraction[("ES", "r", "b")] = 0.00
outlier_fraction[("ES", "r", "u")] = 0.00

outlier_fraction[("OG", "l", "b")] = 0.43
outlier_fraction[("OG", "l", "u")] = 0.43
outlier_fraction[("OG", "r", "b")] = 0.50
outlier_fraction[("OG", "r", "u")] = 0.50

outlier_fraction[("OR", "l", "b")] = 0.48
outlier_fraction[("OR", "l", "u")] = 0.43
outlier_fraction[("OR", "r", "b")] = 0.00
outlier_fraction[("OR", "r", "u")] = 0.00

outlier_fraction[("OX", "l", "b")] = 0.36
outlier_fraction[("OX", "l", "u")] = 0.50
outlier_fraction[("OX", "r", "b")] = 0.50
outlier_fraction[("OX", "r", "u")] = 0.45

threshold = {}

threshold[("AB", "l", "b")] = 0.3301
threshold[("AB", "l", "u")] = 0.8920
threshold[("AB", "r", "b")] = 1.1631
threshold[("AB", "r", "u")] = 0.5813

threshold[("A", "l", "b")] = -0.2582
threshold[("A", "l", "u")] = -2.0376
threshold[("A", "r", "b")] = 1.5638
threshold[("A", "r", "u")] = 1.5178

threshold[("EI", "l", "b")] = 1.1852
threshold[("EI", "l", "u")] = 0.8166
threshold[("EI", "r", "b")] = 1.0026
threshold[("EI", "r", "u")] = 0.7007

threshold[("ER", "l", "b")] = 1.1596
threshold[("ER", "l", "u")] = 0.9944
threshold[("ER", "r", "b")] = 0.6674
threshold[("ER", "r", "u")] = 0.8774

threshold[("ES", "l", "b")] = 0.8044
threshold[("ES", "l", "u")] = 0.5547
threshold[("ES", "r", "b")] = 0.4367
threshold[("ES", "r", "u")] = 0.2270


threshold[("OG", "l", "b")] = 1.0008
threshold[("OG", "l", "u")] = 0.9988
threshold[("OG", "r", "b")] = 0.1360
threshold[("OG", "r", "u")] = -0.0920

threshold[("OR", "l", "b")] = 0.9965
threshold[("OR", "l", "u")] = 0.6799
threshold[("OR", "r", "b")] = 0.7203
threshold[("OR", "r", "u")] = 0.4861

threshold[("OX", "l", "b")] = 0.9011
threshold[("OX", "l", "u")] = 0.4963
threshold[("OX", "r", "b")] = 0.6990
threshold[("OX", "r", "u")] = 0.7840

n_iterations = 100
mapping_distance = 5.0 

def compute_average_scores(pdb_path, cat, it, bu):
    files = glob("%s*_%s_%s.pdb" % (pdb_path, it, bu))
    
    for pdb_filename in sorted(files) :
        pdb_id = basename(pdb_filename)[:-4]
        
        pdb_patch_coord = ("%s%s_patch_coord.txt" % (pdb_path, pdb_id))
        pdb_patch_score = ("%s%s_patch_score.txt" % (pdb_path, pdb_id))
      
        with open(pdb_patch_coord) as coord, open(pdb_patch_score) as score:
            patch_coord = [[float(x) for x in a.split()] for a in coord.readlines()]
            patch_score = [float(x) - threshold[(cat, it, bu)] for x in score.readlines()]
 
        min_v = min(patch_score)
        max_v = max(patch_score)
         
        patch_score_scaled = [(lambda x: -(x / min_v) if x < 0 else (x / max_v))(x) for x in patch_score]
     
        X = np.array([a[0] for a in zip(patch_coord, patch_score_scaled) if a[1] >= 0])
        X_weights = np.array([x for x in patch_score_scaled if x >= 0])
        
        pdb_structure = p.get_structure(pdb_id, pdb_filename)
        atoms = np.array([atm.get_coord() for atm in pdb_structure.get_atoms() if not isHydrogen(atm) and not isHETATM(atm)])
        atoms_tree = KDTree(atoms)     
        
        residues_coord = {}
        for residue in pdb_structure.get_residues() :
            for atm in residue :
                residues_coord[tuple(atm.get_coord())] = residue
    
        average_residues_scores = {residue : 0 for residue in pdb_structure.get_residues()}

        # since the isollation forest algorithm is random, we run it several times to assess the average performance of the method
        
        if outlier_fraction[(cat, it, bu)] : reps = n_iterations
        else : reps = 1
        
        for iteration in xrange(reps) :
            print "Running iteration %d of %d" % (iteration + 1, reps)
            if outlier_fraction[(cat, it, bu)] : 
                forest = IsolationForest(contamination=outlier_fraction[(cat, it, bu)], n_jobs=-1)
                forest.fit(X, sample_weight=X_weights)
         
                prediction_isolation_forest = forest.predict(patch_coord)
                patch_pred_no_outliers = [copysign(1, x) for x in prediction_isolation_forest]
            else : 
                patch_pred_no_outliers = [copysign(1, x) for x in patch_score]
            # here we map the patch predictions on the underlying residues
            for i in xrange(len(patch_coord)) : # for each patch
                # if it was predicted as non-interface continue to the next
                if patch_pred_no_outliers[i] < 0 : continue 
                # multiple residues can be underneath a given patch, we do not want to consider the same residue more than once
                marked_residues = set() 
                # get all atoms within mapping_distance from the given patch center
                indexes = atoms_tree.query_radius([patch_coord[i]], r=mapping_distance, count_only = False, return_distance=True, sort_results = True)
                for ind in zip(indexes[0][0], indexes[1][0]) :
                    # which residue does the current atom belong to?
                    current_res = residues_coord[tuple(atoms[ind[0]])] 
                    # if already considered continue to the next
                    if current_res in marked_residues : continue 
                    # increase the score of the current residue
                    average_residues_scores[current_res] += 1 / (1.0 + ind[1]) # patch_pred_no_outliers[i] / (1.0 + ind[1])
                    # mark as seen for the current patch
                    marked_residues.add(current_res)
             
        average_residues_scores.update((x, y / reps) for x, y in average_residues_scores.items())
        
        residues_with_scores = [(lambda x, y : (x[2], str(x[3][1]) + x[3][2], y))(residue.get_full_id(), score) for residue, score in average_residues_scores.items()]
        residues_with_scores.sort(key=lambda x : x[1])
        residues_with_scores.sort(key=lambda x : x[0])

        prediction_path = pdb_path + "our_prediction/"
        if not path.exists(prediction_path) : makedirs(prediction_path)
        print pdb_id
        with open("%s%s_residue_scores.txt" % (prediction_path, pdb_id), "wb") as output_residue_scores :
            for r in residues_with_scores :
                output_residue_scores.write("%s;%s;%f\n" %(r[0], r[1], r[2]))

for cat in ["A", "AB", "EI", "ER", "ES", "OG", "OR", "OX"]:
    for it in ["l", "r"] :
        for bu in ["u", "b"] :
            print ("Running interface prediction for configuration: (%s, %s, %s)\n\n" % (cat, it, bu))
            pdb_path = "/media/sebastian/Storage/3DZD_interface_prediction_db5/benchmark5/" + cat + "/test_" + it + "/"
            compute_average_scores(pdb_path, cat, it, bu)
            print ("*" * 255)
 
 
