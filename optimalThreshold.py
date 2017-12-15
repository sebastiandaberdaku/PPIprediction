# This script computes the best SVM threshold value that maximises the F1 score.
# Please set the path variable to the current location of the validation samples before running the script.

import numpy as np
from glob import glob
from os.path import basename
import matplotlib.pyplot as plt
from math import copysign
from sklearn.metrics.classification import precision_recall_fscore_support
from multiprocessing import Process, Manager


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    
    return out

def optimal_threshold(path, cat, it, bu):

    files = glob("%s*_%s.pdb" % (path, bu))
    
    all_scores = []
    all_truth = []
    pdb_ids = []
    
    manager = Manager()
    
    for f in sorted(files) :
   
        filename = f
        pdb_id = basename(filename)[:-4]
        print pdb_id
        f_patch_score  = ("%s%s_patch_score.txt" % (path, pdb_id))
        f_patch_truth  = ("%s%s_patch_truth.txt" % (path, pdb_id))
    
        with open(f_patch_score) as score, open(f_patch_truth) as truth :
            patch_score = [float(x) for x in score.readlines()]
            patch_truth = [int(x)   for x in truth.readlines()]
        
        pdb_ids.append(pdb_id)
    
        all_scores.append(patch_score)
        all_truth.append(patch_truth)
    
    all_thresholds = np.unique(np.concatenate(all_scores))
    
    precision = manager.list([0 for _ in all_thresholds])
    recall    = manager.list([0 for _ in all_thresholds])
    
    def compute(indices, all_truth ,all_scores, pdb_ids, all_thresholds, precision, recall):
        print all_thresholds[indices[0]]
        for t in indices :
            for i in xrange(len(pdb_ids)) :
                p, r, _, _ = precision_recall_fscore_support(all_truth[i], [copysign(1, x - all_thresholds[t]) for x in all_scores[i]], average='binary')
                precision[t] += p
                recall[t] += r
            precision[t] /= len(pdb_ids)
            recall[t] /= len(pdb_ids)
    
    
    # Parallel(n_jobs=12)(delayed(compute)(t, all_truth ,all_scores, pdb_ids, all_thresholds, precision, recall) for t in xrange(len(all_thresholds)))
    L = chunkIt(range(len(all_thresholds)), 1000)
    job = [Process(target=compute, args=(indices, all_truth ,all_scores, pdb_ids, all_thresholds, precision, recall)) for indices in L]
    _ = [p.start() for p in job]
    _ = [p.join() for p in job]
    
    def fscore(p, r):
        if p + r > 0 : 
            return 2 * p * r /(p+r)
        else :
            return 0
    
    thresholds_f1scores = [(all_thresholds[t], fscore(precision[t], recall[t])) for t in xrange(len(all_thresholds))]
    
    print
    best_pair = max(thresholds_f1scores, key=lambda x:x[1])
    print ("Maximum F1 obtained for threshold: %s" % str(best_pair))
    
    plt.figure(figsize=(10, 10), dpi=1200)
    plt.xlim([all_thresholds[0], all_thresholds[-1]])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold values')
    plt.ylabel('F1 score')
    plt.title('Threshold versus F1 scores')
    plt.plot(all_thresholds, [a[1] for a in thresholds_f1scores], color='navy', linestyle='solid', linewidth=2)
    plt.scatter(best_pair[0], best_pair[1], marker='x', color='red', s=40)
    plt.plot([best_pair[0], best_pair[0]], [0, best_pair[1]], linestyle="dotted", linewidth=1, color='red')
    plt.plot([all_thresholds[0], best_pair[0]], [best_pair[1], best_pair[1]], linestyle="dotted", linewidth=1, color='red')
    plt.annotate("(%.4f, %.4f)" % (best_pair[0], best_pair[1]), xy=(best_pair[0], best_pair[1]), xytext=(-140, 30),
        textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    # plt.legend()
    plt.savefig("%s_%s_%s_threshold_for_best_F1_score.pdf" % (cat, it, bu), dpi=1200, bbox_inches='tight')
    plt.close()
    plt.clf()


for cat in ["A", "AB", "EI", "ER", "ES", "OG", "OR", "OX"]:
    for it in ["r", "l"] :
        for bu in ["b", "u"] :
            print ("Searching for the optimal threshold for: (%s, %s, %s)\n\n" % (cat, it, bu))
            path = "/media/sebastian/Storage/3DZD_interface_prediction_db5/benchmark5/" + cat + "/train_" + it + "/"
            print path
            optimal_threshold(path, cat, it, bu)
            print ("*" * 255)

