# This script computes the best contamination value for the Isolation Forest algorithm that maximises the F1 score.
# Please set the path variable to the current location of the validation samples before running the script.

from glob import glob
from os.path import basename
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics.classification import precision_recall_fscore_support
from multiprocessing import Manager
from joblib import Parallel, delayed

n_iterations = 100

def compute_scores(o, outlier_fractions, n_iterations, pdb_ids, all_truth, all_coord, all_X, all_X_weights, precision, recall):
    print outlier_fractions[o]
    forest = IsolationForest(contamination=outlier_fractions[o], n_jobs=-1)
    for i in xrange(len(pdb_ids)) :
        print pdb_ids[i]
        current_precision = 0
        current_recall = 0
        for _ in xrange(n_iterations) :
            if not all_X[i].size : continue
            forest.fit(all_X[i], sample_weight=all_X_weights[i])
            patch_pred_no_outliers = forest.predict(all_coord[i])
            p, r, _, _ = precision_recall_fscore_support(all_truth[i], patch_pred_no_outliers, average='binary')
            current_precision += p
            current_recall += r
        current_precision /= n_iterations
        current_recall /= n_iterations
        precision[o] += current_precision
        recall[o] += current_recall
    precision[o] /= len(pdb_ids)
    recall[o] /= len(pdb_ids)


def optimal_contamination(path, cat, it, bu, th, zfs):
    files = glob("%s*_%s.pdb" % (path, bu))
    
    manager = Manager()
    
    
    all_pred = []
    all_truth = []
    all_coord = []
    all_X = []
    all_X_weights = []
    pdb_ids = []
               
    for f in sorted(files) :
        pdb_id = basename(f)[:-4]
        print pdb_id
        pdb_ids.append(pdb_id)
        
        f_patch_coord = ("%s%s_patch_coord.txt" % (path, pdb_id))
        f_patch_score = ("%s%s_patch_score.txt" % (path, pdb_id))
        f_patch_truth = ("%s%s_patch_truth.txt" % (path, pdb_id))
    
        with open(f_patch_coord) as coord, open(f_patch_score) as score, open(f_patch_truth) as truth :
            patch_pred   = [(float(x) - th) for x in score.readlines()]
            patch_truth  = [int(x)   for x in truth.readlines()]
            patch_coord  = [[float(x) for x in a.split()] for a in coord.readlines()]
        
        min_v = min(patch_pred)
        max_v = max(patch_pred)
        
        patch_pred_scaled = [(lambda x: -(x / min_v) if x < 0 else (x / max_v))(x) for x in patch_pred]
    
        X = np.array([a[0] for a in zip(patch_coord, patch_pred) if a[1] >= 0])
        X_weights = np.array([x for x in patch_pred_scaled if x >= 0])
        
        all_X.append(X)
        all_X_weights.append(X_weights)
    
        all_pred.append(patch_pred)
        all_truth.append(patch_truth)
        all_coord.append(patch_coord)
        
    outlier_fractions = list(np.arange(0.01, 0.51, 0.01))

    precision = manager.list([0 for _ in outlier_fractions])
    recall    = manager.list([0 for _ in outlier_fractions])
    

    
    Parallel(n_jobs=-1, verbose=10)(delayed(compute_scores)(o, outlier_fractions, n_iterations, pdb_ids, all_truth, all_coord, all_X, all_X_weights, precision, recall) for o in xrange(len(outlier_fractions)))
    
    def fscore(p, r):
        if p + r > 0 : 
            return 2 * p * r /(p+r)
        else :
            return 0
           
    f1_mean = [fscore(precision[o], recall[o])  for o in xrange(len(outlier_fractions))]
    
    outlier_fractions.insert(0, 0)
    
    f1_mean.insert(0, zfs)
    
    print "outlier_fractions = %s" % outlier_fractions
    print "f1_mean = %s" % f1_mean 
    
    best_pair = max(zip(outlier_fractions, f1_mean), key=lambda x:x[1])
    
    plt.figure(figsize=(10, 10), dpi=1200)
    plt.xlim([0.0, 0.5])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Outlier fraction')
    plt.ylabel('Average F1 score') 
    plt.title('The effect of the outlier fraction parameter on the average F1 score \nafter applying the Isolation Forest algorithm')
    plt.plot(outlier_fractions, f1_mean, color='navy' , linestyle='-', linewidth=1)
    
    plt.scatter(best_pair[0], best_pair[1], marker='x', color='red', s=40)
    plt.plot([best_pair[0], best_pair[0]], [0, best_pair[1]], linestyle="dotted", linewidth=1, color='red')
    plt.plot([0, best_pair[0]], [best_pair[1], best_pair[1]], linestyle="dotted", linewidth=1, color='red')
    if best_pair[0] > 0.25 :
        plt.annotate("(%.2f, %.4f)" % best_pair, xy=best_pair, xytext=(-120, 30),
            textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"))#connectionstyle="arc3,rad=-0.3"))
    else :
        plt.annotate("(%.2f, %.4f)" % best_pair, xy=best_pair, xytext=(30, 30),
            textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"))#connectionstyle="arc3,rad=0.3"))    
        
    plt.savefig("%s_%s_%s_best_outlier_f1_score_all_th.pdf" % (cat, it, bu), dpi=1200, bbox_inches='tight')
    plt.clf()
    plt.close()
    
    

threshold = {}
zero_fscore = {}

threshold[("AB", "l", "b")] = 0.3301
zero_fscore[("AB", "l", "b")] = 0.1488
threshold[("AB", "l", "u")] = 0.8920
zero_fscore[("AB", "l", "u")] = 0.1550
threshold[("AB", "r", "b")] = 1.1631
zero_fscore[("AB", "r", "b")] = 0.3351
threshold[("AB", "r", "u")] = 0.5813
zero_fscore[("AB", "r", "u")] = 0.2726


threshold[("A", "l", "b")] = -0.2582
zero_fscore[("A", "l", "b")] = 0.1029
threshold[("A", "l", "u")] = -2.0376
zero_fscore[("A", "l", "u")] = 0.0967
threshold[("A", "r", "b")] = 1.5638
zero_fscore[("A", "r", "b")] = 0.4440
threshold[("A", "r", "u")] = 1.5178
zero_fscore[("A", "r", "u")] = 0.4341


threshold[("EI", "l", "b")] = 1.1852
zero_fscore[("EI", "l", "b")] = 0.4307
threshold[("EI", "l", "u")] = 0.8166
zero_fscore[("EI", "l", "u")] = 0.3584
threshold[("EI", "r", "b")] = 1.0026
zero_fscore[("EI", "r", "b")] = 0.2781
threshold[("EI", "r", "u")] = 0.7007
zero_fscore[("EI", "r", "u")] = 0.2098


threshold[("ER", "l", "b")] = 1.1596
zero_fscore[("ER", "l", "b")] = 0.3128
threshold[("ER", "l", "u")] = 0.9944
zero_fscore[("ER", "l", "u")] = 0.2608
threshold[("ER", "r", "b")] = 0.6674
zero_fscore[("ER", "r", "b")] = 0.1500
threshold[("ER", "r", "u")] = 0.8774
zero_fscore[("ER", "r", "u")] = 0.1432


threshold[("ES", "l", "b")] = 0.8044
zero_fscore[("ES", "l", "b")] = 0.2517
threshold[("ES", "l", "u")] = 0.5547
zero_fscore[("ES", "l", "u")] = 0.1918
threshold[("ES", "r", "b")] = 0.4367
zero_fscore[("ES", "r", "b")] = 0.1927
threshold[("ES", "r", "u")] = 0.2270
zero_fscore[("ES", "r", "u")] = 0.1828


threshold[("OG", "l", "b")] = 1.0008
zero_fscore[("OG", "l", "b")] = 0.2954
threshold[("OG", "l", "u")] = 0.9988
zero_fscore[("OG", "l", "u")] = 0.1855
threshold[("OG", "r", "b")] = 0.1360
zero_fscore[("OG", "r", "b")] = 0.1154
threshold[("OG", "r", "u")] = -0.0920
zero_fscore[("OG", "r", "u")] = 0.1032


threshold[("OR", "l", "b")] = 0.9965
zero_fscore[("OR", "l", "b")] = 0.2269
threshold[("OR", "l", "u")] = 0.6799
zero_fscore[("OR", "l", "u")] = 0.2064
threshold[("OR", "r", "b")] = 0.7203
zero_fscore[("OR", "r", "b")] = 0.1341
threshold[("OR", "r", "u")] = 0.4861
zero_fscore[("OR", "r", "u")] = 0.1473


threshold[("OX", "l", "b")] = 0.9011
zero_fscore[("OX", "l", "b")] = 0.2313
threshold[("OX", "l", "u")] = 0.4963
zero_fscore[("OX", "l", "u")] = 0.1492
threshold[("OX", "r", "b")] = 0.6990
zero_fscore[("OX", "r", "b")] = 0.1986
threshold[("OX", "r", "u")] = 0.7840
zero_fscore[("OX", "r", "u")] = 0.1664


for cat in ["A", "AB", "EI", "ER", "ES", "OG", "OR", "OX"]:
    for it in ["r", "l"] :
        for bu in ["b", "u"] :
            print ("Searching for the optimal threshold for: (%s, %s, %s)\n\n" % (cat, it, bu))
            path = "/media/sebastian/Storage/3DZD_interface_prediction_db5/benchmark5/" + cat + "/train_" + it + "/"
            print path
            optimal_contamination(path, cat, it, bu, threshold[(cat, it, bu)], zero_fscore[(cat, it, bu)])
            print ("*" * 255)

