from os import chdir
from glob import glob
from sklearn.model_selection import train_test_split
from numpy import append
from scipy.sparse import vstack

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from scipy.stats import expon, randint, uniform
from sklearn.svm import SVC, LinearSVC

import numpy as np

def import_descriptors(path, file_wildcard, size=None):
    chdir(path)
    X = y = None
    labels = []
    files = glob(file_wildcard)
    for f in sorted(files):
        X_f, y_f = load_svmlight_file(f, zero_based=False)
        if size is not None :
            X_f, _, y_f, _ = train_test_split(X_f, y_f, train_size=size, stratify = y_f)
        if X is None:
            X = X_f
            y = y_f
        else :
            X = vstack([X, X_f], "csr")
            y = append(y, y_f)
        current_label = f[:4]
        labels += [current_label] * y_f.size
    return (X, y, labels)

def hyperparameterSearch(training_set_path, cat, rl, bu):
    print ("Importing descriptors from the training set.")
    X, y, labels = import_descriptors(training_set_path, "*_%s_%s_train_descriptors_N20.txt" % (rl, bu))
    print ("Number of features: %d." % X.shape[-1])

    print ("Scaling data.")
    min_max_scaler = MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X.todense())

    print ("Performing feature selection with randomized logistic regression.")
# set n_jobs=-1 to parallelize the Randomized Logistic Regression
# however, there is a bug in the current version of skitlearn (0.18.1) which results in the following message:
# ValueError: assignment destination is read-only, when parallelizing with n_jobs > 1
    feature_selector = RandomizedLogisticRegression(n_jobs=1) 
    X_scale = feature_selector.fit_transform(X_scale, y)
    print ("Reduced number of features: %d." % X_scale.shape[-1])
    
    print ("Running randomized hyper-parameter search with Leave-One-Out validation for the RBF kernel.")
    param_dist_rbf = {'kernel': ['rbf'], 'C': expon(scale=2000), 'gamma': expon(scale=.01)}
    random_sv_rbf = RandomizedSearchCV(SVC(), param_distributions=param_dist_rbf, n_iter=100, scoring='f1', cv=LeaveOneGroupOut(), n_jobs=-1, error_score=0, iid=False, refit=False)
    random_sv_rbf.fit(X_scale, y, groups=labels)
    
    print ("Running randomized hyper-parameter search with Leave-One-Out validation for the linear kernel.")
    param_dist_linear = {'C': expon(scale=2000)}
    random_sv_linear = RandomizedSearchCV(LinearSVC(), param_distributions=param_dist_linear, n_iter=100, scoring='f1', cv=LeaveOneGroupOut(), n_jobs=-1, error_score=0, iid=False, refit=False)
    random_sv_linear.fit(X_scale, y, groups=labels)
      
    print ("Running randomized hyper-parameter search with Leave-One-Out validation for the polynomial kernel.")
    param_dist_poly = {'kernel': ['poly'], 'C': expon(scale=2000), 'degree': randint(2, 11), 'coef0': uniform(loc=-2, scale=4), 'gamma': expon(scale=.01)}
    random_sv_poly = RandomizedSearchCV(SVC(), param_distributions=param_dist_poly, n_iter=100, scoring='f1', cv=LeaveOneGroupOut(), n_jobs=-1, error_score=0, iid=False, refit=False)
    random_sv_poly.fit(X_scale, y, groups=labels)   
     
    print ("Running randomized hyper-parameter search with Leave-One-Out validation for the sigmoid kernel.")
    param_dist_sigmoid = {'kernel': ['sigmoid'], 'C': expon(scale=2000), 'coef0': uniform(loc=-2, scale=4), 'gamma': expon(scale=.01)}
    random_sv_sigmoid = RandomizedSearchCV(SVC(), param_distributions=param_dist_sigmoid, n_iter=100, scoring='f1', cv=LeaveOneGroupOut(), n_jobs=-1, error_score=0, iid=False, refit=False)
    random_sv_sigmoid.fit(X_scale, y, groups=labels)  

    with open("%sbest_parameters_test_%s_%s_%s.txt" % (training_set_path, cat, rl, bu), "w") as best_params :

        extracted_features = ["%d" % (x + 1) for x in feature_selector.get_support(indices=True)]

        print("Best parameters found on training set with the RBF kernel:\n%s %s" % (random_sv_rbf.best_params_, random_sv_rbf.best_score_))
        best_params.write("Best parameters found on training set with the RBF kernel:\n%s %s\n" % (random_sv_rbf.best_params_, random_sv_rbf.best_score_))
        print("kernel[(\"%s\", \"%s\", \"%s\")] = \"%s\"" % (cat, rl, bu, random_sv_rbf.best_params_["kernel"]))
        best_params.write("\nkernel[(\"%s\", \"%s\", \"%s\")] = \"%s\"\n" % (cat, rl, bu, random_sv_rbf.best_params_["kernel"]))
        print("C[(\"%s\", \"%s\", \"%s\")] = %f" % (cat, rl, bu, random_sv_rbf.best_params_["C"]))
        best_params.write("C[(\"%s\", \"%s\", \"%s\")] = %f\n" % (cat, rl, bu, random_sv_rbf.best_params_["C"]))
        print("gamma[(\"%s\", \"%s\", \"%s\")] = %f" % (cat, rl, bu, random_sv_rbf.best_params_["gamma"]))    
        best_params.write("gamma[(\"%s\", \"%s\", \"%s\")] = %f\n" % (cat, rl, bu, random_sv_rbf.best_params_["gamma"]))
        print("features[(\"%s\", \"%s\", \"%s\")] = [%s]\n" % (cat, rl, bu, ", ".join(extracted_features)))
        best_params.write("features[(\"%s\", \"%s\", \"%s\")] = [%s]\n" % (cat, rl, bu, ", ".join(extracted_features)))
        print("Random LOOCV scores on development set:")
        best_params.write("Random LOOCV scores on development set:\n")
        means = random_sv_rbf.cv_results_['mean_test_score']
        stds = random_sv_rbf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, random_sv_rbf.cv_results_['params']):
            print("%0.5f (stdev %0.5f) for %r" % (mean, std, params))
            best_params.write("%0.5f (stdev %0.5f) for %r\n" % (mean, std, params))
        
        print("Best parameters found on training set with the linear kernel:\n%s %s" % (random_sv_linear.best_params_, random_sv_linear.best_score_))
        best_params.write("Best parameters found on training set with the linear kernel:\n%s %s\n" % (random_sv_linear.best_params_, random_sv_linear.best_score_))
        print("kernel[(\"%s\", \"%s\", \"%s\")] = \"%s\"" % (cat, rl, bu, 'linear'))
        best_params.write("\nkernel[(\"%s\", \"%s\", \"%s\")] = \"%s\"\n" % (cat, rl, bu, 'linear'))
        print("C[(\"%s\", \"%s\", \"%s\")] = %f" % (cat, rl, bu, random_sv_linear.best_params_["C"]))
        best_params.write("C[(\"%s\", \"%s\", \"%s\")] = %f\n" % (cat, rl, bu, random_sv_linear.best_params_["C"]))
        print("features[(\"%s\", \"%s\", \"%s\")] = [%s]\n" % (cat, rl, bu, ", ".join(extracted_features)))
        best_params.write("features[(\"%s\", \"%s\", \"%s\")] = [%s]\n" % (cat, rl, bu, ", ".join(extracted_features)))
        print("Random LOOCV scores on development set:")
        best_params.write("Random LOOCV scores on development set:\n")
        means = random_sv_linear.cv_results_['mean_test_score']
        stds = random_sv_linear.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, random_sv_linear.cv_results_['params']):
            print("%0.5f (stdev %0.5f) for %r" % (mean, std, params))
            best_params.write("%0.5f (stdev %0.5f) for %r\n" % (mean, std, params))

        print("Best parameters found on training set with the polynomial kernel:\n%s %s" % (random_sv_poly.best_params_, random_sv_poly.best_score_))
        best_params.write("Best parameters found on training set with the polynomial kernel:\n%s %s\n" % (random_sv_poly.best_params_, random_sv_poly.best_score_))
        print("kernel[(\"%s\", \"%s\", \"%s\")] = \"%s\"" % (cat, rl, bu, random_sv_poly.best_params_["kernel"]))
        best_params.write("\nkernel[(\"%s\", \"%s\", \"%s\")] = \"%s\"\n" % (cat, rl, bu, random_sv_poly.best_params_["kernel"]))
        print("C[(\"%s\", \"%s\", \"%s\")] = %f" % (cat, rl, bu, random_sv_poly.best_params_["C"]))
        best_params.write("C[(\"%s\", \"%s\", \"%s\")] = %f\n" % (cat, rl, bu, random_sv_poly.best_params_["C"]))
        print("gamma[(\"%s\", \"%s\", \"%s\")] = %f" % (cat, rl, bu, random_sv_poly.best_params_["gamma"]))
        best_params.write("gamma[(\"%s\", \"%s\", \"%s\")] = %f\n" % (cat, rl, bu, random_sv_poly.best_params_["gamma"]))
        print("degree[(\"%s\", \"%s\", \"%s\")] = %d" % (cat, rl, bu, random_sv_poly.best_params_["degree"]))
        best_params.write("degree[(\"%s\", \"%s\", \"%s\")] = %d\n" % (cat, rl, bu, random_sv_poly.best_params_["degree"]))
        print("coef0[(\"%s\", \"%s\", \"%s\")] = %f" % (cat, rl, bu, random_sv_poly.best_params_["coef0"]))
        best_params.write("coef0[(\"%s\", \"%s\", \"%s\")] = %f\n" % (cat, rl, bu, random_sv_poly.best_params_["coef0"]))
        print("features[(\"%s\", \"%s\", \"%s\")] = [%s]\n" % (cat, rl, bu, ", ".join(extracted_features)))
        best_params.write("features[(\"%s\", \"%s\", \"%s\")] = [%s]\n" % (cat, rl, bu, ", ".join(extracted_features)))
        print("Random LOOCV scores on development set:")
        best_params.write("Random LOOCV scores on development set:\n")
        means = random_sv_poly.cv_results_['mean_test_score']
        stds = random_sv_poly.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, random_sv_poly.cv_results_['params']):
            print("%0.5f (stdev %0.5f) for %r" % (mean, std, params))
            best_params.write("%0.5f (stdev %0.5f) for %r\n" % (mean, std, params))

        print("Best parameters found on training set with the sigmoid kernel:\n%s %s" % (random_sv_sigmoid.best_params_, random_sv_sigmoid.best_score_))
        best_params.write("Best parameters found on training set with the sigmoid kernel:\n%s %s\n" % (random_sv_sigmoid.best_params_, random_sv_sigmoid.best_score_))
        print("kernel[(\"%s\", \"%s\", \"%s\")] = \"%s\"" % (cat, rl, bu, random_sv_sigmoid.best_params_["kernel"]))
        best_params.write("\nkernel[(\"%s\", \"%s\", \"%s\")] = \"%s\"\n" % (cat, rl, bu, random_sv_sigmoid.best_params_["kernel"]))
        print("C[(\"%s\", \"%s\", \"%s\")] = %f" % (cat, rl, bu, random_sv_sigmoid.best_params_["C"]))
        best_params.write("C[(\"%s\", \"%s\", \"%s\")] = %f\n" % (cat, rl, bu, random_sv_sigmoid.best_params_["C"]))
        print("gamma[(\"%s\", \"%s\", \"%s\")] = %f" % (cat, rl, bu, random_sv_sigmoid.best_params_["gamma"]))
        best_params.write("gamma[(\"%s\", \"%s\", \"%s\")] = %f\n" % (cat, rl, bu, random_sv_sigmoid.best_params_["gamma"]))
        print("coef0[(\"%s\", \"%s\", \"%s\")] = %f" % (cat, rl, bu, random_sv_sigmoid.best_params_["coef0"]))
        best_params.write("coef0[(\"%s\", \"%s\", \"%s\")] = %f\n" % (cat, rl, bu, random_sv_sigmoid.best_params_["coef0"]))
        print("features[(\"%s\", \"%s\", \"%s\")] = [%s]\n" % (cat, rl, bu, ", ".join(extracted_features)))
        best_params.write("features[(\"%s\", \"%s\", \"%s\")] = [%s]\n" % (cat, rl, bu, ", ".join(extracted_features)))
        print("Random LOOCV scores on development set:")
        best_params.write("Random LOOCV scores on development set:\n")
        means = random_sv_sigmoid.cv_results_['mean_test_score']
        stds = random_sv_sigmoid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, random_sv_sigmoid.cv_results_['params']):
            print("%0.5f (stdev %0.5f) for %r" % (mean, std, params))
            best_params.write("%0.5f (stdev %0.5f) for %r\n" % (mean, std, params))
    
for cat in ["A", "AB", "EI", "ER", "ES", "OG", "OR", "OX"] :
    for rl in ["r", "l"] :
        for bu in ["u", "b"] :
            print ("Running hyper-parameter search for configuration: (%s, %s, %s)\n\n" % (cat, rl, bu))
            training_set_path   = "/media/sebastian/Storage/3DZD_interface_prediction_db5/benchmark5/" + cat + "/train_" + rl + "/descriptors_balanced/"
            hyperparameterSearch(training_set_path, cat, rl, bu)
            print ("*" * 255)
