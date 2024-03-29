import os
import numpy as np
import pandas as pd
from time import time
from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores

from core.utils import define_classifiers, define_normalizers, define_combination_methods
from pyod.models.combination import aom, moa, average, maximization


# define data file and read X and y
mat_file_list = ['arrhythmia.mat',
                 'cardio.mat',
                 'glass.mat',
                 'ionosphere.mat',
                 'letter.mat',
                 'lympho.mat',
                 'mnist.mat',
                 'musk.mat',
                 'optdigits.mat',
                 'pendigits.mat',
                 'pima.mat',
                 'satellite.mat',
                 'satimage-2.mat',
                 'shuttle.mat',
                 'vertebral.mat',
                 'vowels.mat',
                 'wbc.mat'
                 ]

# define nine outlier detection tools to be compared
random_state = np.random.RandomState(42)

# define containers for results of all datasets
method_names = []
roc_results = {}
prn_results = {}

for num_mat, mat_file in enumerate(mat_file_list):
    print("\n... Processing", mat_file, '...', '\n')
    mat = loadmat(os.path.join('data', mat_file))

    X = mat['X']
    y = mat['y'].ravel()
    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    print ('Dataset Shape:', X.shape)
    print ('Outliers Percentage', outliers_percentage)

    # construct containers for saving results of each dataset
    roc_list = []
    prn_list = []
    time_list = []

    # 60% data for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=random_state)
    
    # standardizing data for processing
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    # define classifiers
    classifiers = define_classifiers(random_state, outliers_fraction)

    # create df for results
    train_results = pd.DataFrame(columns=classifiers.keys())
    test_results = pd.DataFrame(columns=classifiers.keys())

    print ('\n', 'Outliers Detection', '\n')
    for clf_name, clf in classifiers.items():

        # keep name of convetional models (once on the first itteration)
        if num_mat == 0:
            method_names.append(clf_name)

        # train
        t0 = time()
        clf.fit(X_train_norm)
        train_scores = clf.decision_function(X_train_norm)
        test_scores = clf.decision_function(X_test_norm)
        # Then the output scores are standardized into zero average and unit std before combination. 
        # This step is crucial to adjust the detector outputs to the same scale.
        standarized_train_scores = standardizer(train_scores.reshape(-1, 1))
        standarized_test_scores = standardizer(test_scores.reshape(-1, 1))
        train_results[clf_name] = standarized_train_scores.reshape(1,-1)[0]
        test_results[clf_name] = standarized_test_scores.reshape(1,-1)[0]
        t1 = time()

        #calculate metrics of interest
        duration = round(t1 - t0, ndigits=4)
        roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
        prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

        print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
              'execution time: {duration}s'.format(
            clf_name=clf_name, roc=roc, prn=prn, duration=duration))
            
        time_list.append(duration)
        roc_list.append(roc)
        prn_list.append(prn)

    # total duration of outlier detection methods to run
    duration_until_now = round(np.sum(time_list[4:]), ndigits=4)
    print ('\n', 'Duration of outliers detection', duration_until_now, '\n')

    k= 6 # From the 10 algorithms, 4 are discarded (the best are keeped for stacking). 
    normalizers = define_normalizers(k)
    
    print ('\n', 'Unsupervised Feature Selection & Stacking', '\n')
    for normalizer_name, normalizer in normalizers.items():
        t0 = time()
        # feature selection on training data
        norm_train_results = normalizer.fit_transform(train_results.values)
        # keep scores and isolate the bets estimators
        norm_train_scores = normalizer._get_scores()
        top_k = sorted(np.argsort(norm_train_scores)[-k:])
        # top_estimator_names = train_results.iloc[:, top_k].columns
        t1 = time()
        duration = duration_until_now + round(t1 - t0, ndigits=4)

        # from the testing results keep the selected estimator results
        norm_results = test_results.iloc[:, top_k]

        # define combination methods
        combination_methods = define_combination_methods(normalizer_name, k, norm_results)
        for cmb_name, cmb in combination_methods.items():

            #keep name of combination (once on the first itteration)
            if num_mat == 0:
                method_names.append(cmb_name)

            roc = round(roc_auc_score(y_test, cmb), ndigits=4)
            prn = round(precision_n_scores(y_test, cmb), ndigits=4)

            roc_list.append(roc)
            prn_list.append(prn)
            time_list.append(duration)

            print('{cmb_name} ROC:{roc}, precision @ rank n:{prn}, '
              'execution time: {duration}s'.format(
            cmb_name=cmb_name, roc=roc, prn=prn, duration=duration))

    roc_results[num_mat] = [-1 * x for x in roc_list]
    prn_results[num_mat] = [-1 * x for x in prn_list]

print ('\n Number of method were used: {}'.format(len(method_names), ))

# Export all results
roc_df = pd.DataFrame(roc_results).transpose()
roc_df.columns=method_names
roc_df.to_csv('results/final_roc.csv', index=False)

prn_df = pd.DataFrame(prn_results).transpose()
prn_df.columns=method_names
prn_df.to_csv('results/final_prn.csv', index=False)

# Export results for comparison between the algorithms and the proposed technique
roc_comparison_df = roc_df[['Angle-based Outlier Detector (ABOD)', 'Cluster-based Local Outlier Factor', 'Feature Bagging',
                            'Histogram-base Outlier Detection (HBOS)', 'Isolation Forest', 'K Nearest Neighbors (KNN)',
                            'Local Outlier Factor (LOF)', 'Minimum Covariance Determinant (MCD)','One-class SVM (OCSVM)',
                            'Principal Component Analysis (PCA)', 'Lasso Moa_6']]
prn_comparison_df = prn_df[['Angle-based Outlier Detector (ABOD)', 'Cluster-based Local Outlier Factor', 'Feature Bagging',
                            'Histogram-base Outlier Detection (HBOS)', 'Isolation Forest', 'K Nearest Neighbors (KNN)',
                            'Local Outlier Factor (LOF)', 'Minimum Covariance Determinant (MCD)','One-class SVM (OCSVM)',
                            'Principal Component Analysis (PCA)', 'Lasso Moa_6']]

roc_comparison_df = roc_comparison_df.copy()
prn_comparison_df = prn_comparison_df.copy()

roc_comparison_df['Moa'] = moa(roc_comparison_df.iloc[:, :10], 5)
prn_comparison_df['Moa'] = moa(prn_comparison_df.iloc[:, :10], 5)

roc_comparison_df.to_csv('results/roc_comparison.csv', index=False)
prn_comparison_df.to_csv('results/prn_comparison.csv', index=False)

# Export results for comparison between different choices of feature selection
roc_fs_cols = [col for col in roc_df.columns if 'Moa' in col]
prn_fs_cols = [col for col in prn_df.columns if 'Moa' in col]

roc_comparison_2_df = roc_df[roc_fs_cols]
prn_comparison_2_df = roc_df[prn_fs_cols]

roc_comparison_2_df.to_csv('results/roc_comparison_2.csv', index=False)
prn_comparison_2_df.to_csv('results/prn_comparison_2.csv', index=False)

# Export results for comparison between different choices of aggregation techniques 
roc_agg_cols = [col for col in roc_df.columns if 'Lasso' in col]
prn_agg_cols = [col for col in prn_df.columns if 'Lasso' in col]

roc_comparison_3_df = roc_df[roc_agg_cols]
prn_comparison_3_df = roc_df[prn_agg_cols]

roc_comparison_3_df.to_csv('results/roc_comparison_3.csv', index=False)
prn_comparison_3_df.to_csv('results/prn_comparison_3.csv', index=False)