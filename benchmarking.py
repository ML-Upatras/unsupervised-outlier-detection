from __future__ import division
from __future__ import print_function

import os
import sys
from time import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.combination import aom, moa, average, maximization
from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores

from fsfc.generic import Lasso, LFSBSS
from fsfc.generic import NormalizedCut, GenericSPEC, FixedSPEC, SPEC
from fsfc.generic import MCFS
from fsfc.generic import WKMeans

from stac.nonparametric_tests import friedman_test 


# Define data file and read X and y
mat_file_list = ['arrhythmia.mat',
                 'cardio.mat',
                 'glass.mat',
                 'ionosphere.mat',
                 'letter.mat',
                 'lympho.mat',
                #  'mnist.mat',
                 'musk.mat',
                 'optdigits.mat',
                 'pendigits.mat',
                 'pima.mat',
                 'satellite.mat',
                 'satimage-2.mat',
                #  'shuttle.mat',
                 'vertebral.mat',
                #  'vowels.mat',
                 'wbc.mat'
                 ]

# Define nine outlier detection tools to be compared
random_state = np.random.RandomState(42)

df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc',
              'ABOD', 'CBLOF', 'FB', 'HBOS', 'IForest', 'KNN', 'LOF', 'MCD',
              'OCSVM', 'PCA']
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

first = True
for mat_file in mat_file_list:
    print("\n... Processing", mat_file, '...')
    mat = loadmat(os.path.join('data', mat_file))

    X = mat['X']
    y = mat['y'].ravel()
    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    roc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]

    # 60% data for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=random_state)
    
    print (X_train.shape)
    # standardizing data for processing
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    classifiers = {'Angle-based Outlier Detector (ABOD)': ABOD(
        contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor': CBLOF(
            contamination=outliers_fraction, check_estimator=False,
            random_state=random_state),
        'Feature Bagging': FeatureBagging(contamination=outliers_fraction,
                                          random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(
            contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Local Outlier Factor (LOF)': LOF(
            contamination=outliers_fraction),
        'Minimum Covariance Determinant (MCD)': MCD(
            contamination=outliers_fraction, random_state=random_state),
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
        'Principal Component Analysis (PCA)': PCA(
            contamination=outliers_fraction, random_state=random_state)
    }
    
    results = pd.DataFrame(columns=classifiers.keys())
    
    for clf_name, clf in classifiers.items():
        t0 = time()
        clf.fit(X_train_norm)
        test_scores = clf.decision_function(X_test_norm)
        standarized_scores = standardizer(test_scores.reshape(-1, 1))
        results[clf_name] = standarized_scores.reshape(1,-1)[0]
        t1 = time()
        duration = round(t1 - t0, ndigits=4)
        time_list.append(duration)
        
        roc = round(roc_auc_score(y_test, test_scores), ndigits=4)
        prn = round(precision_n_scores(y_test, test_scores), ndigits=4)

        print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, '
              'execution time: {duration}s'.format(
            clf_name=clf_name, roc=roc, prn=prn, duration=duration))

        roc_list.append(roc)
        prn_list.append(prn)
    
    k= 6 # From the 10 algorithms, 2 are discarded. 
    normalizers ={
        # "Fixed SPEC" : FixedSPEC(k, 2),
        "Generic SPEC" : GenericSPEC(k),
        "NormalizedCut" : NormalizedCut(k),
        "Lasso" : Lasso(k, 0.01),
        # "LFSBSS" : LFSBSS(k),
        # "MCFS" : MCFS(k, 2),
        "WKMeans" : WKMeans(k, 0.1)

    }

    for normalizer_name, normalizer in normalizers.items():

        norm_results = normalizer.fit_transform(results.values)

        combination_methods = {
            normalizer_name+'_average_'+str(k): average(norm_results),
            normalizer_name+'_maximization_'+str(k) : maximization(norm_results),
            normalizer_name+'_aom_'+str(k) : aom(norm_results, int(k/2)),
            normalizer_name+'_moa_'+str(k) : moa(norm_results, int(k/2))
        }

        for cmb_name, cmb in combination_methods.items():

            roc = round(roc_auc_score(y_test, cmb), ndigits=4)
            prn = round(precision_n_scores(y_test, cmb), ndigits=4)

            roc_list.append(roc)
            prn_list.append(prn)
            time_list.append(np.sum(time_list[4:14]))

            if first:
                df_columns.append(cmb_name)
    
    first = False
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)


roc_df.to_excel('results/roc_performance.xlsx')
print('ROC Performance Exported')

prn_df.to_excel('results/precision_performance.xlsx')
print('Precision Performance Exported')

#Calculate Error ROC
results_roc = roc_df.loc[:, 'ABOD' : ]
roc_error = 1 - results_roc
roc_error.index = range (0, len(mat_file_list)) 

#Friedman Ranking ROC
statistic, p_value, ranking, rank_cmp = friedman_test(*roc_error.to_dict().values())
friedman = pd.DataFrame(index = results_roc.columns.tolist())
friedman['ranking'] = ranking
friedman.sort_values(by='ranking').to_excel('results/friedman_roc.xlsx')
print('Friedman Ranking ROC Exported')

# Top 5 ROC Performing Algorithms
temp = roc_df.loc[:, 'ABOD' : ].astype('float32').apply(lambda s: s.abs().nlargest(5).index.tolist(), axis=1)
roc_top5 = pd.DataFrame(columns=['First', 'Second', 'Third', 'Fourth', 'Fifth'], index=range(len(temp)))
for i in range(len(temp)):
    roc_top5.iloc[i] = temp.iloc[i]
    
roc_top5.index = mat_file_list
roc_top5.to_excel('results/roc_top_5.xlsx')
print('ROC Performance Top 5 Exported')

#Calculate Error Precision
results_prn = prn_df.loc[:, 'ABOD' : ]
prn_error = 1 - results_prn
prn_error.index = range (0, len(mat_file_list)) 

#Friedman Ranking Precision
statistic, p_value, ranking, rank_cmp = friedman_test(*prn_error.to_dict().values())
friedman = pd.DataFrame(index = results_prn.columns.tolist())
friedman['ranking'] = ranking
friedman.sort_values(by='ranking').to_excel('results/friedman_prn.xlsx')
friedman.sort_values(by='ranking')
print('Friedman Ranking Precision Exported')

# Top 5 Precision Performing Algorithms
temp = prn_df.loc[:, 'ABOD' : ].astype('float32').apply(lambda s: s.abs().nlargest(5).index.tolist(), axis=1)
prn_top5 = pd.DataFrame(columns=['First', 'Second', 'Third', 'Fourth', 'Fifth'], index=range(len(temp)))
for i in range(len(temp)):
    prn_top5.iloc[i] = temp.iloc[i]

prn_top5.index = mat_file_list
prn_top5.to_excel('results/prn_top_5.xlsx')
print('Precision Performance Top 5 Exported')