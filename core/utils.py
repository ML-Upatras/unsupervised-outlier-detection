import os
import numpy as np
import pandas as pd

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

from fsfc.generic import Lasso, LFSBSS
from fsfc.generic import NormalizedCut, GenericSPEC, FixedSPEC, SPEC
from fsfc.generic import MCFS
from fsfc.generic import WKMeans


def define_classifiers(random_state, outliers_fraction):
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
    return classifiers


def define_normalizers(k):

    normalizers = {
        "Fixed SPEC": FixedSPEC(k, 2),
        "Generic SPEC": GenericSPEC(k),
        "NormalizedCut": NormalizedCut(k),
        "Lasso": Lasso(k, 0.01),
        "WKMeans": WKMeans(k, 0.1)
    }

    return normalizers


def define_combination_methods(normalizer_name, k, norm_results):
    combination_methods = {
        normalizer_name+' Average_'+str(k): average(norm_results),
        normalizer_name+' Maximization_'+str(k): maximization(norm_results),
        normalizer_name+' Aom_'+str(k): aom(norm_results, int(k/2)),
        normalizer_name+' Moa_'+str(k): moa(norm_results, int(k/2))
    }

    return combination_methods
