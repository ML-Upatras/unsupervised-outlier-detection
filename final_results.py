import os

import pandas as pd


def calculate_mean(metric='roc'):
    files = [file for file in os.listdir('results') if file.endswith(f"_final_{metric}.csv")]
    print(files)
    df = pd.read_csv(f"results/{files[0]}")
    for file in files[1:]:
        df_new = pd.read_csv(f"results/{file}")
        df = pd.concat([df, df_new])
    print(len(df))
    df_mean = df.groupby(df.index).mean()
    print(len(df_mean))
    df_mean.to_csv(f"final_results/{metric}_mean.csv", index=False)

    comparison_df = df_mean[
        ['Angle-based Outlier Detector (ABOD)', 'Cluster-based Local Outlier Factor', 'Feature Bagging',
         'Histogram-base Outlier Detection (HBOS)', 'Isolation Forest', 'K Nearest Neighbors (KNN)',
         'Local Outlier Factor (LOF)', 'Minimum Covariance Determinant (MCD)', 'One-class SVM (OCSVM)',
         'Principal Component Analysis (PCA)', 'naive_Aom', 'Fixed SPEC Aom_6']]

    comparison_df = comparison_df.copy()
    comparison_df.to_csv(f'final_results/{metric}_comparison.csv', index=False)

    # Export results for comparison between different choices of feature selection
    fs_cols = [col for col in df_mean.columns if 'Aom' in col]
    fs_comparison_df = df_mean[fs_cols]
    fs_comparison_df.to_csv(f'final_results/{metric}_feature_selection_comparison.csv', index=False)

    # Export results for comparison between different choices of aggregation techniques
    agg_cols = [col for col in df_mean.columns if 'Fixed SPEC' in col]
    agg_comparison_df = df_mean[agg_cols]
    agg_comparison_df.to_csv(f'final_results/{metric}_aggregation_comparison.csv', index=False)


calculate_mean('roc')
calculate_mean('prn')
