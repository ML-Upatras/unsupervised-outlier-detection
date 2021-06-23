import os.path

import pandas as pd
from stac.nonparametric_tests import friedman_test

roc_error = pd.read_csv("final_results/roc_comparison.csv")


def friedman_results(df, metric, comparison):
    statistic, p_value, ranking, rank_cmp = friedman_test(*df.to_dict().values())
    friedman = pd.DataFrame(index = df.columns.tolist())
    friedman['ranking'] = ranking
    friedman.sort_values(by='ranking').to_excel(f'friedman_results/friedman_{comparison}_{metric}.xlsx')
    print(f'Friedman Ranking {metric} Exported')


for metric in ["roc", "prn"]:
    comparison_df  = pd.read_csv(os.path.join("final_results", f"{metric}_comparison.csv"))
    fs_comparison_df = pd.read_csv(os.path.join("final_results", f"{metric}_feature_selection_comparison.csv"))
    agg_comparison_df = pd.read_csv(os.path.join("final_results", f"{metric}_aggregation_comparison.csv"))
    friedman_results(comparison_df, metric, "simple")
    friedman_results(fs_comparison_df, metric, "fs")
    friedman_results(agg_comparison_df, metric, "agg")
