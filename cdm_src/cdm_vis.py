import sys
import textwrap
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Union, Optional
import matplotlib.pyplot as plt
from harness.utils.parsing_results import *

matplotlib.use('TkAgg')


def plot_distributions(data: pd.DataFrame,
                       columns_to_plot: Union[str, list],
                       replicate_index_col: Optional[str] = None,
                       conditions: Optional[pd.DataFrame] = None,
                       num_bins: int = 50):
    """
    Generates overlaid histograms based on user specifications. Use this function to see/compare distributions
    of predictions, replicates, different columns, etc. See below for details.

    :param data: The DataFrame you want to plot data from.
    :param columns_to_plot: The columns in the DataFrame that should be plotted (overlaid).
    :param replicate_index_col: Defaults to None. If you want to plot data from different replicates
                                separately, you can specify the column in the data that defines replicates.
                                If a replicate_index_col is given, then this function will split each column
                                in columns_to_plot by replicate and then plot them separately (overlaid).
                                If your data doesn't have replicates (e.g. untested predictions), then this
                                should be set to None.
    :param conditions: A Pandas DataFrame specifying the conditions that should be plotted. Each row of the
                       DataFrame should represent a single condition. A separate plot will be created for
                       each condition.
                       This argument Defaults to None, which means all rows in the data DataFrame will be used.
    :param num_bins: Number of bins in each histogram. Defaults to 50 bins.
    :return: Nothing for now...
    """

    # if conditions is DataFrame, do a merge
    list_of_plot_dfs = []
    if conditions is not None:
        cond_cols = list(conditions.columns.values)
        conditions = conditions.drop_duplicates()
        for idx, row in conditions.iterrows():
            cond_name = "\n".join(textwrap.wrap(str(row.to_dict()), 100))
            single_cond_df = row.to_frame().T.reset_index()
            list_of_plot_dfs.append((cond_name, pd.merge(data, single_cond_df, "inner", cond_cols)))
    else:
        list_of_plot_dfs.append((None, data))

    fig, axs = plt.subplots(nrows=len(list_of_plot_dfs), figsize=(10, 10 * len(list_of_plot_dfs)))
    for ax_idx, element in enumerate(list_of_plot_dfs):
        cond_name, cond_df = element
        if isinstance(axs, list) or isinstance(axs, np.ndarray):
            axis = axs[ax_idx]
        else:
            axis = axs
        axis.set_title("Condition = {}".format(cond_name), fontsize=14)
        for col in columns_to_plot:
            if replicate_index_col is not None:
                for r in list(cond_df[replicate_index_col].unique()):
                    repl_df = cond_df.loc[cond_df[replicate_index_col] == r]
                    dp = sns.distplot(repl_df[col], bins=num_bins, label="{}_repl_{}".format(col, r),
                                      norm_hist=False, kde=False, ax=axis)
                    dp.set(xlabel=None)
            else:
                dp = sns.distplot(cond_df[col], bins=num_bins, label=col,
                                  norm_hist=False, kde=False, ax=axis)  # if kde is set to True, it will make norm_hist True too...
                dp.set(xlabel=None)
            axis.legend()
    plt.show()
