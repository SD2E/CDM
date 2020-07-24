import time
import inspect
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from cdm_src.utils.names import Names as N
from harness.utils.parsing_results import *
from cdm_src.cdm_base_class import CombinatorialDesignModel


class CircuitFluorescenceModel(CombinatorialDesignModel):
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="BL1-A", num_per_condition_indices=20000):
        self.per_condition_index_col = "dist_position"
        self.num_per_condition_indices = num_per_condition_indices
        super().__init__(initial_data, output_path, leaderboard_query,
                         exp_condition_cols, target_col)

    def add_index_per_existing_condition(self, initial_data):
        """
        Samples the data for each condition and assigns a dist_position index to each row based on its value of self.target_col.
        Rows with smaller self.target_col values are assigned smaller dist_position values.
        dist_position stands for position of a row's self.target_col value in the distribution
        of self.target_col values for any combination of experimental conditions.
        :param initial_data: DataFrame of the original data.
        :type initial_data: Pandas DataFrame
        :return: DataFrame with equal amounts of rows per experimental condition, with each row having a dist_position index.
        :rtype: Pandas DataFrame
        """
        # sample 20,000 points with replacement from each group
        sampled_df = initial_data.groupby(self.exp_condition_cols).apply(
            lambda x: x.sample(n=self.num_per_condition_indices, replace=True))
        sampled_df.reset_index(drop=True, inplace=True)

        sampled_df = sampled_df.groupby(self.exp_condition_cols).apply(
            lambda x: x.sort_values(by=self.target_col, na_position='first'))
        sampled_df.reset_index(drop=True, inplace=True)

        sampled_df[self.per_condition_index_col] = sampled_df.groupby(self.exp_condition_cols).cumcount()

        return sampled_df

    def add_index_per_future_condition(self, future_conditions):
        """
        In this CircuitFluorescenceModel, this method will add n indices to each experimental condition,
        where n is determined from self.num_per_condition_indices.
        For each condition in future_conditions, the indices are combined with that condition,
        making each row a combination of experimental conditions and index.
        :param future_conditions: a list of future conditions returned from the self.retrieve_future_conditions method
        :type future_conditions: list
        :return: DataFrame with all the combinations of experimental conditions and accompanying indices
        :rtype: Pandas DataFrame
        """
        future_conditions_df = pd.DataFrame(future_conditions, columns=self.exp_condition_cols)
        future_conditions_df = future_conditions_df.iloc[future_conditions_df.index.repeat(
            self.num_per_condition_indices)].reset_index(drop=True)
        future_conditions_df[N.dist_position] = future_conditions_df.groupby(self.exp_condition_cols).cumcount()
        return future_conditions_df

    def align_predictions_with_new_data(self, predictions_df, new_data_df):
        new_data_df = new_data_df[self.exp_condition_cols + [self.target_col]]
        sampled_new_df_with_dist_position = self.add_index_per_existing_condition(new_data_df)
        col_order = self.feature_and_index_cols + [self.target_col]
        sampled_new_df_with_dist_position = sampled_new_df_with_dist_position[col_order]

        # This code block is for rounding float columns in our two DataFrames.
        # We need to do this because otherwise floating-point errors will affect our merge in an undesirable way.
        # I.e. when merging, Pandas will think 0.00006 in one DataFrame is different form 0.00006 in the other DataFrame.
        float_cols_1 = sampled_new_df_with_dist_position[self.feature_and_index_cols].select_dtypes(include=[float]).columns.values
        float_cols_2 = predictions_df[self.feature_and_index_cols].select_dtypes(include=[float]).columns.values
        if set(float_cols_1) != set(float_cols_2):
            warnings.warn("float_cols_1 is not the same as float_cols_2 !", stacklevel=100000)
        sampled_new_df_with_dist_position[float_cols_1] = sampled_new_df_with_dist_position[float_cols_1].round(10)
        predictions_df[float_cols_2] = predictions_df[float_cols_2].round(10)

        merged_df = pd.merge(sampled_new_df_with_dist_position, predictions_df,
                             how="inner", on=self.feature_and_index_cols)
        len_preds = len(predictions_df)
        len_new_data = len(new_data_df)
        len_sampled_df = len(sampled_new_df_with_dist_position)
        len_merged = len(merged_df)
        print("Use the following print lines to ensure the inner merge is working correctly...")
        print("len(predictions_df): {}\n"
              "len(new_data_df): {}\n"
              "len(sampled_new_df_with_dist_position): {}\n"
              "len(merged_df): {}\n".format(len_preds, len_new_data,
                                            len_sampled_df, len_merged))
        if not (len_merged == len_sampled_df == len_new_data == len_preds):
            warnings.warn("These 4 DataFrames are not equal in length: "
                          "merged_df, sampled_new_df_with_dist_position, new_data_df, predictions_df\n",
                          stacklevel=100000)

        return merged_df

    def score(self, x, y):
        return wasserstein_distance(x, y)

    # Data quality methods
    def replicate_emd_heatmap(self):
        """
        Creates heatmap of EMD values between the target_col of replicates for each condition.
        Saves the generated matrix as a csv, the heatmap as a png, and an index map table as a csv in a replicate_emd_heatmap folder.

        Note that this code is kind of slow, there's probably a faster way to do it using Pandas corr,
        but that requires transforming the current DataFrame into a different shape (like a pivot table).
        However, the issue is that the pivot table ends up with a bunch of NaN values because not each replicate
        has the same amount of samples. This is an issue because pandas corr does pairwise correlation and will
        ignore all the NaN values. Maybe this can be resolved by making the sampling in the
        add_index_per_existing_condition function to sample equally across replicates as well (not just conditions).
        Anyways, here's some code that would be useful if that issue is ever resolved:
        '''
        # add sorting first if you want more ordered heatmap_index values
        df["heatmap_index"] = df.groupby(conds_and_rep_cols).ngroup()
        pivot = df.pivot(index=self.per_condition_index_col, columns="heatmap_index", values=self.target_col)
        pivot.corr(method=wasserstein_distance)  # this will not work right now because we have NaNs in pivot.
        # remember to generate a heatmap_index_map table for output.
        '''
        """
        df = self.existing_data.copy()
        conds_and_rep_cols = self.exp_condition_cols + ["replicate"]
        unique_conds_and_reps = df[conds_and_rep_cols].drop_duplicates().reset_index(drop=True)
        # print(df.groupby(conds_and_rep_cols, as_index=False).size(), "\n")
        num_unique = len(unique_conds_and_reps)

        start_time = time.time()
        matrix = pd.DataFrame(columns=range(num_unique), index=range(num_unique))
        for idx1, conds_and_rep_1 in unique_conds_and_reps.iterrows():
            mask_1 = (df[conds_and_rep_cols] == conds_and_rep_1).all(axis=1)
            target_1 = df.loc[mask_1, self.target_col]

            for idx2, conds_and_rep_2 in unique_conds_and_reps.iterrows():
                mask_2 = (df[conds_and_rep_cols] == conds_and_rep_2).all(axis=1)
                target_2 = df.loc[mask_2, self.target_col]

                emd = wasserstein_distance(target_1, target_2)
                matrix[idx1][idx2] = float(emd)
                print("\rCurrent index (max is {}_{}): ".format(num_unique - 1, num_unique - 1),
                      "{}_{}".format(idx1, idx2), end="", flush=True)
        print("\rEMD matrix creation took {} seconds".format(round(time.time() - start_time, 2)))
        matrix = matrix.astype(float)

        method_name = str(inspect.stack()[0][3])
        heatmap_output_dir = os.path.join(self.output_path, method_name)
        os.makedirs(heatmap_output_dir, exist_ok=True)

        # write out matrix
        matrix.to_csv(os.path.join(heatmap_output_dir, "replicate_emd_matrix.csv"),
                      index=True)

        # write out key table that maps heatmap indices to conditions and replicates
        unique_conds_and_reps.to_csv(os.path.join(heatmap_output_dir, "heatmap_index_map.csv"),
                                     index=True, index_label="heatmap_index")

        # write out heatmap
        heatmap = sns.heatmap(matrix, xticklabels=True, yticklabels=True, cmap="Greens_r")
        heatmap.set_title('EMD Heatmap Between Conditions and Replicates')
        heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), rotation=90, fontsize=4)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), rotation=0, fontsize=4)
        plt.savefig(os.path.join(heatmap_output_dir, "{}.png".format(method_name)), dpi=200)
