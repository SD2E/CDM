import os
import sys
import inspect
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance

from names import Names as N
from CDM_regression import CDM_regression_model
from CDM_classification import CDM_classification_model
from harness.test_harness_class import TestHarness
from harness.utils.parsing_results import *
from harness.utils.names import Names as Names_TH
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression


class CombinatorialDesignModel:
    # TODO: let user define train_ratio (default to 0.7)
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="BL1-A",
                 **th_kwargs):
        # TODO: build a way to check if user wants to run something that they already ran.
        # TODO: and read that instead of re-running test harnness

        self.output_path = output_path
        self.target_col = target_col
        self.th = TestHarness(output_location=self.output_path)
        self.th_kwargs = th_kwargs

        if exp_condition_cols is None:
            self.exp_condition_cols = ["strain_name", "inducer_concentration_mM"]
        else:
            self.exp_condition_cols = exp_condition_cols

        if leaderboard_query is None:
            # set existing_data and generate future_conditions
            self.existing_data = initial_data
            self.future_conditions = self.generate_future_experiments_df()
        else:
            self.model_id = query_leaderboard(query=leaderboard_query, th_output_location=output_path)[Names_TH.RUN_ID]
            # Put in code to get the path of the model and read it in once Hamed works that in
            self.model = ""

    def retrieve_future_experiments(self):
        """
        computes full condition space and returns experiments that have no data and need to be predicted
        """
        unique_column_values = []
        for col in self.exp_condition_cols:
            col_unique_vals = self.existing_data[col].unique()
            print('Column {0}: contains {1} unique values'.format(col, len(col_unique_vals)))
            unique_column_values.append(col_unique_vals)
        permutations = set(itertools.product(*unique_column_values))
        temp_df = self.existing_data[self.exp_condition_cols].drop_duplicates()
        existing_experiments = set(zip(*[temp_df[column].values for column in self.exp_condition_cols]))
        future_experiments = permutations - existing_experiments
        existing_experiments = list(existing_experiments)
        future_experiments = list(future_experiments)

        print('Input dataframe contains {0} conditions out of {1} possible conditions'
              '\nThere are {2} conditions to be predicted\n'.format(len(existing_experiments),
                                                                    len(permutations),
                                                                    len(future_experiments)))
        return future_experiments

    def generate_future_experiments_df(self):
        """
        generates and returns a dataframe of experiments that lack data
        """
        future_experiments = self.retrieve_future_experiments()
        future_experiments_df = pd.DataFrame(future_experiments, columns=self.exp_condition_cols)
        future_experiments_df[N.index] = future_experiments_df.index
        col_order = list(future_experiments_df.columns.values)
        col_order.insert(0, col_order.pop(col_order.index(N.index)))
        future_experiments_df = future_experiments_df[col_order]
        return future_experiments_df

    def invoke_test_harness(self, train_df, test_df, pred_df, percent_train, num_pred_conditions):
        if "function_that_returns_TH_model" in self.th_kwargs:
            function_that_returns_TH_model = self.th_kwargs["function_that_returns_TH_model"]
        else:
            function_that_returns_TH_model = random_forest_regression
        if "dict_of_function_parameters" in self.th_kwargs:
            dict_of_function_parameters = self.th_kwargs["dict_of_function_parameters"]
        else:
            dict_of_function_parameters = {}
        if "description" in self.th_kwargs:
            more_info = self.th_kwargs["description"]
        else:
            more_info = ""
        if "index_cols" in self.th_kwargs:
            index_cols = self.th_kwargs["index_cols"]
        else:
            index_cols = [N.index] + self.exp_condition_cols
        if "normalize" in self.th_kwargs:
            normalize = self.th_kwargs["normalize"]
        else:
            normalize = False
        if "feature_cols_to_normalize" in self.th_kwargs:
            feature_cols_to_normalize = self.th_kwargs["feature_cols_to_normalize"]
        else:
            feature_cols_to_normalize = None
        if "feature_extraction" in self.th_kwargs:
            feature_extraction = self.th_kwargs["feature_extraction"]
        else:
            feature_extraction = False
        if "sparse_cols_to_use" in self.th_kwargs:
            sparse_cols_to_use = self.th_kwargs["sparse_cols_to_use"]
        else:
            sparse_cols_to_use = ["strain_name"]

        self.th.run_custom(function_that_returns_TH_model=function_that_returns_TH_model,
                           dict_of_function_parameters=dict_of_function_parameters,
                           training_data=train_df,
                           testing_data=test_df,
                           description="CDM_run_type: {}, percent_train: {}, num_pred_conditions: {}, "
                                       "more_info: {}".format(inspect.stack()[1][3], percent_train,
                                                              num_pred_conditions, more_info),
                           target_cols=self.target_col,
                           feature_cols_to_use=self.exp_condition_cols,
                           index_cols=index_cols,
                           normalize=normalize,
                           feature_cols_to_normalize=feature_cols_to_normalize,
                           feature_extraction=feature_extraction,
                           predict_untested_data=pred_df,
                           sparse_cols_to_use=sparse_cols_to_use)
        # return self.th.list_of_this_instance_run_ids[-1]

    def condition_based_train_test_split(self, percent_train):
        """
        Creates train and test DataFrames based on unique combinations in self.exp_condition_cols.
        The unique combinations of values in self.exp_condition_cols are calculated,
        and then {percent_train}% of them are used to create train_df, and the rest are used to create test_df.
        :param percent_train: the percentage of the condition_space to use for training data.
        :type percent_train: int
        :return: returns a train DataFrame and a test DataFrame
        :rtype: pandas.DataFrame
        """
        train_ratio = percent_train / 100.0

        exist_data = self.existing_data.copy()
        # create group column based on self.exp_condition_cols:
        group_col_name = "group"
        exist_data[group_col_name] = exist_data.groupby(self.exp_condition_cols, sort=False).ngroup()

        gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=5)
        train_idx, test_idx = next(gss.split(exist_data, groups=exist_data[group_col_name]))
        assert ((len(train_idx) + len(test_idx)) == len(exist_data) == len(self.existing_data)), \
            "the number of train indexes and test indexes must sum up to " \
            "the number of samples in exist_data and self.existing_data"
        train_df = self.existing_data[self.existing_data.index.isin(train_idx)].copy()
        test_df = self.existing_data[self.existing_data.index.isin(test_idx)].copy()
        return train_df, test_df

    def run_single(self, percent_train=70):
        """
        Generates a train/test split of self.existing_data based on the passed-in percent_train amount,
        and runs a single test harness model on that split by calling self.invoke_test_harness.
        The split is stratified on self.exp_condition_cols.
        """
        train_df, test_df = self.condition_based_train_test_split(percent_train=percent_train)
        # train_conditions = train_df.groupby(self.exp_condition_cols).size().reset_index().rename(columns={0: 'count'})
        # test_conditions = test_df.groupby(self.exp_condition_cols).size().reset_index().rename(columns={0: 'count'})
        # print("train_conditions:\n{}\n".format(train_conditions))
        # print("test_conditions:\n{}\n".format(test_conditions))
        self.invoke_test_harness(train_df=train_df, test_df=test_df, pred_df=self.future_conditions,
                                 percent_train=percent_train, num_pred_conditions=len(self.future_conditions))

    def run_progressive_sampling(self, num_runs=1, start_percent=25, end_percent=100, step_size=5):
        percent_list = list(range(start_percent, end_percent, step_size))
        if end_percent not in percent_list:
            percent_list.append(end_percent)
        percent_list = [p for p in percent_list if 0 < p < 100]  # ensures percentages make sense
        print(percent_list)
        print("Beginning progressive sampling over the following percentages of existing data: {}".format(percent_list))
        for run in range(num_runs):
            for percent_train in percent_list:
                train_df, test_df = self.condition_based_train_test_split(percent_train=percent_train)
                # invoke the Test Harness with the splits we created:
                self.invoke_test_harness(train_df=train_df, test_df=test_df, pred_df=self.future_conditions,
                                         percent_train=percent_train, num_pred_conditions=len(self.future_conditions))
            # TODO: characterizaton has yet to include calculation of knee point

    # validation_data goes into here
    def evaluate_model(self, df, index_col_new_data, target_col_new_data, index_col_predictions_data, custom_metric=None):
        '''
        Take in a dataframe that was produced by the experiment and compare it with the predicted dataframe
        :param df: experiment dataframe
        :param df: a new dataframe generated from data in the lab to compare with prediction dataframe
        :param index_col_new_data: the index column in the new dataset
        :param index_col_predictions_data: the index column that was used in the predictions dataset
        :param query: query on leaderboard to see which outputs you want predicted
        :param th_output_location: path to test harness output
        :param loo: True/False -- is this a LOO Run
        :param classification: is this a classification or regression problem
        :param custom_metric: define a metric you want to use to compare your predictions. Can be a callable method or str
                            r2 = R^2 metric
                            emd = Earth Mover's distance
        :return: scalar performance of model
        '''
        df_all = join_new_data_with_predictions(df, index_col_new_data, index_col_predictions_data, {Names_TH.RUN_ID: self.model_id},
                                                self.path, loo=False, classification=False, file_type=Names_TH.PREDICTED_DATA)
        for col in df_all.columns:
            if '_predictions' in col:
                pred_col = col

        if custom_metric == 'r2':
            return r2_score(df_all[pred_col], df_all[target_col_new_data])
        elif custom_metric == 'emd':
            return wasserstein_distance(df_all[pred_col], df_all[target_col_new_data])
        else:
            # TODO: ensure custom metric_can take at least two arguments
            return custom_metric(df_all[pred_col], df_all[target_col_new_data])

    def rank_results(self, results_df, control_col, prediction_col, rank_name: str):
        results_df[rank_name] = results_df[prediction_col] / results_df[control_col]
        return results_df


class Host_Response_Model(CombinatorialDesignModel):

    def __init__(self):
        pass


class Fluorescence_Output_Model(CombinatorialDesignModel):

    def __init__(self):
        pass
