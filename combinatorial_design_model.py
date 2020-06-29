import os
import itertools
import numpy as np
import pandas as pd
from harness.test_harness_class import TestHarness
from CDM_regression import CDM_regression_model
from CDM_classification import CDM_classification_model
from harness.utils.parsing_results import *
from harness.utils.names import Names as Names_TH
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance


class CombinatorialDesignModel:
    # TODO: let user define train_ratio (default to 0.7)
    def __init__(self, initial_data=None, path=".", leaderboard_query=None, exp_condition_cols=None, **th_properties):
        # TODO: build a way to check if user wants to run something that they already ran.
        # TODO: and read that instead of re-running test harnness

        self.path = path
        self.th = TestHarness(output_location=path)

        if exp_condition_cols is None:
            self.exp_condition_cols = ["strain_name", "inducer_concentration_mM"]
        else:
            self.exp_condition_cols = exp_condition_cols

        if leaderboard_query is None:
            # the data is split into existing_data and future_data
            self.existing_data = initial_data
            self.future_data = self.generate_future_experiments_df()
        else:
            self.model_id = query_leaderboard(query=leaderboard_query, th_output_location=path)[Names_TH.RUN_ID]
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
        return future_experiments_df

    def run_single(self, percent_train=70):
        """
        Generates a train/test split of self.existing_data based on the passed-in percent_train amount,
        and runs a single test harness model on that split by calling self.invoke_test_harness.
        The split is stratified on self.exp_condition_cols.
        """
        train_ratio = percent_train / 100.0
        train_df, test_df = train_test_split(self.existing_data, train_size=train_ratio, random_state=5,
                                             stratify=self.existing_data[self.exp_condition_cols])
        # note this is not finished

    def run_progressive_sampling(self, num_runs=3, start_percent=25, end_percent=100, step_size=5,
                                 percent_train=70):
        train_ratio = percent_train / 100.0
        percent_list = list(range(start_percent, end_percent, step_size))
        if end_percent not in percent_list:
            percent_list.append(end_percent)
        for run in range(1, num_runs):
            for percent in percent_list:
                print(percent)
                if percent == 100:
                    existing_data_sample = self.existing_data.copy()
                else:
                    ratio = percent / 100.0
                    print(ratio)
                    # the following line samples data from self.existing_data with stratification and a set random_state:
                    existing_data_sample, _ = train_test_split(self.existing_data, train_size=percent, random_state=5,
                                                               stratify=self.existing_data[self.exp_condition_cols])
                # now we split the existing_data_sample into train and test, with stratification and a set random_state:
                train_df, test_df = train_test_split(existing_data_sample, train_size=train_ratio, random_state=5,
                                                     stratify=existing_data_sample[self.exp_condition_cols])
                self.build_model(train_df=train_df, test_df=test_df, th_properties={})
            # TODO: characterizaton has yet to include calculation of knee point

    def build_model(self, train_df=None, test_df=None, th_properties={}):
        if train_df is None or test_df is None:
            train_df, test_df = train_test_split(self.existing_data, test_size=0.2)

        # TODO: Fix up test/harness run
        '''
        PUT IN PREDICTED UNTESTED DATAFRAME FLAG!
        '''
        self.th.run_custom(function_that_returns_TH_model=CDM_regression_model,
                           dict_of_function_parameters={'input_size': len(th_properties['feature_cols_to_use']),
                                                        'output_size': len(th_properties['cols_to_predict'])},
                           training_data=train_df, testing_data=test_df,
                           description=th_properties[Names_TH.DESCRIPTION],
                           target_cols=th_properties['cols_to_predict'],
                           feature_cols_to_use=th_properties['feature_cols_to_use'],
                           index_cols=th_properties['index_cols'],
                           normalize=False, feature_cols_to_normalize=None)

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
