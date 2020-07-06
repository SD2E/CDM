import os
import sys
import inspect
import itertools
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
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


class CombinatorialDesignModel(metaclass=ABCMeta):
    # TODO: let user define train_ratio (default to 0.7)
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="BL1-A", **th_kwargs):
        # if type(self) == CombinatorialDesignModel:
        #     raise Exception("CombinatorialDesignModel class may not be instantiated.\n"
        #                     "Please use HostResponseModel or CircuitFluorescenceModel instead.")

        # TODO: build a way to check if user wants to run something that they already ran.
        # TODO: and read that instead of re-running test harnness

        self.output_path = output_path
        self.leaderboard_query = leaderboard_query
        self.target_col = target_col
        self.th = TestHarness(output_location=self.output_path)
        self.th_kwargs = th_kwargs

        if exp_condition_cols is None:
            self.exp_condition_cols = ["strain_name", "inducer_concentration_mM"]
        else:
            self.exp_condition_cols = exp_condition_cols

        if self.leaderboard_query is None:
            # set existing_data and generate future_data
            self.existing_data = self.add_index_per_existing_condition(initial_data)
            self.future_data = self.generate_future_conditions_df()
        else:
            self.run_id = query_leaderboard(query=self.leaderboard_query, th_output_location=output_path)[Names_TH.RUN_ID].values[0]
            assert isinstance(self.run_id, str), "self.run_id should be a string. Got this instead: {}".format(self.run_id)
            # # Put in code to get the path of the model and read it in once Hamed works that in
            # self.model = ""

        self.evaluation_metric = None

    @abstractmethod
    def add_index_per_existing_condition(self, initial_data):
        pass

    def retrieve_future_conditions(self):
        """
        computes full condition space and returns experimental conditions that have no data and need to be predicted
        """
        unique_column_values = []
        for col in self.exp_condition_cols:
            col_unique_vals = self.existing_data[col].unique()
            print('Column {0}: contains {1} unique values'.format(col, len(col_unique_vals)))
            unique_column_values.append(col_unique_vals)
        permutations = set(itertools.product(*unique_column_values))
        temp_df = self.existing_data[self.exp_condition_cols].drop_duplicates()
        existing_conditions = set(zip(*[temp_df[column].values for column in self.exp_condition_cols]))
        future_condtions = permutations - existing_conditions
        existing_conditions = list(existing_conditions)
        future_condtions = list(future_condtions)

        print('Input dataframe contains {0} conditions out of {1} possible conditions'
              '\nThere are {2} conditions to be predicted\n'.format(len(existing_conditions),
                                                                    len(permutations),
                                                                    len(future_condtions)))
        return future_condtions

    @abstractmethod
    def add_index_per_future_condition(self, future_conditions):
        pass

    def generate_future_conditions_df(self):
        """
        generates and returns a dataframe of experimental conditions that lack data
        """
        future_conditions = self.retrieve_future_conditions()
        future_conditions_df = self.add_index_per_future_condition(future_conditions)

        return future_conditions_df

    def invoke_test_harness(self, train_df, test_df, pred_df, percent_train, num_pred_conditions):
        # TODO: figure out how to raise exception or warning for th_kwargs that are passed in but haven't been listed here
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
            index_cols = self.exp_condition_cols
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
            sparse_cols_to_use = None

        self.th.run_custom(function_that_returns_TH_model=function_that_returns_TH_model,
                           dict_of_function_parameters=dict_of_function_parameters,
                           training_data=train_df,
                           testing_data=test_df,
                           description="CDM_run_type: {}, percent_train: {}, num_pred_conditions: {}, "
                                       "more_info: {}".format(inspect.stack()[1][3], percent_train,
                                                              num_pred_conditions, more_info),
                           target_cols=self.target_col,
                           feature_cols_to_use=self.exp_condition_cols + [self.per_condition_index_col],
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
        self.invoke_test_harness(train_df=train_df, test_df=test_df, pred_df=self.future_data,
                                 percent_train=percent_train, num_pred_conditions=len(self.future_data))

    def run_progressive_sampling(self, num_runs=1, start_percent=25, end_percent=100, step_size=5):
        percent_list = list(range(start_percent, end_percent, step_size))
        if end_percent not in percent_list:
            percent_list.append(end_percent)
        percent_list = [p for p in percent_list if 0 < p < 100]  # ensures percentages make sense
        print("Beginning progressive sampling over the following percentages of existing data: {}".format(percent_list))
        for run in range(num_runs):
            for percent_train in percent_list:
                train_df, test_df = self.condition_based_train_test_split(percent_train=percent_train)
                # invoke the Test Harness with the splits we created:
                self.invoke_test_harness(train_df=train_df, test_df=test_df, pred_df=self.future_data,
                                         percent_train=percent_train, num_pred_conditions=len(self.future_data))
            # TODO: characterizaton has yet to include calculation of knee point

    # validation_data goes into here
    def evaluate_predictions(self, new_df, new_df_index):
        '''
        Take in a dataframe that was produced by the experiment and compare it with the predicted dataframe
        :param new_df: a new dataframe generated from data in the lab to compare with prediction dataframe
        :param new_df_index: the index column in the new dataset
        :param pred_df_index: the index column that was used in the predictions dataset
        :param classification: is this a classification or regression problem
        :param custom_metric: define a metric you want to use to compare your predictions. Can be a callable method or str
                            r2 = R^2 metric
                            emd = Earth Mover's distance
        :return: scalar performance of model
        '''
        if self.leaderboard_query is None:
            raise NotImplementedError("evaluate_predictions can only be run if the {} object is instantiated "
                                      "with a leaderboard_query that is not None".format(self.__class__.__name__))

        preds_path = os.path.join(self.output_path, Names_TH.TEST_HARNESS_RESULTS_DIR, Names_TH.RUNS_DIR,
                                  "run_{}".format(self.run_id), "{}.csv".format(Names_TH.PREDICTED_DATA))
        print("Obtaining predictions from this location: {}".format(preds_path))

        df_preds = pd.read_csv(preds_path)
        df_preds = df_preds[self.exp_condition_cols + ["{}_predictions".format(self.target_col)]]
        new_df = new_df[self.exp_condition_cols + [self.target_col]]
        print(df_preds.head())
        print()
        print(new_df.head())
        print()
        df_all = pd.merge(left=df_preds, right=new_df, how="right", left_on=self.exp_condition_cols,
                          right_on=self.exp_condition_cols)

        # df_all = df_preds.merge(new_df)

        # print(df_all)
        # print()

        df_all = join_new_data_with_predictions(new_df, new_df_index, N.index, {Names_TH.RUN_ID: self.run_id},
                                                self.output_path, loo=False, classification=False, file_type=Names_TH.PREDICTED_DATA)

        print(df_all)
        print()

        sys.exit(0)

        for condition in df_all[self.exp_condition_cols].drop_duplicates().values:
            column1_value = condition[0]
            column2_value = condition[1]

            print(condition)

        sys.exit(0)

        if self.evaluation_metric == 'r2':
            return r2_score(df_all[pred_col], df_all[new_df_target])
        elif self.evaluation_metric == 'emd':
            return wasserstein_distance(df_all[pred_col], df_all[new_df_target])
        else:
            raise NotImplementedError()

    def rank_results(self, results_df, control_col, prediction_col, rank_name: str):
        results_df[rank_name] = results_df[prediction_col] / results_df[control_col]
        return results_df


class HostResponseModel(CombinatorialDesignModel):
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="logFC", gene_col="gene", evaluation_metric="r2", **th_kwargs):
        self.per_condition_index_col = gene_col
        super().__init__(initial_data, output_path, leaderboard_query,
                         exp_condition_cols, target_col, **th_kwargs)
        self.evaluation_metric = evaluation_metric

    def add_index_per_existing_condition(self, initial_data):
        """
        This intentionally doesn't do anything because HostResponseModel data already has genes in it.
        """
        return initial_data

    def add_index_per_future_condition(self, future_conditions):
        """
        In this HostResponseModel, this method will add all the genes that exist in the self.existing_data to
        the self.future_data DataFrame. For each condition in future_conditions, all genes are combined with
        that condition, making each row a combination of conditions and gene.
        :param future_conditions: a list of future conditions returned from the self.retrieve_future_conditions method
        :type future_conditions: list
        :return: DataFrame with all the combinations of experimental conditions and genes
        :rtype: Pandas DataFrame
        """
        future_conditions_df = pd.DataFrame(future_conditions, columns=self.exp_condition_cols)
        unique_genes = self.existing_data[self.per_condition_index_col].unique().tolist()
        future_conditions_df[self.per_condition_index_col] = [unique_genes for i in range(len(future_conditions_df))]
        future_conditions_df = future_conditions_df.explode(self.per_condition_index_col).reset_index(drop=True)
        return future_conditions_df


class CircuitFluorescenceModel(CombinatorialDesignModel):
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="BL1-A", num_per_condition_indices=20000, evaluation_metric="emd", **th_kwargs):
        self.per_condition_index_col = "dist_position"
        self.num_per_condition_indices = num_per_condition_indices
        super().__init__(initial_data, output_path, leaderboard_query,
                         exp_condition_cols, target_col, **th_kwargs)
        self.evaluation_metric = evaluation_metric

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
        sampled_df = initial_data.groupby(self.exp_condition_cols).apply(lambda x: x.sample(n=self.num_per_condition_indices, replace=True))
        sampled_df.reset_index(drop=True, inplace=True)

        sampled_df = sampled_df.groupby(self.exp_condition_cols).apply(lambda x: x.sort_values(by=self.target_col, na_position='first'))
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
