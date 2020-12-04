import inspect
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Tuple, Union, Optional
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance
from cdm_src.utils.names import Names as N
from harness.test_harness_class import TestHarness
from harness.utils.parsing_results import *
from harness.utils.names import Names as Names_TH
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression

# gets rid of the annoying SettingWithCopyWarnings
# set this equal to "raise" if you feel like debugging SettingWithCopyWarnings.
pd.options.mode.chained_assignment = None  # default="warn"


def custom_formatwarning(message, category, filename, lineno, line=''):
    return "\n" + str(filename) + ":" + str(lineno) + ":\n" + str(message) + '\n'


warnings.formatwarning = custom_formatwarning


class CombinatorialDesignModel(metaclass=ABCMeta):
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="BL1-A",
                 custom_future_conditions: Optional[pd.DataFrame] = None):
        """

        :param initial_data:
        :param output_path:
        :param leaderboard_query:
        :param exp_condition_cols:
        :param target_col:
        :param custom_future_conditions: None or DataFrame with exp_condition_cols as its columns.
                                         Each row represents a condition. This variable is used by the user
                                         to give a custom set of conditions to predict on when they don't want
                                         the default of all possible conditions to be predicted.
        """
        # if type(self) == CombinatorialDesignModel:
        #     raise Exception("CombinatorialDesignModel class may not be instantiated.\n"
        #                     "Please use HostResponseModel or CircuitFluorescenceModel instead.")

        # TODO: build a way to check if user wants to run something that they already ran.
        # TODO: and read that instead of re-running test harness

        self.output_path = os.path.join(output_path, "cdm_outputs")
        os.makedirs(self.output_path, exist_ok=True)
        self.leaderboard_query = leaderboard_query
        self.target_col = target_col
        self.th = TestHarness(output_location=self.output_path)

        if exp_condition_cols is None:
            self.exp_condition_cols = ["strain_name", "inducer_concentration_mM"]
        else:
            self.exp_condition_cols = exp_condition_cols
        self.feature_and_index_cols = self.exp_condition_cols + [self.per_condition_index_col]
        self.feature_and_index_cols_copy = self.feature_and_index_cols.copy()  # Make a copy because some methods may change it.
        self.initial_data = initial_data  # allows users to see what was passed in before any changes were made
        if self.leaderboard_query is None:
            # set existing_data and generate future_data
            self.existing_data = self.add_index_per_existing_condition(initial_data)
            self.future_data = self.generate_future_conditions_df(custom_future_conditions=custom_future_conditions)
        else:
            self.existing_data = None
            query_matches = query_leaderboard(query=self.leaderboard_query, th_output_location=self.output_path)
            num_matches = len(query_matches)
            if num_matches < 1:
                raise Exception("No leaderboard rows match the query you provided. Here's what the leaderboard looks like:\n"
                                "{}".format(query_leaderboard(query={}, th_output_location=self.output_path)))
            elif num_matches > 1:
                warnings.warn("Your leaderboard query returned {} row matches. "
                              "Only the first match will be used... Here are the matching rows:".format(num_matches))
            else:
                print("Your leaderboard query matched the following row:")
            print(query_matches, "\n")
            run_ids = query_matches[Names_TH.RUN_ID].values
            self.run_id = run_ids[0]
            print("The run_id for the Test Harness run being read-in is: {}".format(self.run_id))
            assert isinstance(self.run_id, str), "self.run_id should be a string. Got this instead: {}".format(self.run_id)

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
        future_conditions = permutations - existing_conditions
        existing_conditions = list(existing_conditions)
        future_conditions = list(future_conditions)

        print('Input dataframe contains {0} conditions out of {1} possible conditions'
              '\nThere are {2} conditions to be predicted\n'.format(len(existing_conditions),
                                                                    len(permutations),
                                                                    len(future_conditions)))
        return future_conditions

    @abstractmethod
    def add_index_per_future_condition(self, future_conditions):
        pass

    def generate_future_conditions_df(self, custom_future_conditions):
        """
        generates and returns a dataframe of experimental conditions that lack data
        """
        if custom_future_conditions is not None:
            future_conditions = list(custom_future_conditions.itertuples(index=False, name=None))
            print('{} conditions will be predicted (derived from the passed-in '
                  'custom_future_conditions DataFrame)\n'.format(len(future_conditions)))
        else:
            future_conditions = self.retrieve_future_conditions()
        future_conditions_df = self.add_index_per_future_condition(future_conditions)

        return future_conditions_df

    def _invoke_test_harness(self, train_df, test_df, pred_df, percent_train, num_pred_conditions, **th_kwargs):
        # TODO: figure out how to raise exception or warning for th_kwargs that are passed in but haven't been listed here
        if "function_that_returns_TH_model" in th_kwargs:
            function_that_returns_TH_model = th_kwargs["function_that_returns_TH_model"]
        else:
            function_that_returns_TH_model = random_forest_regression
        if "dict_of_function_parameters" in th_kwargs:
            dict_of_function_parameters = th_kwargs["dict_of_function_parameters"]
        else:
            dict_of_function_parameters = {}
        if "description" in th_kwargs:
            more_info = th_kwargs["description"]
        else:
            more_info = ""
        if "index_cols" in th_kwargs:
            index_cols = th_kwargs["index_cols"]
        else:
            index_cols = self.feature_and_index_cols
        if "normalize" in th_kwargs:
            normalize = th_kwargs["normalize"]
        else:
            normalize = False
        if "feature_cols_to_use" in th_kwargs:
            warnings.warn("You are overwriting the features to use, this may impact downstream integration with predictions....")
            feature_cols_to_use = th_kwargs['feature_cols_to_use']
        else:
            feature_cols_to_use = self.feature_and_index_cols
        if "feature_cols_to_normalize" in th_kwargs:
            feature_cols_to_normalize = th_kwargs["feature_cols_to_normalize"]
        else:
            feature_cols_to_normalize = None
        if "feature_extraction" in th_kwargs:
            feature_extraction = th_kwargs["feature_extraction"]
        else:
            feature_extraction = False
        if "sparse_cols_to_use" in th_kwargs:
            sparse_cols_to_use = th_kwargs["sparse_cols_to_use"]
        else:
            sparse_cols_to_use = None
        if len(pred_df) == 0:
            pred_df = False

        self.th.run_custom(function_that_returns_TH_model=function_that_returns_TH_model,
                           dict_of_function_parameters=dict_of_function_parameters,
                           training_data=train_df,
                           testing_data=test_df,
                           description="CDM_run_type: {}, percent_train: {}, num_pred_conditions: {}, "
                                       "more_info: {}".format(inspect.stack()[1][3], percent_train,
                                                              num_pred_conditions, more_info),
                           target_cols=self.target_col,
                           feature_cols_to_use=feature_cols_to_use,
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

    def run_single(self, percent_train=70, **th_kwargs):
        """
        Generates a train/test split of self.existing_data based on the passed-in percent_train amount,
        and runs a single test harness model on that split by calling self._invoke_test_harness.
        The split is stratified on self.exp_condition_cols.
        """
        train_df, test_df = self.condition_based_train_test_split(percent_train=percent_train)
        # train_conditions = train_df.groupby(self.exp_condition_cols).size().reset_index().rename(columns={0: 'count'})
        # test_conditions = test_df.groupby(self.exp_condition_cols).size().reset_index().rename(columns={0: 'count'})
        # print("train_conditions:\n{}\n".format(train_conditions))
        # print("test_conditions:\n{}\n".format(test_conditions))
        self._invoke_test_harness(train_df=train_df, test_df=test_df, pred_df=self.future_data,
                                  percent_train=percent_train, num_pred_conditions=len(self.future_data),
                                  **th_kwargs)

    def run_progressive_sampling(self, num_runs=1, start_percent=25, end_percent=100, step_size=5, **th_kwargs):
        percent_list = list(range(start_percent, end_percent, step_size))
        if end_percent not in percent_list:
            percent_list.append(end_percent)
        percent_list = [p for p in percent_list if 0 < p < 100]  # ensures percentages make sense
        print("Beginning progressive sampling over the following percentages of existing data: {}".format(percent_list))
        for run in range(num_runs):
            for percent_train in percent_list:
                train_df, test_df = self.condition_based_train_test_split(percent_train=percent_train)
                # invoke the Test Harness with the splits we created:
                self._invoke_test_harness(train_df=train_df, test_df=test_df, pred_df=self.future_data,
                                          percent_train=percent_train, num_pred_conditions=len(self.future_data),
                                          **th_kwargs)
            # TODO: characterizaton has yet to include calculation of knee point

    @abstractmethod
    def _align_predictions_with_new_data(self, predictions_df, new_data_df):
        """
        This method should align and merge the DataFrame of previous predictions with the DataFrame of new experimental data.
        This method should be implemented in the HRM and CFM subclasses.
        """

    @abstractmethod
    def score(self, x, y):
        """This method will score the agreement between two lists of numbers, using metrics such as R^2 or EMD"""

    def rank_results(self, results_df, control_col, prediction_col, rank_name: str):
        """
        Currently sorts the rows based on how close the predictions were to the "controls"
        """
        results_df[rank_name] = abs((results_df[prediction_col] / results_df[control_col]) - 1)
        results_df.sort_values(by=rank_name, inplace=True)
        return results_df

    def _inspect_condition_overlap(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df1_conds = df1[self.exp_condition_cols].drop_duplicates()
        df2_conds = df2[self.exp_condition_cols].drop_duplicates()
        indicator = "which"
        outer_merge = merge_dfs_with_float_columns(df1=df1_conds, df2=df2_conds,
                                                   on=None, how="outer", indicator=indicator)

        df1_only_conds = outer_merge.loc[outer_merge[indicator] == "left_only"]
        df2_only_conds = outer_merge.loc[outer_merge[indicator] == "right_only"]
        return df1_only_conds, df2_only_conds

    def _inspect_col_overlap_per_condition(self, df1: pd.DataFrame, df2: pd.DataFrame,
                                           column: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # these two lines do two important things by calling round_float_cols_in_df:
        #    1. make copies of df1 and df2, so that the passed-in DataFrames are not altered by this method
        #    2. undertake float columns through rounding to avoid mismatches during the multi-index + subsetting lines
        df1, _ = round_float_cols_in_df(df=df1, cols_to_check=self.exp_condition_cols)
        df2, _ = round_float_cols_in_df(df=df2, cols_to_check=self.exp_condition_cols)

        if column is None:
            column = self.per_condition_index_col
        if (column == self.per_condition_index_col) and (self.__class__.__name__ == "CircuitFluorescenceModel"):
            # this next line can give a different df from sampled_new_df_with_dist_position in the _align_predictions_with_new_data
            # method of the CFM class. But it shouldn't matter because this method is just checking to see if the values
            # in the column of interest overlap, not what they map to in other columns.
            df2 = self.add_index_per_existing_condition(df2)
        df1 = df1[self.exp_condition_cols + [column]]
        df2 = df2[self.exp_condition_cols + [column]]
        df1_conds = df1[self.exp_condition_cols].drop_duplicates()
        df2_conds = df2[self.exp_condition_cols].drop_duplicates()
        joint_conds_only = merge_dfs_with_float_columns(df1=df1_conds, df2=df2_conds,
                                                        on=self.exp_condition_cols, how="inner")

        # creates multi-index from the condition column values in each DataFrame
        keys = list(joint_conds_only.columns.values)
        idx1 = df1.set_index(keys).index
        idx2 = df2.set_index(keys).index
        idx_joint = joint_conds_only.set_index(keys).index
        # subsets the DataFrames to only keep rows that belong to conditions that exist in joint_conds_only
        df1_col_vals_per_mutual_cond = df1[idx1.isin(idx_joint)]
        df2_col_vals_per_mutual_cond = df2[idx2.isin(idx_joint)]

        indicator = "which"
        outer_merge = merge_dfs_with_float_columns(df1=df1_col_vals_per_mutual_cond, df2=df2_col_vals_per_mutual_cond,
                                                   on=None, how="outer", indicator=indicator)
        only_df1_col_vals_per_mutual_cond = outer_merge.loc[outer_merge[indicator] == "left_only"]
        only_df2_col_vals_per_mutual_cond = outer_merge.loc[outer_merge[indicator] == "right_only"]
        only_df1_col_vals_per_mutual_cond.drop(columns=[indicator], inplace=True)
        only_df2_col_vals_per_mutual_cond.drop(columns=[indicator], inplace=True)

        df1_col_summary_per_mutual_cond = only_df1_col_vals_per_mutual_cond.groupby(
            self.exp_condition_cols, as_index=False).agg(
            {
                column: [("{}_count".format(column), "count"),
                         ("{}_mode".format(column), lambda x: x.value_counts().index[0])]
            })

        df2_col_summary_per_mutual_cond = only_df2_col_vals_per_mutual_cond.groupby(
            self.exp_condition_cols, as_index=False).agg(
            {
                column: [("{}_count".format(column), "count"),
                         ("{}_mode".format(column), lambda x: x.value_counts().index[0])]
            })

        return df1_col_summary_per_mutual_cond, df2_col_summary_per_mutual_cond

    def evaluate_predictions(self, new_data_df):
        '''
        Take in a dataframe that was produced by the experiment and compare it with the predicted dataframe
        :param new_data_df: a new dataframe generated from data in the lab to compare with prediction dataframe
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

        print("Obtaining predictions from this location: {}\n".format(preds_path))

        df_preds = pd.read_csv(preds_path)
        target_pred_col = "{}_predictions".format(self.target_col)
        self.df_preds = df_preds[self.feature_and_index_cols + [target_pred_col]]
        self.combined_df = self._align_predictions_with_new_data(predictions_df=self.df_preds, new_data_df=new_data_df)

        pred_only_conds, new_data_only_conds = self._inspect_condition_overlap(df1=self.df_preds, df2=new_data_df)
        if len(pred_only_conds) != 0:
            warnings.warn("The following conditions exist only in the predicted data "
                          "and do not exist in the new experimental data:\n\n{}\n\n".format(pred_only_conds))
        if len(new_data_only_conds) != 0:
            warnings.warn("The following conditions exist only in the new experimental data "
                          "and do not exist in the predicted data:\n\n{}\n\n".format(new_data_only_conds))

        column_to_inspect = self.per_condition_index_col
        pred_only_col_vals, new_data_only_col_vals = self._inspect_col_overlap_per_condition(df1=self.df_preds, df2=new_data_df,
                                                                                             column=column_to_inspect)
        if len(pred_only_col_vals) != 0:
            warnings.warn("Within mutual conditions, the following {}s exist only in the predicted data "
                          "and do not exist in the new experimental data:\n\n{}\n\n".format(column_to_inspect,
                                                                                            pred_only_col_vals))
        if len(new_data_only_col_vals) != 0:
            warnings.warn("Within mutual conditions, the following {}s exist only in the new experimental data "
                          "and do not exist in the predicted data:\n\n{}\n\n".format(column_to_inspect,
                                                                                     new_data_only_col_vals))

        # ranked_results = self.rank_results(results_df=self.combined_df, control_col=self.target_col,
        #                                    prediction_col=target_pred_col, rank_name="ratio_based_ranking")

        score = self.score(x=self.combined_df[self.target_col], y=self.combined_df[target_pred_col])
        return score

    def reset_feature_and_index_cols(self):
        '''
        Reset columns back to original set if it is ever changed
        :return:
        '''
        self.feature_and_index_cols = self.feature_and_index_cols_copy.copy()


def round_float_cols_in_df(df: pd.DataFrame, cols_to_check: Optional[list] = None,
                           decimal_places: int = 10) -> Tuple[pd.DataFrame, list]:
    """
    Finds the float columns in the given DataFrame (only from among cols_to_check) and
    rounds them to the desired number of decimal places.
    :param df: the DataFrame we want to operate on.
    :param cols_to_check: columns within df to check for float columns. If set to None, all columns will be checked.
    :param decimal_places: number of decimal places to round the floats to. I've found that 10 is good for our data sets so far.
    :return: Returns Tuple of (updated DataFrame with rounded float columns, list of float columns whose values were rounded).
    """
    df = df.copy()  # don't want to change the passed-in DataFrames, so make a copy
    float_cols = list(df[cols_to_check].select_dtypes(include=[float]).columns.values)
    df[float_cols] = df[float_cols].round(decimal_places)
    return df, float_cols


def merge_dfs_with_float_columns(df1: pd.DataFrame, df2: pd.DataFrame, on: Optional[list] = None, how: str = "inner",
                                 indicator: Union[str, bool] = False, decimal_places: int = 10) -> pd.DataFrame:
    """
    This function is for rounding float columns in two DataFrames before merging them.
    We need to do this because otherwise floating-point errors will affect our merge in an undesirable way.
    I.e. when merging, Pandas will think 0.00006 in one DataFrame is different from 0.00006 in the other DataFrame.
    :param df1: first DataFrame
    :param df2: second DataFrame
    :param on: Columns that you are planning to merge on.
               These will be checked for float columns and the float columns will be rounded.
               If on is None, then defaults to the intersection of the columns in df1 and df2.
    :param how: type of join (inner, outer, etc)
    :param indicator: what the indicator column should be called (if not False)
    :param decimal_places: number of decimal places to round the floats to. I've found that 10 is good for our data sets so far.
    :return: correctly merged DataFrame
    """
    if on is None:
        on = list(set(df1.columns.values).intersection(set(df2.columns.values)))

    rounded_df1, float_cols_1 = round_float_cols_in_df(df=df1, cols_to_check=on, decimal_places=decimal_places)
    rounded_df2, float_cols_2 = round_float_cols_in_df(df=df2, cols_to_check=on, decimal_places=decimal_places)
    if set(float_cols_1) != set(float_cols_2):
        warnings.warn("float_cols_1 is not the same as float_cols_2 !")
    merged_df = pd.merge(rounded_df1, rounded_df2, how=how, on=on, indicator=indicator)
    return merged_df
