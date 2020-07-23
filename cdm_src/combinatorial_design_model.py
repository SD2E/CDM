import time
import inspect
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance
from cdm_src.utils.names import Names as N
from cdm_src.harness_models.CDM_regression import CDM_regression_model
from cdm_src.harness_models.CDM_classification import CDM_classification_model
from harness.test_harness_class import TestHarness
from harness.utils.parsing_results import *
from harness.utils.names import Names as Names_TH
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression


class CombinatorialDesignModel(metaclass=ABCMeta):
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="BL1-A"):
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

        if self.leaderboard_query is None:
            # set existing_data and generate future_data
            self.existing_data = self.add_index_per_existing_condition(initial_data)
            self.future_data = self.generate_future_conditions_df()
        else:
            query_matches = query_leaderboard(query=self.leaderboard_query, th_output_location=self.output_path)
            num_matches = len(query_matches)
            if num_matches < 1:
                raise Exception("No leaderboard rows match the query you provided. Here's what the leaderboard looks like:\n"
                                "{}".format(query_leaderboard(query={}, th_output_location=self.output_path)))
            elif num_matches > 1:
                warnings.warn("Your leaderboard query returned {} row matches. "
                              "Only the first match will be used... Here are the matching rows:".format(num_matches),
                              stacklevel=100000)
            else:
                print("Your leaderboard query matched the following row:")
            print(query_matches, "\n")
            run_ids = query_matches[Names_TH.RUN_ID].values
            self.run_id = run_ids[0]
            print("The run_id for the Test Harness run being read-in is: {}".format(self.run_id))
            assert isinstance(self.run_id, str), "self.run_id should be a string. Got this instead: {}".format(self.run_id)
            # Put in code to get the path of the model and read it in once Hamed works that in
            # self.model = ""

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
                           feature_cols_to_use=self.feature_and_index_cols,
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
    def align_predictions_with_new_data(self, predictions_df, new_data_df):
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
        df_preds = df_preds[self.feature_and_index_cols + [target_pred_col]]
        combined_df = self.align_predictions_with_new_data(predictions_df=df_preds, new_data_df=new_data_df)

        ranked_results = self.rank_results(results_df=combined_df, control_col=self.target_col,
                                           prediction_col=target_pred_col, rank_name="ratio_based_ranking")

        score = self.score(x=combined_df[self.target_col], y=combined_df[target_pred_col])
        return score


class HostResponseModel(CombinatorialDesignModel):
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="logFC", gene_col="gene"):
        self.per_condition_index_col = gene_col
        super().__init__(initial_data, output_path, leaderboard_query,
                         exp_condition_cols, target_col)

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

    def align_predictions_with_new_data(self, predictions_df, new_data_df):
        new_data_df = new_data_df[self.feature_and_index_cols + [self.target_col]]
        merged_df = pd.merge(new_data_df, predictions_df, how="inner",
                             on=self.feature_and_index_cols)
        len_preds = len(predictions_df)
        len_new_data = len(new_data_df)
        len_merged = len(merged_df)
        print("Use the following print lines to ensure the inner merge is working correctly...")
        print("len(predictions_df): {}\n"
              "len(new_data_df): {}\n"
              "len(merged_df): {}\n".format(len_preds, len_new_data, len_merged))
        return merged_df

    def score(self, x, y):
        return r2_score(x, y)

    def embed_prior_network(self, df_network, src_node='Source', tgt_node='Target', attrs=['Weight'],
                            emb_dim=32):
        '''
        Provide a dataframe in the form of an edge list and embed the network
        :param df_network:
        :param src_node:
        :param tgt_node:
        :param weight:
        :return:
        '''

        from node2vec import Node2Vec
        import networkx as nx

        G = nx.convert_matrix.from_pandas_edgelist(df_network, src_node, tgt_node, attrs)
        node2vec = Node2Vec(G, dimensions=emb_dim, walk_length=30, num_walks=200, workers=4)
        print("Fitting model...")
        print()
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        df_emb = pd.DataFrame(np.asarray(model.wv.vectors), columns=['embcol_' + str(i) for i in range(emb_dim)], index=G.nodes)
        df_emb.reset_index(inplace=True)
        df_emb.rename({df_emb.columns[0]: self.per_condition_index_col}, axis=1, inplace=True)
        # TODO needd to join to all other datasets --
        return df_emb


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
