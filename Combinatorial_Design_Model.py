import pandas as pd
import numpy as np
from harness.test_harness_class import TestHarness
from CDM_regression import CDM_regression_model
from harness.utils.parsing_results import *
from harness.utils.names import Names as Names_TH
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance

class Combinatorial_Design_Model():

    def __init__(self,initial_data="",path=".",query={},**th_properties):
        #TODO: build a way to check if user wants to run something that they already ran.
        #TODO: and read that instead of re-running test harnness

        self.path = path
        self.th = TestHarness(output_location=path)
        if not query:
            self.data = initial_data
            self.build_model(th_properties)
        else:
            self.model_id = query_leaderboard(query=query,th_output_location=path)[Names_TH.RUN_ID]
            #Put in code to get the path of the model and read it in once Hamed works that in
            self.model = ""

    def build_model(self,train_df=None,test_df=None,th_properties={}):
        if train_df is None or test_df is None:
            train_df,test_df = train_test_split(self.data, test_size=0.2)

        #TODO: Fix up test/harness run
        self.th.run_custom(function_that_returns_TH_model=CDM_regression_model, dict_of_function_parameters={},
                      training_data=train_df, testing_data=test_df,
                      data_and_split_description=th_properties[Names_TH.DATA_AND_SPLIT_DESCRIPTION],
                      cols_to_predict=th_properties['cols_to_predict'], feature_cols_to_use=th_properties['feature_cols_to_use'],
                      index_cols=th_properties['index_cols'], normalize=False, feature_cols_to_normalize=None)
        #TODO: Put in code to read in model once test harness writes it out
        self.model = ""

    def evaluate_model(self,df,index_col_new_data,target_col_new_data,index_col_predictions_data,custom_metric=None):
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
        df_all = join_new_data_with_predictions(df,index_col_new_data,index_col_predictions_data,{Names_TH.RUN_ID:self.model_id}, self.path, loo=False, classification=False,file_type=Names_TH.PREDICTED_DATA)
        for col in df_all.columns:
            if '_predictions' in col:
                pred_col = col

        if custom_metric=='r2':
            return r2_score(df_all[pred_col],df_all[target_col_new_data])
        elif custom_metric == 'emd':
            return wasserstein_distance(df_all[pred_col],df_all[target_col_new_data])
        else:
            #TODO: ensure custom metric_can take at least two arguments
            return custom_metric(df_all[pred_col],df_all[target_col_new_data])

class Host_Response_Model(Combinatorial_Design_Model):

    def __init__(self):
        pass


class Fluorescence_Output_Model(Combinatorial_Design_Model):

    def __init__(self):
        pass

