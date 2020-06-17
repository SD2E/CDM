import pandas as pd
import numpy as np
from harness.test_harness_class import TestHarness
from CDM_regression import CDM_regression_model
from harness.utils.parsing_results import *
from harness.utils.names import Names as Names_TH
from sklearn.model_selection import train_test_split

class Combinatorial_Design_Model():

    def __init__(self,initial_data="",path=".",query={},**th_properties):
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


class Host_Response_Model(Combinatorial_Design_Model):

    def __init__(self):
        pass


class Fluorescence_Output_Model(Combinatorial_Design_Model):

    def __init__(self):
        pass

