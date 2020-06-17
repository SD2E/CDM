import pandas as pd
import numpy as np
from harness.test_harness_class import TestHarness
from CDM_regression import CDM_regression_model
from harness.utils.parsing_results import *
from sklearn.model_selection import train_test_split

class Combinatorial_Design_Model():

    def __init__(self,initial_data,path=".",query={}):
        self.path = path
        self.th = TestHarness(output_location=path)
        '''
        TODO: Put in a check to see if query is blank then build a model, if not, then set the model
        '''
        self.data = initial_data
        self.model = ""

    def build_model(self,train_df=None,test_df=None):
        if train_df is None or test_df is None:
            train_df,test_df = train_test_split(self.data, test_size=0.2)
        '''
        TODO: PUT IN TEST HARNESS TRAIN TEST
        '''



class Host_Response_Model(Combinatorial_Design_Model):

    def __init__(self):
        pass


class Fluorescence_Output_Model(Combinatorial_Design_Model):

    def __init__(self):
        pass

