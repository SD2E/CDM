"""
The purpose of this module is to load demo data for both CFM and HRM models.
The data can be used for our testing scripts or any other cases where we want to test some functionality quickly.
I made this module for two reasons:
1) Maximize code reuse: I noticed our testing scripts all have virtually identical code to load HRM and CFM data.
2) Create smaller demo data (rows are sampled per condition/replicate) for faster functionality testing.
"""

import os
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
import data
from cdm_src.utils.process_data_converge_files import process_file_from_data_converge
from cdm_src.circuit_fluorescence_model import CircuitFluorescenceModel
from cdm_src.host_response_model import HostResponseModel

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)


# TODO: add functions for fake new data creation here too (use code from test_evaluation_of_predictions.py)


def load_CFM_demo_data(percent: Union[int, float] = 10) -> pd.DataFrame:
    # TODO: implement percentage sampling. It's a bit trickier than expected and no time rn.
    """
    Load all or part of CFM demo data.

    :param percent: Percentage of data to load. (currently this doesn't do anything).
                    - Should be a number that is greater than 0 and less than or equal to 100.
    :return: CFM demo data
    """
    data_path = Path(data.__file__).parent
    json_path = os.path.join(data_path, "demo_data.json")
    meta_data_path = os.path.join(data_path, "YeastSTATES-CRISPR-Long-Duration-Time-Series-20191208__meta.csv")

    cfm_experimental_condition_cols = ['strain_name', 'inducer_concentration_mM']
    cfm_target_col = 'BL1-A'
    cfm_demo_data = process_file_from_data_converge(file_name=json_path,
                                                    metadata_filename=meta_data_path,
                                                    variable_columns=cfm_experimental_condition_cols,
                                                    prediction_column=cfm_target_col)
    cfm_demo_data.reset_index(drop=True, inplace=True)

    # print("\nCFM demo data looks like:")
    # print(cfm_demo_data, "\n")

    return cfm_demo_data


def load_HRM_demo_data(percent: Union[int, float] = 10) -> pd.DataFrame:
    # TODO: implement percentage sampling. It's a bit trickier than expected and no time rn.
    """
    Load all or part of HRM demo data.

    :param percent: Percentage of data to load. (currently this doesn't do anything).
                    - Should be a number that is greater than 0 and less than or equal to 100.
    :return: CFM demo data
    """
    data_path = Path(data.__file__).parent

    hrm_experimental_condition_cols = ["ca_concentration", "iptg_concentration", "va_concentration",
                                       "xylose_concentration", "timepoint_5.0", "timepoint_18.0"]
    hrm_target_col = "logFC_wt"

    # load HRM data:
    relevant_cols = ["Unnamed: 0"] + hrm_experimental_condition_cols + [hrm_target_col]
    hrm_demo_data = pd.read_csv(os.path.join(data_path, "host_response_sparse_embed.csv"),
                                usecols=relevant_cols)[relevant_cols]
    hrm_demo_data.rename(columns={"Unnamed: 0": "gene"}, inplace=True)

    # print("\nHRM data looks like:\n")
    # print(hrm_demo_data, "\n")

    return hrm_demo_data


def create_fake_CFM_exp_data(cfm_target_col='BL1-A', cfm_experimental_condition_cols=None):
    """
    Creates a DataFrame with fake "new experimental data" for testing purposes.
    Does this by instantiating a temporary CFM model and using the future_data it generates as a template.
    Then the template is filled with fake data.
    :return: DataFrame of fake experimental data
    """
    cfm_data = load_CFM_demo_data()
    if cfm_experimental_condition_cols is None:
        cfm_experimental_condition_cols = ['strain_name', 'inducer_concentration_mM']

    temp_cfm = CircuitFluorescenceModel(initial_data=cfm_data, exp_condition_cols=cfm_experimental_condition_cols,
                                        target_col=cfm_target_col)
    new_experiment_data_fake = temp_cfm.future_data.copy()
    # dropping dist_position since future data won't have it. Instead the CFM evaluate method will generate dist_position
    new_experiment_data_fake.drop(columns=["dist_position"], inplace=True)
    cfm_target_min = cfm_data[cfm_target_col].min()
    cfm_target_max = cfm_data[cfm_target_col].max()
    new_experiment_data_fake[cfm_target_col] = np.random.randint(cfm_target_min, cfm_target_max, len(new_experiment_data_fake))
    # create fake replicates
    new_experiment_data_fake["replicate"] = np.random.randint(1, 4, len(new_experiment_data_fake))
    return new_experiment_data_fake


def create_fake_HRM_exp_data(hrm_target_col="logFC_wt", hrm_experimental_condition_cols=None):
    """
    Creates a DataFrame with fake "new experimental data" for testing purposes.
    Does this by instantiating a temporary HRM model and using the future_data it generates as a template.
    Then the template is filled with fake data.
    :return: DataFrame of fake experimental data
    """
    hrm_data = load_HRM_demo_data()
    if hrm_experimental_condition_cols is None:
        hrm_experimental_condition_cols = ["ca_concentration", "iptg_concentration", "va_concentration",
                                           "xylose_concentration", "timepoint_5.0", "timepoint_18.0"]

    temp_hrm = HostResponseModel(initial_data=hrm_data, exp_condition_cols=hrm_experimental_condition_cols,
                                 target_col=hrm_target_col)
    new_experiment_data_fake = temp_hrm.future_data.copy()
    hrm_target_min = hrm_data[hrm_target_col].min()
    hrm_target_max = hrm_data[hrm_target_col].max()
    new_experiment_data_fake[hrm_target_col] = np.random.randint(hrm_target_min, hrm_target_max, len(new_experiment_data_fake))
    return new_experiment_data_fake
