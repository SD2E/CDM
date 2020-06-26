import pandas as pd
import numpy as np

# from Combinatorial_Design_Model import Combinatorial_Design_Model as CDM

def process_file_from_data_converge(filename='demo_data.json',
                                    metadata_filename='YeastSTATES-CRISPR-Long-Duration-Time-Series-20191208__meta.csv',
                                    variable_columns=['strain_name', 'inducer_concentration_mM'],
                                    prediction_column='BL1-A'):
    df = pd.read_json(filename, orient='records')[['sample_id', prediction_column]]

    df.set_index('sample_id', drop=True, inplace=True)

    meta_df = pd.read_csv(metadata_filename)

    meta_df = meta_df[['sample_id', 'experiment_id'] + variable_columns]

    meta_df.set_index('sample_id', drop=True, inplace=True)

    df = meta_df.join(df)

    final_df = df.explode(prediction_column)

    # perform log conversion; omit this step if data is already log converted
    final_df[prediction_column] = np.log10(final_df[prediction_column].astype('float') + 1e-10)

    final_df.dropna(inplace=True)
    return final_df

filename = 'data/demo_data.json'
meta_filename = 'data/YeastSTATES-CRISPR-Long-Duration-Time-Series-20191208__fc_meta.csv'
df = process_file_from_data_converge(filename=filename,metadata_filename=meta_filename)
