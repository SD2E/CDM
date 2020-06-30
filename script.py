import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)


def process_file_from_data_converge(file_name, metadata_filename,
                                    variable_columns=None, prediction_column='BL1-A',
                                    log_data=False):
    if variable_columns is None:
        variable_columns = ['strain_name', 'inducer_concentration_mM']

    df = pd.read_json(file_name, orient='records')[['sample_id', prediction_column]]

    df.set_index('sample_id', drop=True, inplace=True)

    meta_df = pd.read_csv(metadata_filename)

    meta_df = meta_df[['sample_id', 'experiment_id'] + variable_columns]

    meta_df.set_index('sample_id', drop=True, inplace=True)

    df = meta_df.join(df)

    final_df = df.explode(prediction_column)

    if log_data:
        final_df[prediction_column] = np.log10(final_df[prediction_column].astype('float') + 1e-10)

    final_df.dropna(inplace=True)
    return final_df


def main():
    filename = 'notebooks/demo_data.json'
    meta_filename = 'notebooks/YeastSTATES-CRISPR-Long-Duration-Time-Series-20191208__meta.csv'
    df = process_file_from_data_converge(file_name=filename, metadata_filename=meta_filename)
    print(df)


if __name__ == '__main__':
    main()
