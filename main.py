#%%
import os
import sys
sys.path.append('./code')
import pandas as pd
from file_manipulation import *
from rts2nii import rtstruct2nii
import pickle

# def main():

if __name__ == '__main__':
    os.chdir('/users/project1/pt01190/EUCAIM-PG-GUM/code')

    # DICOM_DIR = os.path.join('../euc_gumed_u6_v2')
    DICOM_DIR = os.path.join('../UC6')
    CSV_FILE = os.path.join('../EuCanImageUseCase68_v2.csv')
    SUBJECT_DIR = sorted([
        d for d in os.listdir(DICOM_DIR)
        if os.path.isdir(os.path.join(DICOM_DIR, d))
    ])

    data = pd.read_csv(CSV_FILE,
                    delimiter=',',
                    encoding='utf-8')

    subset_data = data[data['patientclass'] == 0]

    print("Creating table from DICOM directories...")
    df = make_table(DICOM_DIR)

    merged_data = pd.merge(data, df, how='inner', on='record_id')

    # rtstruct2nii(DICOM_DIR)
    # print('RTSTRUCT to NIfTI conversion completed.')

    updated_df = add_dicom_and_annotation_paths(merged_data, DICOM_DIR)
    updated_df = drop_na(updated_df, ["patientclass"])
    updated_df = drop_ambiguous_rows(updated_df)
    updated_df = filter_both_views_present_or_absent(updated_df)

    updated_df.reset_index(drop=True, inplace=True)

    fname = 'dataset.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(updated_df, f)
        print(f'File saved as {fname}.')


# %%