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
   
    all_dropped = []
    updated_df, dropped_na = drop_na(updated_df, ["patientclass"])
    updated_df, dropped_ambiguous = drop_ambiguous_rows(updated_df)
    updated_df, dropped_views = filter_both_views_present_or_absent(updated_df)
    updated_df, dropped_class0 = drop_class0_no_annotations(updated_df)
    updated_df.reset_index(drop=True, inplace=True)
    all_dropped.append(dropped_na)
    all_dropped.append(dropped_ambiguous)
    all_dropped.append(dropped_views)  # we can use the dropped cases for pretraining of C() and S()
    all_dropped.append(dropped_class0) # we can use the dropped class0 cases for pretraining of C()

    dropped_all_df = pd.concat(all_dropped, ignore_index=True)

    fname = 'dataset.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(updated_df, f)
        print(f'Saved {fname}.')

    fname = 'dropped_rows.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(dropped_all_df, f)
        print(f'Saved {fname}.')

# %%