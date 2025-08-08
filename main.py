#%%
import os
import sys
sys.path.append('./code')
import pandas as pd
from pydicom import dcmread

os.chdir('/users/project1/pt01190/EUCAIM-PG-GUM/code')

# DICOM_DIR = os.path.join('../euc_gumed_u6_v2')
DICOM_DIR = os.path.join('../UC6')
# CSV_FILE = os.path.join('../EuCanImageUseCase68_1.csv')
CSV_FILE = os.path.join('../EuCanImageUseCase68_v2.csv')
SUBJECT_DIR = sorted([
    d for d in os.listdir(DICOM_DIR)
    if os.path.isdir(os.path.join(DICOM_DIR, d))
])

data = pd.read_csv(CSV_FILE,
                   delimiter=',',
                   encoding='utf-8')

subset_data = data[data['patientclass'] == 0]
# %%
from file_manipulation import make_table
if __name__ == "__main__":
    df = make_table(DICOM_DIR)

# %%
merged_data = pd.merge(data, df, how='inner', on='record_id')
# %%
from rts2nii import rtstruct2nii

if __name__ == "__main__":
    rtstruct2nii(DICOM_DIR)

# %%