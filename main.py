#%%
import os
import sys
sys.path.append('./code')
import pandas as pd
from pydicom import dcmread

DICOM_DIR = os.path.join('euc_gumed_u6_v2')
CSV_FILE = os.path.join('EuCanImageUseCase68_1.csv')
SUBJECT_DIR = sorted(os.listdir(DICOM_DIR))


data = pd.read_csv(CSV_FILE,
                   delimiter=';',
                   encoding='utf-8')

subset_data = data[data['patientclass'] == 0]
# %%
from file_manipulation import make_table
if __name__ == "__main__":
    make_table(DICOM_DIR)

# %%
from rts2nii import rtstruct2nii

if __name__ == "__main__":
    rtstruct2nii(DICOM_DIR)


# %%