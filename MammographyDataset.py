# %% import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors as tvt
from PIL import Image
from torchvision.transforms.functional import hflip
import torch
import os
import pydicom
import nibabel as nib
import numpy as np
import pandas as pd
import pickle
from dicom_utils import get_pixels_no_voi
from file_manipulation import make_long_format

class PatientDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns for DICOM and annotation paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.dataframe['classname'] = self.dataframe['patientclass'].map({2.0: 'normal', 1.0: 'benign', 0.0: 'malignant'})

        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sample = {}

        for view in ['CC_L', 'MLO_L', 'CC_R', 'MLO_R']:
            dicom_path = row.get(f'dicom_path_{view}', None)
            annotation_path = row.get(f'annotation_path_{view}', None)

            if dicom_path and os.path.exists(dicom_path):
                dicom = pydicom.dcmread(dicom_path)
                dicom_image = get_pixels_no_voi(dicom, apply_voi=True)
                bits_stored = pydicom.dcmread(dicom_path, stop_before_pixels=True).BitsStored
                # Normalize
                sample[f'image_{view}'] = torch.from_numpy(dicom_image / (2 ** bits_stored - 1)).float()
            else:
                sample[f'image_{view}'] = None

            if annotation_path and os.path.exists(annotation_path):
                annotation_image = nib.load(annotation_path).get_fdata()
                if annotation_image.ndim == 3:
                    annotation_image = annotation_image[:, :, 0]  # Take the first slice if 3D
                sample[f'annotation_{view}'] = annotation_image
            else:
                sample[f'annotation_{view}'] = None

        sample['patientclass'] = torch.tensor(self.dataframe.iloc[idx]['patientclass'])
        sample['record_id'] = self.dataframe.iloc[idx]['record_id']

        if self.transform:
            pass

        return sample
    
class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns for DICOM and annotation paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        data_long = make_long_format(dataframe, ['record_id', 'patientclass', 'laterality'])
        df_rest = data_long[data_long['patientclass'] != 0]

        # only take malignant breast images in malignant patients
        # 'laterality' 0 = left, 1 = right, is the biopsy side of the malignant lesion
        df_with_lesion = data_long[
            ((data_long['patientclass'] == 0) &
            (data_long['annotation_path'].notna()) &
            (
                ((data_long['laterality'] == 0) & data_long['view'].str.contains('L')) |
                ((data_long['laterality'] == 1) & data_long['view'].str.contains('R'))
            ))
        ]

        data_long_filtered = pd.concat([df_rest, df_with_lesion], axis=0)
        data_long_filtered['classname'] = data_long_filtered['patientclass'].map({2: 'normal', 1: 'benign', 0: 'malignant'})
        self.dataframe = data_long_filtered
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        dicom_path = self.dataframe.iloc[idx]['dicom_path']
        annotation_path = self.dataframe.iloc[idx]['annotation_path'] if 'annotation_path' in self.dataframe.columns else None

        dicom = pydicom.dcmread(dicom_path)
        dicom_image = get_pixels_no_voi(dicom, apply_voi=True)
        bits_stored = pydicom.dcmread(dicom_path, stop_before_pixels=True).BitsStored
        image = torch.from_numpy(dicom_image / (2 ** bits_stored - 1)).float()  # Normalize to [0, 1]

        annotation = None
        if annotation_path and os.path.exists(annotation_path):
            annotation_image = nib.load(annotation_path).get_fdata()
            if annotation_image.ndim == 3:
                annotation_image = annotation_image[:, :, 0]  # Take the first slice if 3D
            # Convert annotation (numpy) to torch Tensor
            annotation = torch.from_numpy(np.asarray(annotation_image)).float()
        
        sample = {}
        sample['patientclass'] = torch.tensor(self.dataframe.iloc[idx]['patientclass'])
        sample['record_id'] = self.dataframe.iloc[idx]['record_id']
        sample['view'] = self.dataframe.iloc[idx]['view'].split('_')[0]  # 'CC' or 'MLO'
        sample['laterality'] = self.dataframe.iloc[idx]['view'].split('_')[1]  # 'L' or 'R'
        
        if sample['laterality'] == 'R':
            # image is already a torch.Tensor (from_numpy). Flip tensors directly.
            image = hflip(image)
            if annotation is not None:
                annotation = hflip(annotation)
        if self.transform:
            pass

        sample['image'] = image
        if annotation is None:
            annotation = torch.zeros_like(image)
        sample['annotation'] = annotation
        return sample

if __name__ == '__main__':
    os.chdir('/users/project1/pt01190/EUCAIM-PG-GUM/code')

    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    patient_dataset = PatientDataset(data)
    print(f'Patient Dataset size: {len(patient_dataset)} samples.')
    patient = patient_dataset[0]
    print(f'Patient Sample keys: {list(patient.keys())}')


    image_dataset = ImageDataset(data)
    print(f'Image Dataset size: {len(image_dataset)} samples.')
    image_sample = image_dataset[0]
    print(f'Image Sample keys: {list(image_sample.keys())}')

# %%
