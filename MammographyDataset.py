# %% import torch
import cv2
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
import cv2


def crop_to_breast(img, annot, threshold=0.05):
    #  [H,W]
    if img.dim() == 3:
        gray = img.mean(0)
    else:
        gray = img
    
    # binary mask
    mask = gray > threshold
    
    # remove tiny objects by keeping largest connected component
    mask_np = mask.cpu().numpy().astype('uint8')
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
    if num_labels <= 1:  # no objects
        return img
    largest_idx = stats[1:, cv2.CC_STAT_AREA].argmax() + 1  # +1 because 0 is background
    largest_mask = (labels == largest_idx)
    
    largest_mask = torch.tensor(largest_mask, dtype=torch.uint8)  # if it's numpy
    coords = largest_mask.nonzero(as_tuple=False)
    mins = coords.min(0)[0]
    maxs = coords.max(0)[0]
    y_min, x_min = mins[0].item(), mins[1].item()
    y_max, x_max = maxs[0].item(), maxs[1].item()
    
    return img[y_min:y_max+1, x_min:x_max+1], annot[y_min:y_max+1, x_min:x_max+1]

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
            annotation[annotation > 1] = 0.  # Binarize annotation
        
        sample = {}
        sample['classname'] = self.dataframe.iloc[idx]['classname']
        sample['patientclass'] = torch.tensor([1 if sample['classname'] == 'malignant' else 0]).long()  # Binary label for malignant vs non-malignant
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
        # sample['image'] = image
        if annotation is None:
            annotation = torch.zeros_like(image)
        sample['image'], sample['annotation'] = crop_to_breast(image, annotation)
        # sample['annotation'] = annotation
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
