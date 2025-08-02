#%%
from typing import List, Set
from pydicom.uid import UID
import pydicom
from pydicom import dcmread
import logging
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import os
import json


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# DICOM tags for linking and identification
Referenced_SOP_Instance_UID = (0x8, 0x1155)  # Link to DICOM by ID
SOP_Instance_UID = (0x8, 0x18)               # DICOM ID

# DICOM tags for dicom image properties (pixels)
Imager_Pixel_Spacing = (0x18, 0x1164) # Scaling mm to px
Rows = (0x28, 0x10)                   # Image height in px
Columns = (0x28, 0x11)                # Image width

# DICOM tags for referencing and sequences
Referenced_Frame_of_Reference_Sequence = (0x3006, 0x10)
RT_Referenced_Study_Sequence = (0x3006, 0x12)
RT_Referenced_Series_Sequence = (0x3006, 0x14)
Contour_Image_Sequence = (0x3006, 0x16)

# DICOM tags for ROI and contour sequences
Structure_Set_ROI_Sequence = (0x3006, 0x20)
ROI_Number = (0x3006, 0x22)  # Link between ROI Contour Sequence and Structure Set ROI Sequence
ROI_Name = (0x3006, 0x26)    # User-defined name for ROI

ROI_Contour_Sequence = (0x3006, 0x39)  # Sequence of Contour Sequences defining ROIs
Contour_Sequence = (0x3006, 0x40)      # Sequence of Contours defining ROI
Contour_Data = (0x3006, 0x50)          # Sequence of (x, y, z) triplets defining a contour
Ref_ROI_Number = (0x3006, 0x84)        # Referenced ROI number
#%%
def get_rtstruct(folder: str) -> pydicom.FileDataset:
    """Returns dicom (FileDataset) with rtstruct data from given folder
    of single dicom series
    
    Parameters
    - folder: str or PathLike or file-like object
    """
    dcm_rts = None
    path = None
    for root, _, files in os.walk(folder, topdown=False):
        for name in files:
            if name.endswith(".dcm"):
                dcm = dcmread(os.path.join(root, name), stop_before_pixels=True)
                if dcm.Modality == 'RTSTRUCT':
                    dcm_rts = dcm
                    path = os.path.join(root, name)
    if not dcm:
        logger.warning(f"No DICOM files found in: {folder}.")
    return dcm_rts, path


def parse_ref_sop_uid(dcm_rts: pydicom.FileDataset) -> Set[UID]:
    """Returns Set of pydicom.uid.UID objects, links between DCM and DCM_RT-struct

    Parameters
    - dcm_rts: dicom (FileDataset) with rtstruct data
    """
    ref_sop_uid_mask_set: Set[UID] = set()
    for item in dcm_rts[ROI_Contour_Sequence]:
        # logger.info(item[(0x3006, 0x40)])
        for contours in item[Contour_Sequence].value:
            # logger.info(contours[((0x3006, 0x46))])
            for contour in contours[Contour_Image_Sequence]:
                if contour[Referenced_SOP_Instance_UID].value not in ref_sop_uid_mask_set:
                    ref_sop_uid_mask_set.update([contour[Referenced_SOP_Instance_UID].value])
                    # logger.info(contour[Referenced_SOP_Instance_UID].value) 
    return ref_sop_uid_mask_set

def save_mask(mask: np.ndarray, filename: str):
    """Saves a multilabel mask as a NIfTI file using the nibabel module.
    
    Parameters:
    - mask: numpy array representing the multilabel mask
    - filename: name of the output NIfTI file
    """
    nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), filename)
    logger.info('Saving mask...')
    try:
        nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), filename)
        logger.info('Mask saved successfully: {filename}')
    except Exception as e:
        logger.error(f'Failed to save mask: {str(e)}')

def create_mask(dcm: pydicom.FileDataset, dcm_rts: pydicom.FileDataset,
                ref_sop_uid: pydicom.uid.UID, bitwise_operator="OR") -> np.ndarray:
    """Returns NumPy array mask
    
    Parameters
    - dcm: dicom (FileDataset) data
    - dcm_rts: dicom (FileDataset) with rtstruct data
    - ref_sop_uid: pydicom.uid.UID
    - bitwise_operator: str -> "OR" or "XOR" for joining contour sequences
    """
    image_spacing = dcm[Imager_Pixel_Spacing].value
    image_shape = (dcm[Rows].value, dcm[Columns].value)
    roi_contour_seq = dcm_rts[ROI_Contour_Sequence]
    roi_struct_set = dcm_rts[Structure_Set_ROI_Sequence]
    mask = np.zeros(image_shape, dtype=np.uint8)
    roi_names = []
    roi_dict = {"number": [], "name": [], "mask" :[]}
    
    for item in roi_struct_set:
        roi_dict['name'].append(item.ROIName)
        roi_dict['number'].append(item.ROINumber)
        roi_dict['mask'].append(np.zeros(image_shape, dtype=bool))
    logger.info(f"roi dict {roi_dict['name']}")
    for roi in roi_contour_seq:
        _maxlenpoints = 0
        append_roi_name = False
        for contour_sequence in roi[Contour_Sequence].value:
            for contour in contour_sequence[Contour_Image_Sequence]:
                if contour[Referenced_SOP_Instance_UID].value == ref_sop_uid:
                    append_roi_name = True
                    points = np.array(contour_sequence[Contour_Data].value)
                    if len(points) > _maxlenpoints:
                        _maxlenpoints = len(points)
                    points = points.reshape(-1, 3)
                    points = np.delete(points, obj=2, axis=1) # del empty z axis xy[z] - 01[2] column
                    points = points / image_spacing
                    points = np.floor(points)
                    points = points[:, [1, 0]]
                    mask_ith = ski.draw.polygon2mask(image_shape, points)
                    mask_ith.astype(bool)
                    if bitwise_operator == "OR":
                        roi_dict['mask'][roi[Ref_ROI_Number].value - 1] = np.logical_or(roi_dict['mask'][roi[Ref_ROI_Number].value - 1], mask_ith)
                    elif bitwise_operator == "XOR":
                        roi_dict['mask'][roi[Ref_ROI_Number].value - 1] = np.logical_xor(roi_dict['mask'][roi[Ref_ROI_Number].value - 1], mask_ith)
                    else:
                        raise ValueError
                else:
                    continue
        # print(_maxlenpoints)
        # TO DO: roi name independent saving
        # now: save L(1-3) as 1-3, I as 4, else raise NotImplementedError or Warning
        roi_name = roi_dict['name'][roi[Ref_ROI_Number].value - 1]
        print(f'roi name: {roi_name}')
        if roi_name[0] == 'L':
            value = np.uint8(roi_dict['name'][roi[Ref_ROI_Number].value - 1].split('L')[1])
        elif roi_name == 'I': #[I]gnore (e.g. biopsy markers, etc.)
            value = 4
        else:
            logger.warning(f'Unknown ROI name: {roi_name}! Skipping...')
            continue
        mask[roi_dict['mask'][roi[Ref_ROI_Number].value - 1]==1] = value
        if append_roi_name:
            roi_names.append(roi_name)

    # masks are made in semi-automatic manner, doing some postprocessing
    mask = ski.filters.median(mask) # smoothen edges
    mask = ski.morphology.area_opening(mask) # remove salt
    mask = ski.morphology.area_closing(mask) # remove pepper
    logger.info(f'unique mask values (0 for background): {np.unique(mask)}')
    return mask, roi_names

def plot_mask(dcm: pydicom.FileDataset, mask: np.ndarray)-> None:
    """Plots masks and images
    
    Parameters
    - dcm: dicom (FileDataset) data
    - mask: array with segmentation mask
    """
    image = dcm.pixel_array

    num_rows = 1
    num_cols = 3
    figsize = (dcm.Rows/100, dcm.Columns/100)
    fig, ax = plt.subplots(num_rows, num_cols)
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])

    masked = np.ma.masked_where(mask == 0, mask)
    ax[0].imshow(image, cmap='gray', interpolation='none')

    ax[0].tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)

    ax[1].imshow(image, cmap='gray', interpolation='none')
    ax[1].imshow(masked, cmap='autumn', interpolation='none', alpha=0.5)
    ax[1].tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)

    ax[2].imshow(mask, cmap='gray', interpolation='none')

    ax[2].tick_params(left = False, right = False , labelleft = False ,
                      labelbottom = False, bottom = False)
    plt.show()

def rtstruct2nii(path: str)-> None:
    """
    Convert RTSTRUCT DICOM files to NIfTI format.

    Args:
        path (str): The path to the directory containing the DICOM files.

    Returns:
        None
    """
    original_path = os.getcwd()
    json_keys = set([])
    os.chdir(path)
    folders = sorted(os.listdir(os.getcwd()))
    for i, folder in enumerate(folders):
        logger.info('\n---------------------------------------')
        logger.info(f'current folder path: {folder} ({i+1}/{len(folders)})')
        dcm_rts, seg_path = get_rtstruct(folder)
        seg_save_path = f'{os.path.splitext(seg_path)[0]}.nii.gz'
        if os.path.exists(seg_save_path):
            logger.info(f'Segmentation file already exists: {seg_save_path}')
            continue
        if dcm_rts != None:
            ref_sop_uid_mask_list = parse_ref_sop_uid(dcm_rts)
        for root, _, files in os.walk(folder, topdown=False):
            for name in files:
                if name.endswith(".dcm"):
                    dcm = dcmread(os.path.join(root, name))
                    if dcm.Modality == 'RTSTRUCT':
                        continue
                    for ref_sop_uid in ref_sop_uid_mask_list:
                        if ref_sop_uid == dcm[SOP_Instance_UID].value:
                            mask, rname = create_mask(dcm, dcm_rts, ref_sop_uid)
                            rname_str = '_'.join(rname)
                            save_mask(mask, seg_save_path)
                            # TO DO: each roi in single channel (will not overlap)
                            # TO DO: enable more file formats
                            # plot_mask(dcm, mask)
        # break # if for 1 subject only
    os.chdir(original_path)
# %%