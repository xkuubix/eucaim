# %%
import os
import sys
import pydicom
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
sys.path.append('./code')

def load_image_and_annotation(dicom_path, annotation_path):
    """Load DICOM image and NIfTI annotation (optional)."""
    image = pydicom.dcmread(dicom_path)
    image = get_pixels_no_voi(image, apply_voi=True)
    annotation = None
    if annotation_path is not None:
        annotation = nib.load(annotation_path).get_fdata()
        if annotation.ndim == 3:
            annotation = annotation[:, :, 0]
    return image, annotation

def get_pixels_no_voi(ds, apply_voi=True):
    """
    Returns DICOM pixel values as numpy array.
    
    If apply_voi=True and VOI LUT exists, applies it (LINEAR/SIGMOID) and scales to 0-BitsStored.
    If VOI LUT is missing or apply_voi=False, returns **raw pixel_array exactly** as stored, 
    with no changes (no slope/intercept, no windowing).
    """
    img = ds.pixel_array

    # check VOI LUT
    voi_lut_func = getattr(ds, 'VOILUTFunction', None)
    if not apply_voi or voi_lut_func is None:
        # return raw pixels exactly
        return img.copy()
    # otherwise, apply VOI LUT (LINEAR/SIGMOID)

    img = img.astype(np.float32)

    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    img = img * slope + intercept

    bits_stored = int(getattr(ds, 'BitsStored', 12))
    max_val = 2**bits_stored - 1

    voi_lut_func = voi_lut_func.upper()
    window_centers = getattr(ds, 'WindowCenter', None)
    window_widths = getattr(ds, 'WindowWidth', None)

    # pick first WC/WW if multiple (["NORMAL", "HARDER", "SOFTER"])
    wc = float(window_centers[0]) if isinstance(window_centers, (list, pydicom.multival.MultiValue)) else float(window_centers or img.mean())
    ww = float(window_widths[0]) if isinstance(window_widths, (list, pydicom.multival.MultiValue)) else float(window_widths or (img.max() - img.min()))

    if voi_lut_func == 'LINEAR':
        img_voi = np.clip((img - (wc - 0.5 - (ww-1)/2)) / (ww - 1), 0, 1)
    elif voi_lut_func == 'SIGMOID':
        img_voi = 1 / (1 + np.exp(-4 * (img - wc) / ww))
    else:
        raise ValueError(f"Unsupported VOI LUT Function: {voi_lut_func}")

    img_voi_bits = np.round(img_voi * max_val).astype(np.uint16)
    return img_voi_bits

def plot_patient_images(row):
    """Plot images and annotations for one patient row."""
    patient_class = row['patientclass']
    lesion_count = row['lesion_count']
    ID = row["record_id"]
    path = f'images/patient_{ID}.png'
    if os.path.exists(path):
        return  # skip if already exists
    views = ['CC_L', 'MLO_L', 'CC_R', 'MLO_R']
    
    annotations_present = any(row.get(f'annotation_path_{view}', None) is not None for view in views)
    if lesion_count == 0 and annotations_present:
        print(f"!!! Warning: Patient {ID} has lesion_count=0 but annotations exist!")
    elif lesion_count >= 1 and not annotations_present:
        print(f"!!! Warning: Patient {ID} has lesion_count={lesion_count} but no annotations!")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'ID: {ID} Patient Class: {int(patient_class)} (n={lesion_count})', fontsize=16)

    for i, view in enumerate(views):
        dicom_path = row.get(f'dicom_path_{view}', None)
        annotation_path = row.get(f'annotation_path_{view}', None)

        if dicom_path is None:
            axes[0, i].axis('off')
            axes[1, i].axis('off')
            continue

        image, annotation = load_image_and_annotation(dicom_path, annotation_path)

        ax_img = axes[0, i]
        ax_img.imshow(image, cmap='gray')
        ax_img.set_title(view)
        ax_img.axis('off')

        ax_ann = axes[1, i]
        ax_ann.imshow(image, cmap='gray')
        if annotation is not None:
            ax_ann.imshow(annotation, cmap='jet', alpha=0.7)
        ax_ann.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    os.chdir('/users/project1/pt01190/EUCAIM-PG-GUM/code')

    with open('dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        plot_patient_images(row)


# %%
