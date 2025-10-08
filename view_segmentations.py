# %%
import os
import sys
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
sys.path.append('./code')

def load_image_and_annotation(dicom_path, annotation_path):
    """Load DICOM image and NIfTI annotation (optional)."""
    image = pydicom.dcmread(dicom_path).pixel_array
    annotation = None
    if annotation_path is not None:
        annotation = nib.load(annotation_path).get_fdata()
        if annotation.ndim == 3:
            annotation = annotation[:, :, 0]
    return image, annotation

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
