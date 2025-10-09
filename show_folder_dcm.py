# %%
import os
import pydicom
import matplotlib.pyplot as plt

def plot_dicoms_from_folder(folder_path, max_images=6):
    """Plot up to `max_images` DICOM files from a folder, showing SOP Instance UID as the title."""
    
    dicom_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if f.endswith('.dcm')]
    dicom_files = sorted(dicom_files)[:max_images]  # take up to max_images
    
    if not dicom_files:
        print("No DICOM files found in the given folder.")
        return
    
    cols = 3
    rows = (len(dicom_files) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    for i, file_path in enumerate(dicom_files):
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array
        
        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img
        
        sop_uid = ds.SOPInstanceUID if 'SOPInstanceUID' in ds else os.path.basename(file_path)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"SOP UID:\n{sop_uid}", fontsize=8)
        axes[i].axis('off')
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    return ds

def main():
    path = "/users/project1/pt01190/EUCAIM-PG-GUM/UC6/ECI_GUM_S3671/"
    plot_dicoms_from_folder(path)

if __name__ == "__main__":
    main()
# %%