import numpy as np
import pydicom

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