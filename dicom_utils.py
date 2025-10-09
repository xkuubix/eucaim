import numpy as np
import pydicom

def get_pixels_no_voi(ds, apply_voi=True, lut_index=0):
    """
    Returns pixel data as numpy array.
    - If apply_voi=True and VOI LUT (sequence or function) exists, applies it.
    - Otherwise returns raw pixel_array exactly as stored (no rescale, no windowing).

    Parameters
    ----------
    ds : pydicom.Dataset
        The DICOM dataset.
    apply_voi : bool
        Whether to apply VOI LUT transformation.
    lut_index : int
        Index of LUT to use if multiple VOI LUT Sequence entries exist.

    Returns
    -------
    np.ndarray : The image as uint16 array (0–2**BitsStored-1 range).
    """
    img = ds.pixel_array
    if not apply_voi:
        return img.copy()
    
    bits_stored = int(getattr(ds, 'BitsStored', 12))
    max_val = 2**bits_stored - 1

    # --- Case 1: explicit VOI LUT Sequence ---
    if 'VOILUTSequence' in ds:
        seq = ds.VOILUTSequence
        lut_index = np.clip(lut_index, 0, len(seq)-1)
        lut_item = seq[lut_index]
        
        lut_data = lut_item.LUTData
        lut_desc = lut_item.LUTDescriptor  # [num_entries, first_mapped_pixel_value, bits_per_entry]
        num_entries, first_map, bits_per_entry = [int(x) for x in lut_desc]

        lut_array = np.asarray(lut_data, dtype=np.uint16)
        if len(lut_array) < num_entries:
            lut_array = np.pad(lut_array, (0, num_entries - len(lut_array)), mode='edge')

        img = img.astype(np.int32) - first_map
        img = np.clip(img, 0, num_entries - 1)
        img_voi = lut_array[img]
        return img_voi.astype(np.uint16)

    # --- Case 2: VOI LUT Function (LINEAR / SIGMOID) ---
    voi_lut_func = getattr(ds, 'VOILUTFunction', '').upper()
    if voi_lut_func not in ('LINEAR', 'SIGMOID'):
        # return raw pixels exactly
        return img.copy()

    img = img.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    img = img * slope + intercept


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