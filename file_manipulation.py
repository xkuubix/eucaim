#%%
from typing import List
import pandas as pd
import pydicom, logging, json, glob, os
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


class Counter:
    count_var = 0  # static counter variable
    def __init__(self):
        pass
    def count():
        """Increments the counter and returns the current count."""
        Counter.count_var += 1
        return Counter.count_var

def get_json_info(folder: os.PathLike) -> dict:
    """Returns json info from given folder of single dicom series
    
    Parameters
    - folder: str or PathLike or file-like object
    """
    
    json_info = {}
    for root, _, files in os.walk(folder, topdown=False):
        for name in files:
            if name.startswith('form_') and name.endswith(".json"):
                with open(os.path.join(root, name), 'r') as f:
                    json_file = json.load(f)
                    json_info.update({name: json_file})
    if not json_info:
        logger.warning(f"No JSON files found in: {folder}.")
        # print(f"Counter: {Counter.count()}")
    return json_info


def make_table(path: str)-> None:
    """
    Convert RTSTRUCT DICOM files to NIfTI format.

    Args:
        path (str): The path to the directory containing the DICOM files.

    Returns:
        None
    """
    original_path = os.getcwd()
    os.chdir(path)
    folders = sorted(os.listdir(os.getcwd()))
    template_dict = make_header_dict(folders)
    dataframe = fill_table(template_dict, folders)
    os.chdir(original_path)
    # print(f"JSON keys: {template_dict}")
    return dataframe


def make_header_dict(folders: list) -> dict:
    json_keys = {'subject': set(), 'lesion': set()}
    logger.info('Creating header dictionary template...')
    for folder in tqdm(folders, desc="Collecting JSON keys"):
        logger.info('\n---------------------------------------')
        logger.info(f'current folder path: {folder}')
        json_info = get_json_info(folder)
        for main_key in json_info.keys():
            if 'subject' in json_info[main_key]['name'].lower():
                json_keys['subject'].update(key for key in json_info[main_key]['values'].keys())
            if ('lesion' in json_info[main_key]['name'].lower()) \
                and ('lesion' in list(json_info[main_key]['values'].keys())):
                json_keys['lesion'].update(key for key in json_info[main_key]['values'].keys())
        template_keys = ['record_id']                                       + \
            ['subject_' + key for key in list(json_keys['subject'])] + \
            ['L1_' + key for key in list(json_keys['lesion'])]       + \
            ['L2_' + key for key in list(json_keys['lesion'])]       + \
            ['L3_' + key for key in list(json_keys['lesion'])]
    return {key: [] for key in template_keys}


def fill_table(template_dict: dict, folders: list) -> pd.DataFrame:
    logger.info('Filling the table with data...')
    all_keys = list(template_dict.keys())
    all_keys.remove('record_id')
    for folder in tqdm(folders, desc="Filling table"):
        logger.info('\n---------------------------------------')
        logger.info(f'current folder path: {folder}')
        json_info = get_json_info(folder)
        row = {key: None for key in all_keys}
        row['record_id'] = folder
        for main_key in json_info.keys():
            if 'subject' in json_info[main_key]['name'].lower():
                for key in json_info[main_key]['values'].keys():
                    row['subject_' + key] = json_info[main_key]['values'][key]
            if ('lesion' in json_info[main_key]['name'].lower()) \
                and ('lesion' in list(json_info[main_key]['values'].keys())):
                lesion_label = json_info[main_key]['values']['lesion']
                for key in json_info[main_key]['values'].keys():
                    template_key = f"{lesion_label}_{key}"
                    if template_key in row:
                        if type(json_info[main_key]['values'][key]) is list \
                        and len(json_info[main_key]['values'][key]) == 1:
                            row[template_key] = json_info[main_key]['values'][key][0]
                        else:
                            row[template_key] = json_info[main_key]['values'][key]
        for key in template_dict:
            template_dict[key].append(row.get(key, None))
    template_dict = additional_columns(template_dict)
    return pd.DataFrame(template_dict)


def additional_columns(template_dict: dict) -> dict:
    lesion_cols = ['L1_lesion', 'L2_lesion', 'L3_lesion']
    lesion_counts = []
    for i in range(len(template_dict['record_id'])):
        count = sum(
            bool(template_dict.get(col, [None])[i])
            for col in lesion_cols
        )
        lesion_counts.append(count)
    template_dict = {k: v for k, v in template_dict.items()}
    template_dict = dict(list(template_dict.items())[:1] +
                        [('lesion_count', lesion_counts)] +
                        list(template_dict.items())[1:])
    return template_dict


def add_dicom_and_annotation_paths(df, dicom_dir):
    views = ["MLO", "CC"]
    laterality = ["L", "R"]

    for v in views:
        for lat in laterality:
            df[f'dicom_path_{v}_{lat}'] = None
            df[f'annotation_path_{v}_{lat}'] = None

    dicom_files = glob.glob(os.path.join(dicom_dir, "**", "resources", "DICOM", "files", "*.dcm"), recursive=True)

    patient_dicoms = {}
    for dcm_path in tqdm(dicom_files, desc="Reading DICOMs"):

        # skip DES (dual energy subtraction) files
        if "_DES/" in dcm_path:
            continue
        try:
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            if dcm.PhotometricInterpretation != "MONOCHROME2":
                continue

            lat = getattr(dcm, "ImageLaterality", None)
            view = getattr(dcm, "ViewPosition", None)
            if view not in views or lat not in laterality:
                continue

            rows = getattr(dcm, "Rows", 0)
            cols = getattr(dcm, "Columns", 0)
            resolution = rows * cols

            record_id = None
            for rid in df['record_id']:
                if rid in dcm_path:
                    record_id = rid
                    break
            if record_id is None:
                continue
            patient_dicoms.setdefault(record_id, []).append((dcm_path, view, lat, resolution))
        except Exception as e:
            print(f"Error reading {dcm_path}: {e}")

    seg_files = glob.glob(os.path.join(dicom_dir, "**", "resources", "annotations", "files", "**", "segmentation_*.nii.gz"), recursive=True)
    seg_info = {os.path.basename(s).replace("segmentation_", "").replace(".nii.gz", ""): s for s in seg_files}

    for idx, row in df.iterrows():
        record_id = row['record_id']
        if record_id not in patient_dicoms:
            continue

        dicoms = patient_dicoms[record_id]

        # group by resolution
        resolution_groups = {}
        for dcm_path, view, lat, resolution in dicoms:
            resolution_groups.setdefault(resolution, []).append((dcm_path, view, lat))

        # find a group with all 4 views
        selected_group = None
        for res, group in sorted(resolution_groups.items(), key=lambda x: -x[0]):  # highest resolution first
            views_present = {(v, l) for _, v, l in group}
            if all((v, l) in views_present for v in views for l in laterality):
                selected_group = group
                break

        if selected_group:
            for dcm_path, view, lat in selected_group:
                abs_dcm_path = os.path.abspath(dcm_path)
                df.at[idx, f'dicom_path_{view}_{lat}'] = abs_dcm_path
                dicom_name = os.path.basename(dcm_path).replace(".dcm", "")
                ann_path = seg_info.get(dicom_name, None)
                if ann_path:
                    abs_ann_path = os.path.abspath(ann_path)
                    df.at[idx, f'annotation_path_{view}_{lat}'] = abs_ann_path
    return df

def drop_ambiguous_rows(df):
    initial_count = len(df)
    #get patientclass 2 with annotation_path not null
    mask = ((df['patientclass'] == 2) & (
        df[['annotation_path_CC_L', 'annotation_path_CC_R', 'annotation_path_MLO_L', 'annotation_path_MLO_R']]
        .notnull().sum(axis=1) > 1
    ))

    dropped = df[mask].copy()
    kept = df[~mask].copy()
    final_count = len(kept)

    print(f"Dropped {initial_count - final_count} ambiguous rows with patientclass 2.")
    return kept, dropped


def drop_na(df, subset_cols: List[str]):
    initial_count = len(df)
    mask = df[subset_cols].isnull().all(axis=1)

    dropped = df[mask].copy()
    kept = df[~mask].copy()
    final_count = len(kept)
    print(f"Dropped {initial_count - final_count} rows with all null values in {subset_cols}.")
    return kept, dropped


def filter_both_views_present_or_absent(df):
    # Check if any CC/MLO annotations are present
    cc_present = df[['annotation_path_CC_L', 'annotation_path_CC_R']].notnull().any(axis=1)
    mlo_present = df[['annotation_path_MLO_L', 'annotation_path_MLO_R']].notnull().any(axis=1)

    # Check that all four dicom paths are present
    all_dicoms_present = df[['dicom_path_CC_L', 'dicom_path_CC_R', 
                             'dicom_path_MLO_L', 'dicom_path_MLO_R']].notnull().all(axis=1)

    # Keep rows where:
    # 1. Both CC and MLO are present, OR
    # 2. Both CC and MLO are absent
    # 3. All four dicoms are present
    keep_mask = ((cc_present & mlo_present) | (~cc_present & ~mlo_present)) & all_dicoms_present

    kept = df[keep_mask].copy()
    dropped = df[~keep_mask].copy()

    print(f"Number of cases kept (both views present or both absent with all dicoms): {len(kept)}")
    return kept, dropped

def drop_class0_no_annotations(df):

    annotation_cols = [
        'annotation_path_MLO_L',
        'annotation_path_MLO_R',
        'annotation_path_CC_L',
        'annotation_path_CC_R'
    ]
    if 'annotation_count' not in df.columns:
        df['annotation_count'] = df[annotation_cols].notnull().sum(axis=1)
    
    dropped = df[(df['patientclass'] == 0) & (df['annotation_count'] == 0)]
    kept = df.drop(dropped.index).copy()
    print(f"\nRemoved {len(dropped)} rows. New size: {len(kept)}")

    return kept, dropped
