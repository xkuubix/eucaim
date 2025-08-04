#%%
import pandas as pd
import logging
import json
import os

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

# def parse_json_tags(json_info: dict, key: str) -> List[str]:

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
    print(f"JSON keys: {template_dict}")
    return dataframe

def make_header_dict(folders: list) -> dict:
    json_keys = {'subject': set(), 'lesion': set()}
    logger.info('Creating header dictionary template...')
    for folder in folders:
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
    for folder in folders:
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