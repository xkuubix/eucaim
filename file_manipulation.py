#%%
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
            if name.endswith(".json"):
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
    json_keys = {'subject': set(), 'lesion': set()}
    os.chdir(path)
    folders = sorted(os.listdir(os.getcwd()))
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
            for sub_key in json_info[main_key].keys():
                for value in json_info[main_key]['values'].keys():
                    pass
                    # print(f"{value}:{json_info[main_key]['values'][value]}")
        continue
    os.chdir(original_path)
    print(f"JSON keys: {json_keys}")