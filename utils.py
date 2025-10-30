from typing import Counter, Dict
import re
import pandas as pd
import torchvision.transforms as T
import argparse
from pandas import DataFrame
import torch, random
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from MammographyDataset import *


def get_args_parser():
    default = '/users/project1/pt01190/EUCAIM-PG-GUM/code/config.yml'
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=default,
                        help=help)
    return parser

def reset_seed(SEED=42):
    """Reset random seeds for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def random_split_df(df: DataFrame,
                    train_frac: float,
                    val_frac: float,
                    cal_frac: float,
                    seed: int = 42) -> tuple:
    """
    Split DataFrame into train, val, cal (calibration), and test sets by unique ID.

    Args:
        df: input DataFrame with an 'ID' column.
        train_frac: fraction of data to use for training (e.g., 0.6).
        val_frac: fraction of the *remaining* data for validation.
        cal_frac: fraction of the *remaining* data for calibration.
        seed: random seed for reproducibility.

    Returns:
        Tuple of DataFrames: (train, val, cal, test)
    """
    unique_ids = df['record_id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(n_total * train_frac)

    remaining_ids = unique_ids[n_train:]
    n_remaining = len(remaining_ids)

    n_val = int(n_remaining * val_frac)
    n_cal = int(n_remaining * cal_frac)
    n_test = n_remaining - n_val - n_cal

    print(f"Total unique IDs: {n_total}\n"
          f"Train: {n_train}, Val: {n_val}, Cal: {n_cal}, Test: {n_test}")

    train_ids = unique_ids[:n_train]
    val_ids = remaining_ids[:n_val]
    cal_ids = remaining_ids[n_val:n_val + n_cal]
    test_ids = remaining_ids[n_val + n_cal:]

    train = df[df['record_id'].isin(train_ids)]
    val = df[df['record_id'].isin(val_ids)]
    cal = df[df['record_id'].isin(cal_ids)]
    test = df[df['record_id'].isin(test_ids)]

    return train, val, cal, test


def seed_worker(worker_id):
    """Ensure each worker has a different seed based on the initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloaders(
        train_dataset: Dataset,
        val_dataset: Dataset,
        cal_dataset: Dataset,
        test_dataset: Dataset,
        config: dict,
        g: torch.Generator,
        sampler: WeightedRandomSampler = None
        ) -> Dict[str, DataLoader]:
    params = config['training_plan']['parameters']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True if sampler is None else False,
        num_workers=params['num_workers'],
        worker_init_fn=seed_worker,
        generator=g,
        sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers'],
        worker_init_fn=seed_worker,
        generator= g
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g
    )

    dataloaders: Dict[str, DataLoader] = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    if cal_dataset is not None:
        cal_loader = DataLoader(
            cal_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g
        )
        dataloaders['cal'] = cal_loader

    return dataloaders


def get_fold_dataloaders(config, fold_idx):
    """
    Splits the dataset into training, validation, calibration, and test sets for cross-validation.

    Args:
        config (dict): Configuration dictionary containing data paths and settings.
        fold_idx (int): The fold index to be used for validation.

    Returns:
        dict: A dictionary containing 'train', 'val', 'cal', and 'test' DataLoaders.
    """
    df = pd.read_pickle(config['data']['metadata_path'])
    seed = config['seed']
    k_folds = config.get('data', {}).get('cv_folds', 5)
    train_frac = config.get('data', {}).get('fraction_train', 0.8)
    val_frac = config.get('data', {}).get('fraction_val', 0.1)
    cal_frac = config.get('data', {}).get('fraction_cal', 0.1)

    train_val_df, _, cal_df, test_df = random_split_df(
        df, train_frac, 0, cal_frac, seed=seed
    )

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    indices = list(range(len(train_val_df)))

    train_indices, val_indices = None, None
    for i, (train_idx, val_idx) in enumerate(kf.split(indices)):
        if i == fold_idx:
            train_indices, val_indices = train_idx, val_idx
            break

    if train_indices is None or val_indices is None:
        raise ValueError(f"Invalid fold index {fold_idx}. Must be in range 0-{k_folds - 1}.")

    train_transforms = T.Compose([T.ToTensor()])
    val_test_transforms = T.Compose([T.ToTensor()])

    print(f"Fold {fold_idx + 1}:")
    print(f"Patients No. TRAIN: {len(train_indices)}, VAL: {len(val_indices)}, CAL: {len(cal_df)}, TEST: {len(test_df)}")

    train_dataset = ImageDataset(
        train_val_df.iloc[train_indices],
        transform=train_transforms,
    )
    val_dataset = ImageDataset(
        train_val_df.iloc[val_indices],
        transform=val_test_transforms
    )
    cal_dataset = ImageDataset(
        cal_df,
        transform=val_test_transforms,
    )
    test_dataset = ImageDataset(
        test_df,
        transform=val_test_transforms,
    )


    print("Class counts per set:")
    print(f"  Train set: {Counter(train_dataset.dataframe.classname)}")
    print(f"  Validation set: {Counter(val_dataset.dataframe.classname)}")
    print(f"  Calibration set: {Counter(cal_dataset.dataframe.classname)}")
    print(f"  Test set: {Counter(test_dataset.dataframe.classname)}")

    g = torch.Generator()
    g.manual_seed(seed)

    return create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        cal_dataset=cal_dataset,
        config=config,
        g=g
    )


def get_fold_number(model_name: str) -> int:
    # regex looks for 'fold_' followed by one or more digits
    match = re.search(r"fold_(\d+)", model_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No fold number found in '{model_name}'")