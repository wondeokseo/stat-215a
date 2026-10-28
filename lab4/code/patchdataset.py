import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from clean import clean_data


TRAIN_IMAGES = ["O013257.npz", "O012791.npz"]
TEST_IMAGES = ["O013490.npz"]


class PatchDataset(Dataset):
    """
    Wrapper dataset class for our datasets of patches.
    """

    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        sample = self.patches[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_image(filename, data_dir="../data"):
    path = Path(data_dir) / filename
    return np.load(path)["arr_0"]


def load_images(filenames, data_dir="../data"):
    return [load_image(name, data_dir) for name in filenames]


def get_unlabeled_files(data_dir="../data"):
    labeled = set(TRAIN_IMAGES + TEST_IMAGES)

    return sorted(
        p.name for p in Path(data_dir).glob("*.npz")
        if p.name not in labeled
    )


def get_labeled_data(filenames, features=None, data_dir="../data"):
    images = load_images(filenames, data_dir)
    images = clean_data(images)

    data = np.vstack(images)
    data = data[data[:, 10] != 0]

    X = data[:, 2:10]
    y = data[:, 10]

    if features is not None:
        X = X[:, features]

    return X, y


def get_unlabeled_data(features=None, data_dir="../data"):
    filenames = get_unlabeled_files(data_dir)
    images = load_images(filenames, data_dir)
    images = clean_data(images)

    data = np.vstack(images)
    X = data[:, 2:10]

    if features is not None:
        X = X[:, features]

    return X


def add_features(X, new_features):
    return np.column_stack([X, new_features])