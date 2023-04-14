import os
import torch
from PIL import Image
from torch.utils.data import Dataset


TRAIN_IMAGES_PATH = "data/train"
TEST_IMAGES_PATH = "data/test"

TRAIN_LABELS_PATH = "data/train.csv"
TEST_LABELS_PATH = "data/test.csv"


class SportDatasetTrain(Dataset):
    def __init__(self,
                 transform=None):
        train_image_to_label = {}
        with open(TRAIN_LABELS_PATH) as fin:
            for line in fin.readlines()[1:]:
                image_and_label = line[:-1].split(",")
                train_image_to_label[
                    os.path.join(TRAIN_IMAGES_PATH, image_and_label[0])
                ] = image_and_label[1]

        self.labels = list(set(train_image_to_label.values()))
        self.data = [
            {"path": path, "label": self.labels.index(label)}
            for path, label in train_image_to_label.items()
        ]  # [:1000]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["path"]).convert("RGB")

        label_tensor = torch.zeros(len(self.labels))
        label_tensor[item["label"]] = 1

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

    def idx_to_str(self, idx):
        return self.labels[idx]


class SportDatasetTest(Dataset):
    def __init__(self,
                 transform=None):
        self.data = []
        with open(TEST_LABELS_PATH) as fin:
            for line in fin.readlines()[1:]:
                self.data.append(os.path.join(TEST_IMAGES_PATH, line[:-1]))
        # self.data = self.data[:300]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    def get_path_by_id(self, idx):
        return self.data[idx]
