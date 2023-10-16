import os

import cv2
import dex
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

dex.eval()


class ImdbCSVDataset(Dataset):
    def __init__(self, root, csv_file, transform=None, return_flip=False):
        self.root = root
        print("reading csv file", csv_file)
        self.labels = pd.read_csv(csv_file)
        self.transform = transform
        self.return_flip = return_flip

    def __len__(self):
        return len(self.labels)

    def read_and_correct_bbox(self, row, img):
        x_min, y_min, x_max, y_max = row.x_min, row.y_min, row.x_max, row.y_max
        x_min, y_min, x_max, y_max = np.round([x_min, y_min, x_max, y_max])
        h, w = img.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        return x_min, y_min, x_max, y_max

    def read_image_age_gender(self, row):
        age = int(row.age)
        if age > 100:
            age = 100
        if age < 0:
            age = 0
        return row.filename, age, row.gender

    def __getitem__(self, index):
        row = self.labels.iloc[index]
        filename, age, gender = self.read_image_age_gender(row)
        fn = os.path.join(self.root, filename)
        img = cv2.imread(fn)
        if img is None:
            raise IOError(f"{fn}")
        bbox = self.read_and_correct_bbox(row, img)
        return filename, img, bbox


if __name__ == "__main__":
    root_dir = "/mnt/trainingdb0/data/age-gender/imdb/imdb-clean-1024/"
    csv_fn = root_dir + "imdb_test_new_1024.csv"
    ds = ImdbCSVDataset(root_dir, csv_fn)

    res = []

    for i in tqdm(range(len(ds))):
        # image, id_label, img_idx, path = ds[i]
        filename, image, bbox = ds[i]
        age, female, male = dex.estimate(image)
        gender = "F" if female > male else "M"
        res.append([filename, age, gender])
    with open(
        "../results/imdb_test_new_1024_dex_age_gender_results.csv",
        "w",
    ) as f:
        f.write("filename,age_pred,gender_pred\n")
        for item in res:
            f.write(",".join(map(str, item)) + "\n")
