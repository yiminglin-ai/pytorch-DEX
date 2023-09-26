import logging
import os

import numpy as np

import cv2
import dex
import mxnet as mx
from torch.utils.data import Dataset
from tqdm import tqdm

dex.eval()

GENDER_MAP = {
    "F": 0,
    "M": 1,
}


class MXFaceDataset(Dataset):
    """
    Mxnet RecordIO face dataset.
    """

    def __init__(self, root_dir: str, transforms=None, **kwargs) -> None:
        super(MXFaceDataset, self).__init__()
        self.transform = transforms
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, "test.rec")
        path_imgidx = os.path.join(root_dir, "test.idx")
        path_imglst = os.path.join(root_dir, "test.lst")
        items = [
            line.strip().split("\t") for line in open(path_imglst, "r")
        ]  # img_idx, 0, img_path

        self.img_idx_to_path = {int(item[0]): item[-1] for item in items}
        # path_landmarks = os.path.join(root_dir, "landmarks.csv")
        # self.path_to_landmarks = _read_landmarks(path_landmarks)

        logging.info("loading recordio %s...", path_imgrec)
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        logging.info("loading recordio %s done", path_imgrec)
        self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        img_idx = self.imgidx[index]
        s = self.imgrec.read_idx(img_idx)
        header, sample = mx.recordio.unpack_img(s, cv2.IMREAD_UNCHANGED)
        if self.transform is not None:
            sample = self.transform(sample)

        return (
            sample,
            header.label[0],
            img_idx,
            self.img_idx_to_path[img_idx],
        )

    def __len__(self):
        return len(self.imgidx)


if __name__ == "__main__":
    root_dir = "/mnt/trainingdb0/data/face-recognition/internal.face-verification/v5.1/multilabel_5_1_cache_2/test/"
    ds = MXFaceDataset(root_dir)

    res = []

    for i in tqdm(range(len(ds))):
        image, id_label, img_idx, path = ds[i]
        age, female, male = dex.estimate(image)
        gender = 0.0 if female > male else 1.0
        res.append([img_idx, id_label, age, gender, path])

    # save to lst
    with open("multilabel_5_1_cache_2_test_dex_age_gender_results.lst", "w") as f:
        for item in res:
            f.write("\t".join(map(str, item)) + "\n")
