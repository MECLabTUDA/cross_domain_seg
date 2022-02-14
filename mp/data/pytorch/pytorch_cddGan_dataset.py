import copy
import torch
import torchio
from mp.data.pytorch.pytorch_dataset import PytorchDataset
import mp.data.pytorch.transformation as trans
import mp.eval.inference.predictor as pred
from mp.data.pytorch.pytorch_seg_dataset import PytorchSegmnetationDataset
import numpy as np

class PytorchCDDGanDataset(PytorchSegmnetationDataset):
    r"""Divides images into 2D slices. If resize=True, the slices are resized to
    the specified size, otherwise they are center-cropped and padded if needed.
    """
    def __init__(self, dataset, ix_lst=None, size=(1, 256, 256), 
        norm_key='rescaling', aug_key='standard', channel_labels=True, resize=False):
        if isinstance(size, int):
            size = (1, size, size)
        super().__init__(dataset=dataset, ix_lst=ix_lst, size=size, 
            norm_key=norm_key, aug_key=aug_key, channel_labels=channel_labels)
        assert len(self.size)==3, "Size should be 2D"
        self.resize = resize
        self.predictor = pred.Predictor2D(self.instances, size=self.size, 
            norm=self.norm, resize=resize)

        self.idxs = []
        for instance_ix, instance in enumerate(self.instances):
            for slide_ix in range(instance.shape[-1]):
                self.idxs.append((instance_ix, slide_ix))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        r"""Returns x and y values each with shape (c, w, h)"""
        instance_idx, slice_idx = self.idxs[idx]

        subject = copy.deepcopy(self.instances[instance_idx].get_subject())
        subject.load()

        subject = self.transform_subject(subject)

        x_affine = subject.x.affine

        x = subject.x.tensor.permute(3, 0, 1, 2)[slice_idx]
        y = subject.y.tensor.permute(3, 0, 1, 2)[slice_idx]

        pose = subject.pose
        identity = subject.group_id
        name = subject.name

        if self.resize:
            x = trans.resize_2d(x, size=self.size)
            y = trans.resize_2d(y, size=self.size, label=True)
        else:
            x = trans.centre_crop_pad_2d(x, size=self.size)
            y = trans.centre_crop_pad_2d(y, size=self.size)

        if self.channel_labels:
            # merge labels
            y[y==2] = 1
            y = trans.per_label_channel(y, self.nr_labels)
            
        return x, y, pose, identity, name, x_affine

    def get_subject_dataloader(self, subject_ix):
        dl_items = []
        idxs = [idx for idx, (instance_idx, slice_idx) in enumerate(self.idxs) 
            if instance_idx==subject_ix]
        for idx in idxs:
            x, y, pose, group_id, name, affine = self.__getitem__(idx)
            dl_items.append((x.unsqueeze_(0), y.unsqueeze_(0), pose, group_id, name, affine))
        return dl_items