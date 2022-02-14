# ------------------------------------------------------------------------------
# Hippocampus segmentation task from the Medical Segmentation Decathlon 
# (http://medicaldecathlon.com/)
# ------------------------------------------------------------------------------

import os
import numpy as np
import SimpleITK as sitk
from mp.utils.load_restore import join_path
from mp.data.datasets.dataset_segmentation import SegmentationDataset, SegmentationInstance
from mp.paths import storage_data_path
import mp.data.datasets.dataset_utils as du
import torchio
import math

class HippocampusInstance(SegmentationInstance):
    def __init__(self, x_path, y_path, name=None, class_ix=0, group_id=None, prefixList=None):
        # define group and id
        self.name = name
        prefix = name[0]
        split = name[1:].split("_")
        group_id, self.slice_nr = split[-2], split[-1]
        group_id = int(group_id)
        self.pose = 0
        for i in range(len(prefixList)):
            if prefixList[i] == prefix:
                self.pose = i

        super().__init__(x_path, y_path, name, class_ix, group_id)

    def get_subject(self):
        t = torchio.Subject(
            x=self.x,
            y=self.y,
            pose=self.pose,
            group_id=self.group_id,
            name=self.name,
            affine=self.x.affine
        )
        return t
class Hippocampus(SegmentationDataset):
    r"""Class for the hippocampus segmentation decathlon challenge, only for T2,
    found at http://medicaldecathlon.com/.
    """
    def __init__(self, subset=None, hold_out_ixs=[], merge_labels=True, global_name=None, domain_prefixes=None):
        assert subset is None, "No subsets for this dataset."

        if global_name is None:
            global_name = 'Hippocampus'
        dataset_path = os.path.join(storage_data_path, global_name)
        original_data_path = du.get_original_data_path(global_name)

        dataset_path_images = os.path.join(original_data_path, "imagesTr")
        dataset_path_labels = os.path.join(original_data_path, "labelsTr")

        # Fetch all patient/study names
        study_names = set(file_name.split('.nii.gz')[0] for file_name #.split('_gt')[0]
            in os.listdir(dataset_path_images))

        # Build instances
        instances = []
        for study_name in study_names:
            hip_instance = HippocampusInstance(
                x_path=os.path.join(dataset_path_images, study_name+'.nii.gz'),
                y_path=os.path.join(dataset_path_labels, study_name+'.nii.gz'),
                name=study_name,
                group_id=None,
                prefixList = domain_prefixes
                )
            #print(hip_instance.get_subject().x.tensor)
            if math.isnan(hip_instance.get_subject().x.tensor[0][0][0][0]) is False:
                instances.append(hip_instance)
                #y_tens = hip_instance.get_subject().y.tensor
                #hip_instance.get_subject().y.tensor[y_tens != y_tens] = 0
                #print(hip_instance.get_subject().y.tensor)
            else:
                print("Found nan value -> excluded from run, name: " + str(hip_instance.get_subject().pose) + ", id:" + str(
                    hip_instance.get_subject().group_id))
        if merge_labels:
            label_names = ['background', 'hippocampus']
        else:
            label_names = ['background', 'hippocampus upper', 'hippocampus lower']
        super().__init__(instances, name=global_name, label_names=label_names, 
            modality='MR', nr_channels=3, hold_out_ixs=[])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]
