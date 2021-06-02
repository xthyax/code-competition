import os
import cv2
import pandas as pd
import numpy as np
import glob
import json
import torch
from torch.utils.data import Dataset
from .utils import preprocess_input, load_and_crop, metadata_count
import random

class DataGenerator(Dataset):
    def __init__(self, input_dir, classes, failClasses, passClasses, input_size, binary_option, testing=False, crop=True, augmentation=None):
        """
            Args:
                input_dir (list): list of input image
                label (list): list of label corresponding to each image
                classes (list): list of class
                height (int) : desire height
                width (int): desire width
                transform: pytorch transforms for transforms and tensor conversion
                augmentation: augment function
        """
        super(DataGenerator, self).__init__()
        if isinstance(input_dir, list):
            self.input_dir = input_dir
        else:
            self.input_dir = [input_dir]
        self.failClasses = failClasses
        self.passClasses = passClasses
        self.binary_option = binary_option
        self.classes = self.load_classes(classes)
        self.num_of_classes = len(classes)
        self.crop = crop
        self.input_size = input_size
        self.img_path_labels = self.load_data()
        self.metadata = metadata_count(self.input_dir, self.classes, self.gt_list, show_table=True)
        self.augmentation= augmentation
        self.testing = testing
    #     self.seeds = None
    #     self.set_up_new_seeds()

    # def set_up_new_seeds(self):
    #     self.seeds = self.get_new_seeds()

    # def get_new_seeds(self):
    #     return np.random.randint(0, 100, len(self))

    def load_classes(self, classes):
        if self.binary_option:
            return ['Reject', 'Pass']
        else:
            return classes  
    def load_data(self):
        img_path_labels = []
        self.gt_list = []
        for path_data in self.input_dir:
            # print(f"[DEBUG] path_data.lower(): {path_data.lower()}")
            if "train" in path_data.lower().split("\\")[-1]:
                path_data = os.path.join(path_data,"OriginImage")
            else:
                pass
            
            # for img_path in glob.glob(os.path.join(path_data,"*.npz")):
            for img_path in glob.glob(os.path.join(path_data,"*.bmp")):
                json_path = img_path + ".json"
                # print(f"[DEBUG] {json_path}")
                try:
                    with open(json_path, encoding='utf-8') as json_file:
                        json_data = json.load(json_file)
                        # print("[DEBUG] Json opened")
                    if self.binary_option:
                        id_image = 'Reject' if json_data['classId'][0] in self.failClasses else 'Pass'
                        # print(f'[DEBUG] {id_image}')
                    else:
                        id_image = json_data['classId'][0]
                    self.gt_list.append(id_image)
                    img_path_labels.append( (img_path, self.classes.index(id_image)) )
                except:
                    img_path_labels.append( (img_path, self.num_of_classes) )
                    print(f"[DEBUG] Missing {json_path}")
        # print(f"[DEBUG] {img_path_labels}")
        return img_path_labels


    def __len__(self):
        return len(self.img_path_labels)
        # return 256

    def __getitem__(self, index):
        # seed = self.seeds[index]
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        image_name = self.img_path_labels[index][0].split("\\")[-1]

        if self.augmentation and torch.randint(0, 2,(1,)).bool().item():
            img_path = os.path.join(self.input_dir[0], "TransformImage", random.choice(self.augmentation)+"_"+image_name)
            # print("[DEBUG] Used augment image")
            # print(random.choice(self.augmentation))
        else:
            # print("[DEBUG] Used origin image")
            img_path = self.img_path_labels[index][0]
        
        try:
            single_label= self.img_path_labels[index][1] # Pytorch don't use one-hot label
        except:
            single_label= self.num_of_classes

        if self.testing:
            return (img_path , single_label)

        else:

            img, _ = load_and_crop(img_path, self.input_size, crop_opt=self.crop)
            img = preprocess_input(img)

            return (img, single_label, img_path)
        # sample = {'img': img, 'label': single_label}
        # sample = preprocess_input(sample)
        # return sample
