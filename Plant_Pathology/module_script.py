import sys
sys.path.append("..")
import os
os.environ["MKL_NUM_THREADS"] = "3" # "6"
os.environ["OMP_NUM_THREADS"] = "2" # "4"
os.environ["NUMEXPR_NUM_THREADS"] = "3" # "6"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from utils.custom_dataloader import FastDataLoader

#Preliminaries
from tqdm import tqdm
import math
import random
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# Visuals and CV2
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import glob

# print(data.head(10))
# print("blank space")
# data = data[data['fold']!=0].reset_index(drop=True)
# p_data = data.reset_index()
# row = p_data.iloc[123]
# print(row)
# sys.exit(0)


# torch relevant

# albumentations for augs
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import ttach as tta
import torch
from classification_nn_pytorch import EfficientNet
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
    

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# def f1_score(y_true, y_pred):
#     y_true = y_true.apply(lambda x: set(x.split()))
#     y_pred = y_pred.apply(lambda x: set(x.split()))
#     intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
#     len_y_true = y_true.apply(lambda x: len(x)).values
#     len_y_pred = y_pred.apply(lambda x: len(x)).values
#     f1 = 2 * intersection / (len_y_true + len_y_pred)
    
#     return f1

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0 
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def fetch_loss(multi_labels=False):
    if multi_labels:
        loss = nn.MultiLabelSoftMarginLoss()
    
    else:
        loss = nn.CrossEntropyLoss()
    
    return loss

class Preprocess:
    def __init__(self, img_size):
        self.img_size = img_size

    def get_train_transforms(self):
        return A.Compose(
            [
                A.OneOf([
                    A.RGBShift(),
                    A.ChannelShuffle(),
                ], p=0.2),

                A.RandomResizedCrop(self.img_size, self.img_size, scale=(0.9, 1), p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
                A.CLAHE(clip_limit=(1,4), p=0.5),

                A.OneOf([
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.7),
                    A.ElasticTransform(alpha=3),
                ], p=0.2),

                A.OneOf([
                    A.GaussNoise(var_limit=[10, 50]),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                    A.MedianBlur(),
                ], p=0.2),

                A.OneOf([
                    # A.RandomSunFlare(), 
                    A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.1),
                    A.RandomBrightness(limit=0.3, p=1),
                ], p=0.2),

                A.Resize(self.img_size, self.img_size, always_apply=True),

                A.OneOf([
                    A.JpegCompression(),
                    A.Downscale(scale_min=0.1, scale_max=0.15),
                ], p=0.2),

                A.IAAPiecewiseAffine(p=0.2),
                A.IAASharpen(p=0.2),
                A.Cutout(max_h_size=int(self.img_size * 0.1), max_w_size=int(self.img_size * 0.1), num_holes=5, p=0.5),
                A.Normalize(),
                ToTensorV2(p=1.0)
            ]
        )

    def get_valid_transforms(self):
        
        return A.Compose(
        [
            A.Resize(self.img_size, self.img_size, always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ])

class PPDataset(Dataset):
    def __init__(self, csv, classes, transforms=None, training=True):
        self.csv = csv.reset_index()
        self.classes = classes
        self.augmentations = transforms
        self.training = training

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
            
        if self.training:
            return image, torch.tensor(row[1:len(self.classes) + 1])
        
        else:
            return image, torch.tensor(1)

class Create_model:
    def __init__(self, CFG, pretrain_option=True):
        self.config = CFG
        self.pretrain_option = pretrain_option

    def _build_model(self):

        if self.pretrain_option:
            print("Loading pretrained :", self.config.pretrained_path)
            model = EfficientNet.from_pretrained(
                                                    self.config.model_name,
                                                    weights_path=self.config.pretrained_path,
                                                    advprop=False,
                                                    num_classes=len(self.config.classes),
                                                    image_size=self.config.img_size
                                                )
        
        else:
            print(f'Buidling Model Backbone for {self.config.model_name} model.')
            model =  EfficientNet.from_name(
                                                self.config.model_name,
                                                num_classes=len(self.config.classes),
                                                image_size=self.config.img_size
                                            )

        return model

class PPScheduler(_LRScheduler):
    def __init__(self, 
                optimizer, 
                lr_start=5e-6, 
                lr_max=1e-5, 
                lr_min=1e-6, 
                lr_ramp_ep=5, 
                lr_sus_ep=0, 
                lr_decay=0.8, 
                last_epoch=-1):

        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        super(PPScheduler, self).__init__(optimizer,last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warings.warn("To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            self.last_epoch += 1

            return [self.lr_start for _ in self.optimizer.param_groups]
        
        lr = self._compute_lr_from_epoch()
        self.last_epoch += 1

        return [lr for _ in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        return self.base_lrs

    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.lr_ramp_ep:
            lr = ((self.lr_max - self.lr_start) / 
            self.lr_ramp_ep * self.last_epoch +
            self.lr_start)

        elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max
        
        else:
            lr = ((self.lr_max - self.lr_min) * self.lr_decay ** (self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep) + self.lr_min)

        return lr    

class Main_session:
    def __init__(self, GET_CV, CFG):
        self.GET_CV = GET_CV
        self.config = CFG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tta_option = tta.Compose([
                    tta.Rotate90([0, 90, 180, 270])
                ])
        GPU_COUNT = torch.cuda.device_count()
        print(f"Found {GPU_COUNT} GPUs")

    def read_dataset(self):
        if self.GET_CV:
            df = pd.read_csv(os.path.join(self.config.root, 'train.csv'), index_col='image')
            init_len = len(df)
            
            with open(self.config.dup_csv_path, 'r') as file:
                duplicates = [x.strip().split(",") for x in file.readlines()]

            for row in duplicates:
                unique_labels = df.loc[row].drop_duplicates().values
                if len(unique_labels)  == 1:
                    df = df.drop(row[1:], axis=0)

                else:
                    df = df.drop(row, axis=0)

            print(f"Dropping {init_len - len(df)} duplicates samples")

            original_labels = df['labels'].values.copy()
            
            df['labels'] = [x.split(' ') for x in df['labels']]

            labels = MultiLabelBinarizer(classes=self.config.classes).fit_transform(df['labels'].values)

            df = pd.DataFrame(columns=self.config.classes, data=labels, index=df.index)
            
            skf = StratifiedKFold(n_splits=self.config.folds, shuffle=True, random_state=self.config.seed)
            # skf.get_n_splits(df.index, df['label_group'])
            np_fold = np.zeros((len(df),))
            df['fold'] = 0

            for i, (train_index, test_index) in enumerate(skf.split(df.index, original_labels)):

                df['fold'].iloc[test_index] = i
                np_fold[test_index] = i

            value_counts = lambda x: pd.Series.value_counts(x, normalize=True)
            
            df_check = df.drop(columns=['fold'])
            # print(df_check[np_fold == 0].apply(value_counts).loc[1])

            df_occurence = pd.DataFrame({
                "origin": df_check.apply(value_counts).loc[1],
                'fold_0': df_check[np_fold == 0].apply(value_counts).loc[1],
                'fold_1': df_check[np_fold == 1].apply(value_counts).loc[1],
                'fold_2': df_check[np_fold == 2].apply(value_counts).loc[1],
                'fold_3': df_check[np_fold == 3].apply(value_counts).loc[1],
                'fold_4': df_check[np_fold == 4].apply(value_counts).loc[1],
            })

            # print(df_occurence)
            df['image'] = df.index
            df['origin_labels'] = original_labels
            df['filepath'] = df['image'].apply(lambda x: os.path.join(self.config.root, 'train_images',x))

        else:
            test_path = glob.glob(os.path.join(self.config.root, "test_images","*"))
            test_images = [path.split("/")[-1] for path in test_path]
            df = pd.DataFrame(columns=['image'], data=test_images)

            df['filepath'] = df['image'].apply(lambda x: os.path.join(self.config.root, 'test_images',x))

        return df

    def train_fn(self, dataloader, model, criterion, optimizer, device, epoch):
        model.train()
        loss_score = AverageMeter()
        acc_score = AverageMeter()

        tk0 = tqdm(enumerate(dataloader), total = len(dataloader))
        for bi,d in tk0:
            
            batch_size = d[0].shape[0]
            
            images = d[0]
            targets = d[1]

            if self.config.num_gpus == 1:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)

            targets = targets.to(outputs.device, non_blocking=True)
            # loss = criterion(outputs, torch.max(targets, 1)[1])
            loss = criterion(outputs.float(), targets.float())

            _, preds = torch.max(outputs, 1)
            _, targets = torch.max(targets, 1)

            loss.backward()
            optimizer.step()

            loss_score.update(loss.detach().item() * batch_size, batch_size)
            acc_score.update((preds == targets).sum().item(), batch_size)

            # print("Optimizer param : ", optimizer.param_groups)
            tk0.set_postfix(Train_Loss=loss_score.avg, Train_acc=acc_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

        return loss_score

    def eval_fn(self, data_loader, model, criterion, device):
        
        # Evaluate metric part
        custom_dict =   {
            "y_pred":   [],
            "y_gtruth": []
                        }
        loss_score = AverageMeter()
        acc_score = AverageMeter()

        model.eval()
        tk0 = tqdm(enumerate(data_loader), total=len(data_loader))

        if self.config.num_gpus > 1:
            evaluate_model = model.module
            evaluate_model.eval()
            tta_model = tta.ClassificationTTAWrapper(evaluate_model, self.tta_option)
            tta_model = nn.DataParallel(tta_model)

        else:
            evaluate_model = model
            tta_model = tta.ClassificationTTAWrapper(evaluate_model, self.tta_option)

        tta_model.eval()

        with torch.no_grad():
            for bi,d in tk0:
                batch_size = d[0].size()[0]

                images = d[0]
                targets = d[1]

                if self.config.num_gpus == 1:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                outputs = tta_model(images)
            
                targets_ml = targets.to(outputs.device, non_blocking=True)
                # loss = criterion(outputs, torch.max(targets, 1)[1])
                loss = criterion(outputs.float(), targets_ml.float())

                _, preds = torch.max(outputs, 1)
                _, targets = torch.max(targets_ml, 1)

                propability = torch.nn.Softmax(dim = 1)(outputs)
                propability = propability.detach().cpu().numpy()

                custom_dict["y_pred"].extend( (propability > 1 / len(self.config.classes)).astype(np.int32).tolist() )
                custom_dict["y_gtruth"].extend(targets_ml.tolist())

                loss_score.update(loss.detach().item() * batch_size, batch_size)
                acc_score.update((preds == targets).sum().item(), batch_size)

                tk0.set_postfix( Eval_Train=acc_score.avg, Eval_Loss=loss_score.avg)

        f1 = f1_score(np.array(custom_dict["y_gtruth"]), np.array(custom_dict["y_pred"]), average="micro")
        print("F1 score on validate set: {:.5}".format(f1))

        try:
            del evaluate_model
        except NameError:
            pass

        return loss_score, f1

    def run_training(self, continue_training=False, custom_metric=True):
        data = self.read_dataset()

        list_models = [0] * len(data['fold'].unique())
        for i in range(len(data['fold'].unique())):
        # for i in [0]:
            model_path = os.path.join(self.config.models_path, f'model_{self.config.model_name}_IMG_SIZE_{self.config.img_size}_f{i}.pth')
            log_path = f"{model_path}.txt"

            if custom_metric:
                try:
                    log_file = open(log_path, "r")
                    lineList = log_file.readlines()
                    best_value = max( [float(line.split("F1_val: ")[1].split(" - ")[0]) for line in lineList] )

                except:
                    log_file = open(log_path, "w")
                    best_value = 0

            else:
                try:
                    log_file = open(log_path, "r")
                    lineList = log_file.readlines()
                    best_val_loss = min( [float(line.split("Valid_loss: ")[1].strip("\n")) for line in lineList] )

                except:
                    log_file = open(log_path, "w")
                    best_val_loss = 100

            log_file.close()
            logs = []

            train = data[data['fold']!=i].reset_index(drop=True)
            valid = data[data['fold']==i].reset_index(drop=True)

            define_preprocess = Preprocess(self.config.img_size)
            # Defining Dataset
            train_dataset = PPDataset(
                csv = train,
                classes= self.config.classes,
                transforms = define_preprocess.get_train_transforms(),
            )

            valid_dataset = PPDataset(
                csv = valid,
                classes= self.config.classes,
                transforms = define_preprocess.get_valid_transforms(),
            )
            
            seed_torch(seed=self.config.seed)
            train_loader = FastDataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                pin_memory=False,
                drop_last=True,
                num_workers=self.config.num_workers
            )
            seed_torch(seed=self.config.seed)
            valid_loader = FastDataLoader(
                valid_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
                num_workers=self.config.num_workers
            )
            
            if continue_training:
                if os.path.isfile(model_path):
                    model = Create_model(self.config, False)
                    list_models[i] = model._build_model()

                    print(f"Loading trained weights {model_path} to model...")
                    list_models[i].load_state_dict(torch.load(model_path))

                else:
                    model = Create_model(self.config, True)
                    list_models[i] = model._build_model()
                
            if self.config.num_gpus > 1:
                list_models[i] = list_models[i].to(device=self.device)
                list_models[i] = nn.DataParallel(list_models[i])
            
            else:
                list_models[i].to(device=self.device)

            # Defining criterion
            criterion = fetch_loss(multi_labels=True)
            criterion.to(device=self.device)

            ### Scheduler and its params ###
            scheduler_params = {
                "lr_start" : self.config.lr_rate,
                "lr_max" : 1e-4 * self.config.batch_size,
                "lr_min" : 1e-6,
                "lr_ramp_ep" : 5,
                "lr_sus_ep" : 0,
                "lr_decay" : 0.8,    
            }
            optimizer = Adam(list(list_models[i].parameters()), lr=scheduler_params['lr_start'])

            # Defining LR Scheduler
            # scheduler = PPScheduler(optimizer, **scheduler_params)
            scheduler = ReduceLROnPlateau(optimizer, mode='max' if custom_metric else 'min', 
                                            patience=self.config.patient // 2, verbose=True)

            # THE ENGINE LOOP
            early_s_count = 0
            compare_value = best_value if custom_metric else best_val_loss

            for epoch in range(1, self.config.epochs + 1):

                seed_torch(seed=epoch)
                train_loss = self.train_fn(train_loader, list_models[i], criterion, optimizer, self.device, epoch=epoch)
                
                valid_loss, f1 = self.eval_fn(valid_loader, list_models[i], criterion, self.device)
                
                print("* " * 30 )

                # Write training log into txt file
                with open(log_path, "a") as file:
                    file.write(f"Epoch: {epoch} - LR: {optimizer.param_groups[0]['lr']} - Train_loss: {train_loss.avg} - F1_val: {f1} - Valid_loss: {valid_loss.avg}\n")
                
                # Schedule learning rate
                scheduler.step(f1 if custom_metric else valid_loss.avg)

                early_s_count, compare_value = self.check_point(compare_value, 
                                                                f1 if custom_metric else valid_loss.avg, 
                                                                model=list_models[i],
                                                                model_path=model_path,
                                                                patient_count=early_s_count,
                                                                mode='max'
                                                                )

                # Early stopping
                if self.config.patient == early_s_count:
                    break

            torch.cuda.empty_cache()

    def check_point(self, compare_value, running_value, model, model_path, patient_count=5, mode='min'):
        
        assert mode == 'min' or mode == 'max', f"Unsupport {mode} mode"
        save_flag = 0

        if mode == 'min' and running_value < compare_value:
            compare_value = running_value
            patient_count = 0
            save_flag = 1

        elif mode == 'max' and  running_value > compare_value:
            compare_value = running_value
            patient_count = 0
            save_flag = 1

        else:
            patient_count += 1

        if self.config.num_gpus == 1 and save_flag:
            print("Saving model...")
            torch.save(model.state_dict(), model_path)

        elif self.config.num_gpus > 1 and save_flag :
            print("Saving multi-gpus model...")
            torch.save(model.module.state_dict(), model_path)

        else:
            pass

        return patient_count, compare_value

    def get_predict_results(self):
        data = self.read_dataset()
        data = data.reset_index(drop=True)
        
        list_models = [0] * self.config.folds
        
        probs_dict = {}
        
        for i in [0]:
            probs_list = []
            preds_list = []
            
            model = Create_model(self.config, pretrain_option=False)
            
            list_models[i] = model._build_model()
            model_path = os.path.join(self.config.models_path, f'model_{self.config.model_name}_IMG_SIZE_{self.config.img_size}_f{i}.pth')
            
            list_models[i].load_state_dict(torch.load(model_path), strict=False)
            list_models[i].to(device=self.config.device)
            list_models[i].eval()
            
            if self.config.num_gpus > 1:
                list_models[i] = tta.ClassificationTTAWrapper(list_models[i].module, self.tta_option)
                list_models[i] = nn.DataParallel(list_models[i])

            else:
                list_models[i] = tta.ClassificationTTAWrapper(list_models[i], self.tta_option)
            
            list_models[i].eval()
            
            image_dataset = PPDataset(data, transforms=Preprocess(self.config.img_size).get_valid_transforms(), training= self.GET_CV)

            image_loader = DataLoader(
                image_dataset,
                batch_size=self.config.batch_size,
                pin_memory=True,
                drop_last=False,
                num_workers=self.config.num_workers
            )

            with torch.no_grad():
                for img, label in tqdm(image_loader):
                    img = img.to(self.device)
                    label = label.to(self.device)
                    outputs = list_models[i](img)

                    propability = torch.nn.Softmax(dim = 1)(outputs)
                    propability = propability.detach().cpu().numpy()
                    probs_list.append(propability)
                    preds_list.extend([np.array(self.config.classes)[np.where(prob > 1 / len(self.config.classes))] for prob in propability])
                    
            del model
            probs_list = np.concatenate(probs_list)
            preds_list = np.array(preds_list)
            probs_dict[f"fold_{i}"] = probs_list
            data[f"fold_{i}"] = preds_list
            import gc
            gc.collect()
            
        return probs_dict, data


def combine_predictions(row):
    temp_fold = [0]
    x = np.concatenate([row[f'fold_{i}'] for i in range(len(temp_fold))])
    
    return ' '.join(x)