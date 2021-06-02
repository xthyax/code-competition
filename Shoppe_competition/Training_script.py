# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append("..")
import os
os.environ["MKL_NUM_THREADS"] = "3" # "6"
os.environ["OMP_NUM_THREADS"] = "2" # "4"
os.environ["NUMEXPR_NUM_THREADS"] = "3" # "6"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from custom_dataloader import FastDataLoader

def ShowDataGraph(data_frame):

    colors = ["Red", "Green", "Blue", "Orange", "Gold", "Darkseagreen"]

    len_columns = len(data_frame.columns)

    columns_name = list(data_frame.columns)

    fig = make_subplots(rows=len_columns//2 + 1, cols=2, subplot_titles=tuple(columns_name))
    current_col = 1

    for i in columns_name:
        if data_frame[i].dtype == "object":
            fig.add_trace(go.Bar(x=list(dict(data_frame[i].value_counts(sort=False)).keys()) ,y=list(dict(data_frame[i].value_counts(sort=False)).values()) ), row=columns_name.index(i) //2 + 1 , col=current_col)
        
        else:
            fig.add_trace(go.Histogram(x=list(data_frame[i])), row=columns_name.index(i) //2 + 1 , col=current_col)
        current_col = current_col + 1 if current_col < 2 else 1
            
    fig.update_layout(height=200 * len_columns// 2 , width= 900 ,title="Feature values",template="plotly_white", showlegend=False)
    
    fig.show()

def showFeatureImportant(X_frame, y, target_names=["Dead", "Survived"]):
    x = X_frame.unique()
    y_list = []
    unique_list = []
    
    for x_value in x:
        indexList =  X_frame.index[X_frame == x_value]
        Target = y.loc[indexList.tolist()]
        values, counts = np.unique(Target, return_counts=True)

        if len(values) == len(target_names):
            unique_list.append(values)
            y_list.append(counts)
        
        else:
            counts_temp = np.zeros((2,), dtype='int64')
            counts_temp[values] = counts
            unique_list.append(np.array([i for i in range(len(target_names))], dtype='int64'))
            y_list.append(counts_temp)

    y_show = [[y_list[i][value] for i in range(len(y_list))] for value in range(len(target_names))]

    fig = go.Figure()
    for i in range(len(target_names)):
        fig.add_trace(go.Bar(x=x , y=y_show[i], name=target_names[i]))

    fig.update_layout(barmode='stack')
    fig.show()

def showFeatureDistribute(X_frame, showing_features, y, plot_mode=2):
    
    y_show = y.copy()

    data_show = pd.concat([X_frame, y_show], axis=1)

    data_show[y.name] = ["Yes" if value==1 else "No" for value in data_show[y.name]]

    if plot_mode == 2: 
        assert len(showing_features) == 2
        fig = px.scatter(data_show, x=showing_features[0], y=showing_features[1], color=str(y.name))

        fig.update_traces(marker=dict(size=12,
                                line=dict(width=2,
                                            color='DarkSlateGrey')),
                    selector=dict(mode='markers'))
        fig.update_layout(hovermode="x")
    else:
        assert len(showing_features) == 3
        fig = px.scatter_3d(data_show, x=showing_features[0], y=showing_features[1], z=showing_features[2], color=y.name, symbol=y.name)

    fig.show()


def make_mi_score(X, y):
    from sklearn.feature_selection import mutual_info_classif
    X = X.copy()
    X.dropna(axis=1, inplace=True)
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores

# %% [markdown]
# # Load data

# %%
source_folder = r"shopee-product-matching"
# source_folder = r"D:\Coding_practice\_Data\shopee-product-matching"
if os.path.exists(source_folder):
    GET_CV = True
    
else:
    source_folder = r"../input/shopee-product-matching/"
    GET_CV = False


# %%
NUM_GPUS = 2
if GET_CV:
    from scripts.function_test import set_GPU
    set_GPU(NUM_GPUS)


# %%
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')


# %%
#Preliminaries
from tqdm import tqdm
import math
import random
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Visuals and CV2
import cv2

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# torch
import torch
import timm
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

# %% [markdown]
# # Configuration

# %%
def read_dataset():
    if GET_CV:
        df = pd.read_csv(os.path.join(source_folder, 'train.csv'))
        tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
        df['matches'] = df['label_group'].map(tmp)
        df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(df['posting_id'], df['label_group'])
        fold = 0
        df['fold'] = 0
        for _, test_index in skf.split(df['posting_id'], df['label_group']):
            
            df['fold'].iloc[test_index] = fold
            fold += 1

        df['filepath'] = df['image'].apply(lambda x: os.path.join(source_folder, 'train_images',x))
    else:
        df = pd.read_csv(os.path.join(source_folder, 'test.csv'))

        df['filepath'] = df['image'].apply(lambda x: os.path.join(source_folder, 'test_images',x))

    return df

data = read_dataset()


# %%

encoder = LabelEncoder()
data['label_group'] = encoder.fit_transform(data['label_group'])
print(len(data['label_group'].unique()))


# %%
DIM = (300, 300)

NUM_WORKERS = 2 * NUM_GPUS if NUM_GPUS > 1 else 2
TRAIN_BATCH_SIZE = 16 * NUM_GPUS
VALID_BATCH_SIZE = 32 * NUM_GPUS
EPOCHS = 100
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### MODEL ###
models_path = "model_files"
model_name = 'efficientnet_b3' 

### Metric Loss and its params ###
loss_module = 'arcface'
s = 30.0
m = 0.5
ls_eps = 0.0
easy_margin = False

### Scheduler and its params ###
scheduler_params = {
    "lr_start" : 1e-5,
    "lr_max" : 1e-5 * TRAIN_BATCH_SIZE,
    "lr_min" : 1e-6,
    "lr_ramp_ep" : 5,
    "lr_sus_ep" : 0,
    "lr_decay" : 0.8,    
}

### Model Params ###
model_params = {
    "n_classes": 11014,
    'model_name': model_name,
    'use_fc' :False,
    'fc_dim' : 512,
    'dropout': 0.0,
    'loss_module': loss_module,
    's': s,
    'margin': m,
    'ls_eps': ls_eps,
    'theta_zero': 0.785,
    'pretrained': True
}

# %% [markdown]
# # Utils

# %%
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(SEED)


# %%
def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_true = y_true.apply(lambda x: len(x)).values
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_true + len_y_pred)
    
    return f1


# %%
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
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# %%
def fetch_loss():
    loss = nn.CrossEntropyLoss()
    return loss

# %% [markdown]
# # Augmentations

# %%
def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0], DIM[1], always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6),  p=0.5),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )
def get_valid_transforms():
    
    return albumentations.Compose(
     [
        albumentations.Resize(DIM[0], DIM[1], always_apply=True),
        albumentations.Normalize(),
        ToTensorV2(p=1.0)
     ]   
    )

# %% [markdown]
# # Dataset

# %%
class ShopeeDataset(Dataset):
    def __init__(self, csv,  transforms=None, inference=False):

        self.csv = csv.reset_index()
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        text = row.title
        
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(row.label_group)

# %% [markdown]
# # Model

# %%
class ShopeeNet(nn.Module):
    
    def __init__(
                self,
                n_classes,
                model_name ='efficientnet_b0',
                use_fc = False,
                fc_dim =512,
                dropout = 0.0,
                loss_module = 'softmax',
                s = 30.0,
                margin = 0.50,
                ls_eps = 0.0,
                theta_zero = 0.785,
                pretrained = True
                ):
        super(ShopeeNet, self).__init__()
        print(f'Buidling Model Backbone for {model_name} model.')

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes, s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
            
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init_xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
    
    def forward(self, x, label):
        feature = self.extract_feat(x)

        if self.loss_module in ('arcface'):
            logits = self.final(feature, label)

        else:
            logits = self.final(feature)
        
        return feature, logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x

# %% [markdown]
# # Metric Learning Losses

# %%
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps # Label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)

        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot =  (1 - self.ls_eps) * one_hit + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

# %% [markdown]
# # Custom LR

# %%
class ShopeeScheduler(_LRScheduler):
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
        super(ShopeeScheduler, self).__init__(optimizer,last_epoch)

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

# %% [markdown]
# # Training function

# %%
def train_fn(dataloader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    loss_score = AverageMeter()

    tk0 = tqdm(enumerate(dataloader), total = len(dataloader))
    for bi,d in tk0:
        
        batch_size = d[0].shape[0]
        
        images = d[0]
        targets = d[1]

        if NUM_GPUS == 1:
            images = images.to(device)
            targets = targets.to(device)

        optimizer.zero_grad()

        _, output = model(images, targets)

        # output = model(images, targets)
        targets = targets.to(output.device, non_blocking=True)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])
    
    if scheduler is not None:
        scheduler.step()

    return loss_score

# %% [markdown]
# # Evaluation Function

# %%
def eval_fn(data_loader, model, criterion, device):

    loss_score = AverageMeter()

    model.eval()
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))

    with torch.no_grad():
        for bi,d in tk0:
            batch_size = d[0].size()[0]

            images = d[0]
            targets = d[1]

            if NUM_GPUS == 1:
                images = images.to(device)
                targets = targets.to(device)

            _, output = model(images, targets)
            # output = model(images, targets)
            targets = targets.to(output.device, non_blocking=True)
            loss = criterion(output, targets)

            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)

    return loss_score


# %%
def run(continue_training=False):
    list_models = [0] * len(data['fold'].unique())
    for i in range(len(data['fold'].unique())):
    # for i in [0]:
        model_path = os.path.join(models_path, f'model_{model_name}_IMG_SIZE_{DIM[0]}_{loss_module}_f{i}.pth')
        log_path = f"{model_path}.txt"

        try:
            log_file = open(log_path, "r")
            lineList = log_file.readlines()
            best_val_loss = min( [float(line.split("Valid_loss: ")[1].strip("\n")) for line in lineList] )

        except:
            log_file = open(log_path, "r")
            best_val_loss = 100

        log_file.close()
        logs = []

        train = data[data['fold']!=i].reset_index(drop=True)
        valid = data[data['fold']==i].reset_index(drop=True)

        # Defining Dataset
        train_dataset = ShopeeDataset(
            csv=train,
            transforms=get_train_transforms(),
        )

        valid_dataset = ShopeeDataset(
            csv=valid,
            transforms=get_valid_transforms(),
        )

        train_loader = FastDataLoader(
            # DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            pin_memory=True,
            drop_last=True,
            num_workers=NUM_WORKERS
        )
        valid_loader = FastDataLoader(
            # DataLoader(
            valid_dataset,
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=NUM_WORKERS
        )

        # Defining Model for specific fold
        list_models[i] = ShopeeNet(**model_params)
        # model = ShopeeNet(**model_params)
        
        if continue_training:
            if os.path.isfile(model_path):
                print(f"Loading trained weights {model_path} to model...")
                list_models[i].load_state_dict(torch.load(model_path))

            else:
                pass
            
        if NUM_GPUS > 1:
            list_models[i] = list_models[i].to(device=device)
            list_models[i] = nn.DataParallel(list_models[i])
        
        else:
            list_models[i].to(device=device)

        # Defining criterion
        criterion = fetch_loss()
        criterion.to(device=device)

        optimizer = Adam(list(list_models[i].parameters()), lr=scheduler_params['lr_start'])

        # Defining LR Scheduler
        scheduler = ShopeeScheduler(optimizer, **scheduler_params)

        # THE ENGINE LOOP
        for epoch in range(EPOCHS):
            train_loss = train_fn(train_loader, list_models[i], criterion, optimizer, device, scheduler=scheduler, epoch=epoch)
            
            valid_loss = eval_fn(valid_loader, list_models[i], criterion, device)

            with open(log_path, "a") as file:
                file.write(f"Epoch: {epoch} - LR: {optimizer.param_groups[0]['lr']} - Train_loss: {train_loss.avg} - Valid_loss: {valid_loss.avg}\n")

            if valid_loss.avg < best_val_loss:
                best_val_loss = valid_loss.avg

                if NUM_GPUS == 1:
                    print("Saving model...")
                    torch.save(list_models[i].state_dict(), model_path)

                else:
                    print("Saving multi-gpus model...")
                    torch.save(list_models[i].module.state_dict(), model_path)

        torch.cuda.empty_cache()

# %% [markdown]
# # Inference part

# %%
def get_neighbors(df, embeddings, KNN = 50, image =True):
    from sklearn.neighbors import NearestNeighbors
    
    model = NearestNeighbors(n_neighbors = KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    if GET_CV:
        if image:
            thresholds = list(np.arange(2, 4, 0.1))

        else:
            thresholds = list(np.arange(0.1, 1, 0.1))

        scores = []
        for threshold in thresholds:
            predictions = []

            for k in range(embeddings.shape[0]):
                idx = np.where(distances[k,] < threshold)[0]
                ids = indices[k, idx]
                posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
                predictions.append(posting_ids)

            df['pred_matches'] = predictions
            df['f1'] = f1_score(df['matches'], df['pred_matches'])
            score = df['f1'].mean()
            print(f"F1 score for threshold {threshold} is {score}")
            scores.append(score)
            
        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f"Our best score is {best_score} and has a threshold {best_threshold}")

        # Use threshold
        predictions = []
        for k in range(embeddings.shape[0]):
            if image:
                idx = np.where(distances[k,] < 2.7)[0]
            else:
                idx = np.where(distances[k,] < 0.6)[0]
            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)

    else:
        predictions = []
        for k in tqdm(range(embeddings.shape[0])):
            if image:
                idx = np.where(distances[k,] < 2.7)[0]

            else:
                idx = np.where(distances[k,] < 0.6)[0]

            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids)

    del model, distances, indices
    import gc
    gc.collect()
    return df, predictions


# %%
def get_image_embeddings(csv, IMG_MODEL_PATH):
    embeds = []

    model = ShopeeNet(n_classes=model_params["n_classes"], model_name=model_name)
    model.eval()

    model.load_state_dict(torch.load(IMG_MODEL_PATH),strict=False)
    model = model.to(device)

    image_dataset = ShopeeDataset(csv, transforms=get_valid_transforms())
    image_loader = FastDataLoader(
        image_dataset,
        batch_size=VALID_BATCH_SIZE,
        pin_memory=True,
        drop_last=False,
        num_workers=NUM_WORKERS
    )

    with torch.no_grad():
        for img, label in tqdm(image_loader):
            img = img.to(device)
            label = label.to(device)
            feat, _ = model(img, label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

    del model
    image_embeddings = np.concatenate(embeds)
    print(f"Our image embeddings shape is {image_embeddings.shape}")
    del embeds
    import gc
    gc.collect()
    return image_embeddings


# %%
def combine_predictions(row):
#     x = np.concatenate([row['image_predictions'], row['text_predictions']])
#     return ' '.join( np.unique(x) )
    x = np.concatenate([row[f'image_predictions_f{i}'] for i in range(5)])
    return ' '.join( np.unique(row['image_predictions']) )



# %%
if __name__ == '__main__':

    if GET_CV:
        run(continue_training=True)
    sys.exit()
# %%
    for i in range(5):
        IMG_MODEL_PATH = os.path.join(models_path, f'model_{model_name}_IMG_SIZE_{DIM[0]}_{loss_module}_f{i}.pth')
        image_embeddings = get_image_embeddings(data, IMG_MODEL_PATH)
        data, image_predictions = get_neighbors(data, image_embeddings, KNN = 50, image = True)
        data[f'image_predictions_f{i}'] = image_predictions
        torch.cuda.empty_cache()


# %%
    if GET_CV:
        # data['text_predictions'] = text_predictions
        data['pred_matches'] = data.apply(combine_predictions, axis=1)
        # data['pred_matches'] = image_predictions
        data['f1'] = f1_score(data['matches'], data['pred_matches'])
        score = data['f1'].mean()
        print(f"Our final f1 cv score is {score}")
        data['matches'] = data['pred_matches']
        data[['posting_id', 'matches']].to_csv('submission.csv', index = False)

    else:
        # data['text_predictions'] = text_predictions
        data['matches'] = df.apply(combine_predictions, axis = 1)
        # data['matches'] = image_predictions
        data[['posting_id', 'matches']].to_csv('submission.csv', index = False)


# %%



