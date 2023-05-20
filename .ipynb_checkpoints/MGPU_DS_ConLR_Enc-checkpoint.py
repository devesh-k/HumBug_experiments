
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath('ComParE2022_VecNet/src'))
import config,config_pytorch
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score ,confusion_matrix, classification_report

import math
import pickle

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from datetime import datetime
import time


import matplotlib
import matplotlib.pyplot as plt



from tqdm.notebook import tqdm

import random
import torchaudio
import torchaudio.transforms as AT
import torchvision.transforms as VT
from torch.cuda.amp import autocast, GradScaler
from timm.scheduler.cosine_lr import CosineLRScheduler
import timm
import timm.optim
from timm.models import model_parameters
from glob import glob
## nnAudio
from nnAudio import features
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import argparse
## DDp Import
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")

import argparse
import deepspeed
from torch.utils.tensorboard import SummaryWriter 
import torch.profiler
from contextlib import ExitStack


def add_argument():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Trainable_SpecAugment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    
    parser.add_argument('--with_cuda',
                        default= True,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    
    parser.add_argument('--batch_size', type=int, default= 256,
                        help='input batch size for training')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')

    parser.add_argument('--pin_memory',default= True,
                        help='pin_memory')
    parser.add_argument('--test_batch_size',type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--DEBUG',default= False,
                        help='whether or not to print error messages')
    
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        help="output logging information at a given interval")
    
    parser.add_argument('--profile-execution', type=bool, default=False,
                       help='Use pytorch profiler during execution ')
    
    parser.add_argument('--profile-name', default=False,
                       help=' Profile folder name ') 

    parser.add_argument('--num_epochs', type=int, default=400,
                        help='number of epochs to train')
    parser.add_argument('--USE_SHORT_AUDIO',  default= True,
                        help='number of epochs to train')
    parser.add_argument('--num-nodes', type=int, default=4,
                        help='Number of available nodes/hosts')
    parser.add_argument('--node-id', type=int, default=0,
                        help='Unique ID to identify the current node/host')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs in each node')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


classes = ['an arabiensis','culex pipiens complex', 'ae aegypti','an funestus ss','an squamosus',
               'an coustani','ma uniformis','ma africanus' ]
args = add_argument()


def get_offsets_df(df, short_audio=False):
    audio_offsets = []
    #This is same as defined in config -min_duration = win_size * frame_duration
    min_length = config.win_size*config.NFFT/(((1/config.n_hop)*config.NFFT)*config.rate)
    step_frac = config.step_size/config.win_size
    stride = step_frac*min_length
#     print("min_length = " +str(min_length))
#     print("step_frac = " +str(step_frac))
#     print("stride = " +str(stride))
    for _,row in df.iterrows():
        #processed_data keeps track of the tensor_values processed thus far
        if row['length'] > min_length:
            processed_data = 0
            #total_data is the total tensor present in the audio
            total_data = config.rate*row['length']
            #print("********")
            count = 0
            #print("count = " +str(count))
            #print("id = " + str(row['id']) + " duration = " +str(row['length']) + "total x vals = " + str(total_data))
            inner_loop_flag = False
            #print("going into the inner loop to offset....")
            while(processed_data < total_data):
                #print("inside inner loop.....")
                start = count*stride*config.rate
                #now find out the row_len
                if total_data - (start + min_length*config.rate) >= 0:
                    #print("full chunk ")
                    row_len = min_length
                    end = start + row_len*config.rate
                    audio_offsets.append({'id':row['id'], 'offset':count, 'length': row_len,'specie_ind': row['specie_ind'],'start':start,'end':end})
                    #print("count = " +str(count) + "offset = " +str(count) + "start = " +str(start) + "end = " +str(end))
                    #print("for count.... = " + str(count) + "processed data = " +str(processed_data))
                    count+=1
                    processed_data = (count*stride)*config.rate
                    
                else:
                    inner_loop_flag = True
                    break
                    
                                                       
            #for processing residual data
            if(inner_loop_flag):
                #print("processing residual ....processed " +str(processed_data) + " of " + str(total_data))
                start = count*stride*config.rate
                resid_durn = round((total_data - processed_data)/config.rate,2)
                end = total_data
                #print("for..." + str(row['id']) + " adding the residual data in the data frame with duration = " + str(resid_durn))
                audio_offsets.append({'id':row['id'], 'offset':count, 'length':resid_durn ,'specie_ind': row['specie_ind'],'start':start,'end':end})
            
        elif short_audio:
            start = 0
            end = row['length']*config.rate
            audio_offsets.append({'id':row['id'], 'offset':0,'length': row['length'],'specie_ind': row['specie_ind'],'start':0 , 'end':end})
    return pd.DataFrame(audio_offsets)       


# ####### Prepare df######

def prepare_df(classes ,csv_loc = config.data_df  ):
    """This function reads a csv and creates a dataframe for further processing."""
    df = pd.read_csv(csv_loc)
    #df = df.loc[df['Grade'].notnull()]
    df = df.loc[df['species'].notnull()]
    # a new column for specie_index to hold numerical values for specie
    df['specie_ind'] = "NULL_VAL"
    ind = 0
    for specie in classes:
        print("specie = " + str(specie) + "and its index = " + str(ind) )
        row_indexes=df[df['species']==specie].index 
        df.loc[row_indexes,'specie_ind']= ind
        ind+=1
    #remove all the rows where specie is other than the one present in classes
    df.drop(df[df['specie_ind'] == "NULL_VAL"].index, inplace=True)
    #filter the data for TZ and cup recordings only
    idx_multiclass = np.logical_and(df['country'] == 'Tanzania', df['location_type'] == 'cup')
    df_all = df[idx_multiclass]
    df_all.reset_index(inplace=True, drop = True )
    return df_all

#### plt df
def plot_df(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    sns.countplot(x = 'species', data = df , ax = ax , hue = 'gender',palette='dark')
    #ax.bar_label(ax.containers[0])
    #ax.bar_label(ax.containers[-1], fmt='Count:\n%.2f', label_type='center')
    plt.xticks(rotation=90 )
    plt.title("Distribution of Species ")
    plt.rc('xtick', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('axes', labelsize=15)
    plt.rc('figure', titlesize=15)
    plt.show

### Train _test split####
def train_test_split(df_all):
    np.random.seed(42)
    msk_test = np.random.rand(len(df_all)) < 0.2
    df_test = df_all[msk_test]
    df_train_temp  = df_all[~msk_test]
    msk_train = np.random.rand(len(df_train_temp)) < 0.2
    df_val = df_train_temp[msk_train]
    df_train  = df_train_temp[~msk_train]
    return df_train ,df_val ,df_test

### Validate split ####
def validate_split(df1 , df2):
    df_temp = pd.merge(df1,df2, on = 'id', how = 'inner')
    #print(df_temp)
    common_elem = len(df_temp)
    #print("common_elem = ",common_elem)
    con = (common_elem == 0)
    #print("condition = ",con)
    assert (con), "Split has issues"
    print("split is a success")

### Specie _distribution ###
def get_specie_distri(df , classes , type_df = None):
    """This function takes a dataframe and provides a count of each specie class"""
    for i in range(len(classes)):
        print("DF type = " + str(type_df))
        df_temp = df[df['specie_ind'] == i]
        print("i = " +str(i))
        print(len(df_temp))

## Class weights to address imbalance in classes ###
def get_class_weights(df):
    np.array(df.specie_ind)
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced',classes=np.unique(np.array(df.specie_ind)),y=np.array(np.array(df.specie_ind)))
    print(type(class_weights))
    print(class_weights.shape)
    return class_weights

### Pad_mean #####
# This function pads a short-audio tensor with its mean to ensure that it becomes a 1.92 sec long audio equivalent
def pad_mean(x_temp,rate = config.rate, min_length = config.min_duration ):
    
    x_mean = torch.mean(x_temp)
    #x_mean.cuda()
    
    left_pad_amt = int((rate*min_length-x_temp.shape[1])//2)
    left_pad = torch.zeros(1,left_pad_amt) #+ (0.1**0.5)*torch.randn(1, left_pad_amt)
    left_pad_mean_add = left_pad + x_mean
    right_pad_amt = int(rate*min_length-x_temp.shape[1]-left_pad_amt)
    right_pad = torch.zeros(1,right_pad_amt)# + (0.1**0.5)*torch.randn(1, right_pad_amt)
    right_pad_mean_add = right_pad + x_mean
      
    f = torch.cat([left_pad,x_temp,right_pad],dim=1)[0]
    f = f.unsqueeze(dim = 0)
    #print("returning a tensor of shape = " + str(f.shape))
    return(f)


### Plot confusion Matrix ######
def plot_confusion_matrix(y_hat,y_true,classes):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_hat, y_true ,labels= range(len(classes)))
    import seaborn as sns
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g',cmap = 'Blues'); #annot=True to annotate cellsplt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(classes, fontsize = 10)
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=90)
    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(classes, fontsize = 10)
    plt.yticks(rotation=0)
    plt.show()

class Normalize_batch(nn.Module):
    def __init__(self):
        super(Normalize_batch, self).__init__()
        
    def forward(self, x):
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_std = torch.std(x, dim=0, keepdim=True)
        epsilon = 1e-8
        batch_std = torch.sqrt(batch_std ** 2 + epsilon)
        batch_normalized = (x - batch_mean) / batch_std
        return batch_normalized

#####

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}



class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


    

# ### Model class ####
# Subclass the pretrained model and make it a binary classification
# Subclass the pretrained model and make it a binary classification

# Subclass the pretrained model and make it a binary classification

class MyModel(nn.Module):
    def __init__(self, model_name, image_size = 224):
        super().__init__()
        # num_classes=0 removes the pretrained head
        #self.backbone = timm.create_model(model_name, pretrained=True, num_classes=8, in_chans=1, drop_path_rate=0.2, global_pool='max', drop_rate=0.25)
        #####  This section is model specific
        #### It freezes some fo the layers by name
        #### you'll have to inspect the model to see the names
                #### end layer freezing
        #self.out = nn.Linear(self.backbone.num_features, 1)
        self.sizer = VT.Resize((image_size,image_size),antialias = True)
        self.encoder = SupConResNet()
        self.batch_norm = nn.BatchNorm2d(num_features= 1)
        self.spec_layer = features.STFT(n_fft=int(config.NFFT), freq_bins=None, hop_length=int(config.n_hop),
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect',
                           sr=config.rate, output_format="Magnitude", trainable=False,verbose = False).to('cuda')
        #self.augment_layer = augment_audio(trainable = True, sample_rate = config.rate)
        
    def forward(self, x,train = True):
        # first compute spectrogram
        spec_gram = self.spec_layer(x)
        output = {}
        #print("post spec gram shape = ",spec_gram.shape)
        spec_gram = self.batch_norm(spec_gram.unsqueeze(dim = 1).half())
        #print("post norm shape = ",spec_gram.shape)
        spec_gram_nan_check = torch.isnan(spec_gram).any().item()
        assert not (spec_gram_nan_check) ,"Tensor contains NaN values after spec gram creation."
        
        with torch.no_grad():
            if train == True:
                #generate a random number and if condition is met apply aug
                ta_transformations_rndm_choice = VT.RandomChoice([AT.FrequencyMasking(freq_mask_param=100),AT.TimeMasking(time_mask_param=50)], p=[.4, .4])
                ta_transformations_rndm_apply = VT.RandomApply([AT.FrequencyMasking(freq_mask_param=50),AT.TimeMasking(time_mask_param=25)],p = .2)
                spec_gram = ta_transformations_rndm_choice(spec_gram)
                spec_gram = ta_transformations_rndm_apply(spec_gram)
                spec_gram_nan_check = torch.isnan(spec_gram).any().item()
                assert not (spec_gram_nan_check) ,"Tensor contains NaN values after augmentations  "
                aug_bat = [ta_transformations_rndm_choice(spec_gram),ta_transformations_rndm_choice(spec_gram)]
                aug_bat = torch.cat(aug_bat , dim = 0)
                #print("shape of augmented batch = ",aug_bat.shape)
                #output['feat'] = aug_bat
                
        
        encoder = self.encoder.to('cuda')
        features = encoder(aug_bat)
        #print("output of encoder shape = ",features.shape)
        bsz = x.shape[0]
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        #loss = criterion(features, y)
        output['feat'] = features
        #x = self.sizer(spec_gram.squeeze(dim = 1))
        #print("post sizer shape = ",x.shape)
        #x = x.unsqueeze(dim = 1)
        #print("post unsqueeze shape = ",x.shape)
        
        # then repeat channels
        del spec_gram,aug_bat
        #backbone_op_nan_check = torch.isnan(x).any().item()
        #assert not (backbone_op_nan_check) ,"Tensor contains NaN values in the backbone OP "
        #print("x shape = " + str(x.shape))
        #print("x = " +str(x))
        #pred = nn.Softmax(x)
        #pred = x
        #print(np.argmax(pred.detach().cpu().numpy()))
        #print(pred)
        #output["prediction"]=  pred 
        #print(output)
        return output

##
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

### Test Model####
#model, test_loader, criterion = nn.CrossEntropyLoss(), classes = classes ,device=device, call = "test"
def test_model(model, loader, device,fp16,criterion, call = "val"):
    #print("device = " + str(device) + " call = " + str(call))
    with torch.no_grad():
                
        test_loss = 0.0
        model.eval()
        all_y = []
        all_y_pred = []
        for idx,(x,y) in enumerate(loader):
            x = x.to(device).float() 
            y = y.type(torch.LongTensor).to(device)
            if fp16 :
                x = x.half()
                #y = y.half()
            
            with autocast():
                y_pred = model(x,train = False)['prediction']
                loss = criterion(y_pred, y)
            preds = torch.argmax(y_pred, axis = 1)
            y_pred_cpu = y_pred.cpu().detach()
            all_y_pred.append(preds.cpu().detach())
            test_loss += loss.item()
            all_y.append(y.cpu().detach())
            #all_y_pred.append(np.argmax(y_pred.cpu().detach().numpy()))
            
            del x
            del y
            del y_pred
        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
              
        test_loss = test_loss/len(loader)
        test_f1 = f1_score(all_y.numpy(), all_y_pred.numpy(),average='weighted')
        #print("**********")
        #print("inside" +str(call) +"->loss =  " +str(test_loss) + "test_f1 = " + str(test_f1))
        #print("**********")
        #test_f1 = reduce_op(test_f1,device)
        #dist.all_reduce(torch.tensor(test_f1), op=dist.ReduceOp.AVG)
        #print("returning from test... test_f1 = ", test_f1)
 
    return test_loss, test_f1 , all_y,all_y_pred

## Train_model ####
#train_loader, model,classes,class_weights,train_sampler,device
def train_model(train_loader, model,classes,class_weights,device,e ,base_optimiser ,fp16 ):
    # Creates a GradScaler once at the beginning of training.
    torch.manual_seed(0)
    device = device
    #print(f'Training on {device}')    
    weights_adj = torch.tensor(class_weights).type(torch.float).to(device)
    start_time = time.time()
    train_loss = 0.0
    model.train()
    all_y = []
    all_y_pred = []
    criterion = SupConLoss(temperature= .07).to(device)
    for batch_i, inputs in enumerate(train_loader):
        if batch_i % 200 == 0 and torch.distributed.get_rank() == 0:
            bat_time = time.time()
            durn = (bat_time - start_time)/60
            print("epoch = " +str(e) + "batch = " +str(batch_i) + " of " + str(len(train_loader)) + "duraation = " + str(durn))
        x = inputs[0].to(device).float()
        if fp16 :
            x = x.half()
        y = inputs[1].type(torch.LongTensor).to(device)
        with autocast():
            output = model(x,train = True)
            features = output['feat']
            loss = criterion(features, y)    
    
        train_loss += loss.item()
        model.backward(loss)
        model.step()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),error_if_nonfinite=False ,max_norm = 1.0 )
        
        del x
        del y
                        
    train_loss = train_loss/len(train_loader)
    return train_loss

#### Dataste class #####
class MozDataset(Dataset):

    def __init__(self, audio_df, data_dir, min_length, cache=None, transform=None):
        """
        Args:
            audio_df (DataFrame): from get_offsets_df function 
            noise_df (DataFrame): the df of noise files and lengths
            data_dir (string): Directory with all the wavs.
            cache (dict): Empty dictionary used as cache
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audio_df = audio_df
        #self.noise_df = noise_df
        self.data_dir = data_dir
        self.min_length = min_length
        self.transform = transform
        self.cache = cache

    def __len__(self):
        return len(self.audio_df)
    
    def __getitem__(self, idx):
        #real_idx = idx % len(self.audio_df)
        temp_id = int(self.audio_df.loc[idx]['id'])
        file_path = os.path.join("ComParE2022_VecNet","data","audio")
        path_var = file_path +"/" +str(temp_id)+ str(".wav")
        entire_aud, inp_rate = torchaudio.load(path_var)
        if inp_rate != config.rate:
            #print(" Original sample rate = " +str(inp_rate)+ " resampling ...")
            import torchaudio.transforms as T
            resampler = T.Resample(inp_rate, config.rate, dtype=entire_aud.dtype)
            entire_aud = resampler(entire_aud)
            #print("processsing file on " +str(path_var) + "Post resample shape =  " + str(entire_aud.shape))
        
        aud_len = self.audio_df.loc[idx]['length']
        offset = int(self.audio_df.loc[idx]['offset'])
        #print("sliced val = " +str(int((offset+config.min_duration)*config.rate)))
        start_pos = int(round(self.audio_df.loc[idx]['start']))
        #print("start_pos = " +str(start_pos))
        end_pos =  int(round(self.audio_df.loc[idx]['end']))
        #print("end_pos = " +str(end_pos))
        x = entire_aud[:,start_pos:end_pos]
        #print("extracted x = " +str(x))
        #print("x shape = " +str(x.shape))
        if aud_len < config.min_duration:
            #r = math.ceil((config.rate*self.min_length)/waveform.shape[1])
            #print("padding on " +str(path_var))
            f_out = pad_mean(x)
            #print("returning from padding  SHape = " +str(f_out.shape))
        else:
            f_out = x[0]
            f_out = f_out.unsqueeze(0)
            
                
        #x_val = x[:,start:end]
        #now that we have final x- let's create specgram and add augmentations.
                 
        return (f_out,self.audio_df.loc[idx]['specie_ind'] )


### Get_indices ####
def get_indices(num_values ,df ,classes = classes):
    new_df = pd.DataFrame()
    for ind in range(len(classes)):
        #print("ind = ", ind)
        op = df[df['specie_ind'] == ind]
        #print("len op = ", len(op))
        op_new = op.sample(n = 1)
        #print("rand_ind = " , rand_ind)
        #([df1, df2], axis=1)
        new_df = pd.concat([op_new,new_df],axis = 0)
        #print("elem = " , elem)
        #new_list.append(elem)
    if len(new_df) < num_values:
        diff =  num_values - len(new_df)
        #print("diff = ", diff)
        remaining_elems= df.sample(n = diff)
        #print("len of remaining elems = ", len(remaining_elems))
        new_df = pd.concat([remaining_elems,new_df],axis = 0)
        
    #print("new_df = ", new_df)    
    new_df_1 = new_df.reset_index(drop = True)
    return new_df_1


def reduce_op(val,device):
    val_t = torch.tensor(val).to(device)
    val_t = val_t.to(dtype=torch.float32)  # Make sure the tensor is of float32 data type
    val_t = val_t.contiguous()  # Make sure the tensor is stored in contiguous memory
    dist.all_reduce(val_t, op=dist.ReduceOp.AVG)
    return val_t

def print_result(all_y_test,all_y_pred_test, classes = classes):
    print("Now printing classification rport... ")
    print("********************************")
    from sklearn.metrics import classification_report
    # at times output is not getting printed. Could be due to multi threading and hence adding sleep
    print(classification_report(all_y_test.numpy(), all_y_pred_test.numpy(), target_names= classes))
    print("********************************")
    plot_confusion_matrix(all_y_pred_test.numpy(), all_y_test.numpy() , classes)
    plt.show()

    

    

# if __name__ == '__main__':

warnings.filterwarnings("ignore")
## Code that was previously in main
batch_size = args.batch_size
pin_memory = args.pin_memory
num_workers = args.num_workers
num_epochs = args.num_epochs
short_audio=args.USE_SHORT_AUDIO
DEBUG = False

classes = ['an arabiensis','culex pipiens complex', 'ae aegypti','an funestus ss','an squamosus',
           'an coustani','ma uniformis','ma africanus' ]
csv_loc = os.path.join("ComParE2022_VecNet","data","metadata","neurips_2021_zenodo_0_0_1.csv")
df = prepare_df(classes = classes,csv_loc = csv_loc)
plot_df(df)
df_train ,df_val ,df_test = train_test_split(df)
print("now validating the split post loading and keeping TZ data")
validate_split(df_train ,df_val)
validate_split(df_train ,df_test)
validate_split(df_test ,df_val)


df_train_offset = get_offsets_df(df_train, short_audio=True)
df_test_offset = get_offsets_df(df_test, short_audio=True)
df_val_offset = get_offsets_df(df_val, short_audio=True)
df_train_offset.reset_index(inplace = True , drop = True)
df_test_offset.reset_index(inplace = True , drop = True)
df_val_offset.reset_index(inplace = True , drop = True)
print("now validating the split post offset_creation")
validate_split(df_train_offset ,df_val_offset)
validate_split(df_train_offset ,df_test_offset)
validate_split(df_test_offset ,df_val_offset)

class_weights = get_class_weights(df_train_offset)
#print("inside main. class_weigths type = ", type(class_weights))
model_b = MyModel('convnext_xlarge_in22k',224)
#model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
min_length = (config.win_size * config.n_hop) / config.rate

train_dataset = MozDataset(df_train_offset,  config.data_dir, min_length)
val_dataset = MozDataset(df_val_offset,  config.data_dir, min_length)
test_dataset = MozDataset(df_test_offset,  config.data_dir, min_length)


# train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=WORLD_SIZE,rank=global_rank)
# test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,num_replicas=WORLD_SIZE,rank=global_rank)
# val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,num_replicas=WORLD_SIZE,rank=global_rank)


train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers,batch_size = batch_size, pin_memory=True,shuffle = True )
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,num_workers= num_workers, pin_memory=pin_memory, shuffle = True)


#model.to(device)
#model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
#model = DDP(model, device_ids=[torch.cuda.device])
model_engine, optimizer, trainloader, __ = deepspeed.initialize(args=args, model= model_b, model_parameters= model_b.parameters(), training_data=train_dataset )
fp16 = True
device = model_engine.local_rank




lr = .0015
base_optimiser = optimizer
#look_optimiser = timm.optim.Lookahead(base_optimiser)
#scheduler = timm.scheduler.CosineLRScheduler(base_optimiser, t_initial= args.num_epochs,lr_min= lr/100,warmup_t = 5,warmup_lr_init= lr/10,noise_std=.075)

num_epochs = args.num_epochs
cooldown_epoch = 50
all_train_loss = []
all_train_f1 = []
best_loss = np.inf
best_val_f1 = -np.inf
best_train_f1 = -np.inf
best_epoch = -1
checkpoint_name = None
overrun_counter = 0
lr_log = []

#(train_loader, val_loader,test_loader, model ,classes,class_weights,num_epochs,train_sampler,device )
for e in range(num_epochs + cooldown_epoch):
    #train_sampler.set_epoch(e)
    train_loss = train_model(trainloader, model_engine,classes,class_weights,device,e ,base_optimiser, fp16)
    dist.barrier()
    #averaging and reducing
    train_loss = reduce_op(train_loss , device)
    if train_loss < best_loss: 
        #dist.barrier()
        overrun_counter = -1
        checkpoint_name = f'model_e{e}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
        #model_engine.save_checkpoint(os.path.join(config.model_dir), checkpoint_name)
        #torch.save(model.module.state_dict(), os.path.join(config.model_dir,  checkpoint_name))
        best_epoch = e
        best_loss = train_loss
        model_engine.save_checkpoint(os.path.join(config.enc_model_dir), checkpoint_name)
        if torch.distributed.get_rank() == 0:
            print("Better loss score .." )
            print("Device = " + str(device) + 'Epoch: %d, Train Loss: %.8f, overrun_counter %i' % (e, train_loss,  overrun_counter))
            print('Saving model to:', os.path.join(config.model_dir,  checkpoint_name))
            


    else:
        overrun_counter += 1
        if torch.distributed.get_rank() == 0:
            print("on device..." +str(device) + "..Overrun....no improvement")
            print("Device = " + str(device) + 'Epoch: %d, Train Loss: %.8f, overrun_counter %i' % (e, train_loss,  overrun_counter))
            print("current_time =",str(datetime.now()))

    
    if overrun_counter > config_pytorch.max_overrun:
        print("MAX OVERRUNS.. BREAKING")
        break

        ### print classification report#######
    #scheduler.step(e + 1)
print("Finished all epochs ALL DONE!!!!")

# #
