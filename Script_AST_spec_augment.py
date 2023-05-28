
import os
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('ComParE2022_VecNet/src'))

import config ,config_pytorch
#from evaluate import get_results
import numpy as np

# Troubleshooting and visualisation
# import IPython.display as ipd

# humbug lib imports
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#from PyTorch import config_pytorch
from datetime import datetime
import math
import pickle

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime
import time

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.metrics import average_precision_score
import sys

from tqdm.notebook import tqdm
# additional pytorch tools
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
from nnAudio import features , Spectrogram
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import argparse
import wget
os.environ['TORCH_HOME'] = '../../pretrained_models'
from timm.models.layers import to_2tuple,trunc_normal_


# Parse input arguments
parser = argparse.ArgumentParser(description='Trainable_SpecAugment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers')

parser.add_argument('--pin_memory',default= True,
                    help='pin_memory')
parser.add_argument('--test_batch_size',type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--DEBUG',default= False,
                    help='whether or not to print error messages')

parser.add_argument('--num_epochs', type=int, default=250,
                    help='number of epochs to train')
parser.add_argument('--USE_SHORT_AUDIO',  default= True,
                    help='number of epochs to train')
#short_audio=USE_SHORT_AUDIO


# TODO Step 1: Add the following to the argument parser:
# number of nodes (num_nodes, type = int, default = 1), 
# ID for the current node (node_id, type = int, default = 0)
# number of GPUs in each node (num_gpus, type = int, default = 1)
args = parser.parse_args()
classes = ['an arabiensis','culex pipiens complex', 'ae aegypti','an funestus ss','an squamosus',
               'an coustani','ma uniformis','ma africanus' ]

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
    np.array(df_train_offset.specie_ind)
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced',classes=np.unique(np.array(df.specie_ind)),y=np.array(np.array(df.specie_ind)))
    print(type(class_weights))
    print(class_weights.shape)
    return class_weights

### Pad_mean #####
# This function pads a short-audio tensor with its mean to ensure that it becomes a 1.92 sec long audio equivalent
def pad_mean(x_temp,rate = config.rate, min_length = config.min_duration ):
    if DEBUG:
        print("inside padding mean...")
    x_mean = torch.mean(x_temp)
    #x_mean.cuda()
    
    if DEBUG:
        print("X_mean = " + str(x_mean))
    left_pad_amt = int((rate*min_length-x_temp.shape[1])//2)
    if DEBUG:
        print("left_pad_amt = " + str(left_pad_amt))
    left_pad = torch.zeros(1,left_pad_amt) #+ (0.1**0.5)*torch.randn(1, left_pad_amt)
    if DEBUG:
        print("left_pad shape = " + str(left_pad.shape))
    left_pad_mean_add = left_pad + x_mean
    if DEBUG:
        print("left_pad_mean shape = " + str(left_pad_mean_add))
        print("sum of left pad mean add = " + str(torch.sum(left_pad_mean_add)))
    
    right_pad_amt = int(rate*min_length-x_temp.shape[1]-left_pad_amt)
    right_pad = torch.zeros(1,right_pad_amt)# + (0.1**0.5)*torch.randn(1, right_pad_amt)
    if DEBUG:
        print("right_pad shape = " + str(right_pad.shape))
    right_pad_mean_add = right_pad + x_mean
    if DEBUG:
        print("right_pad_mean shape = " + str(right_pad_mean_add))
        print("sum of right pad mean add = "  + str(torch.sum(right_pad_mean_add)))
    
    
    
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
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cellsplt.xticks(rotation=90)
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


##AST Code###
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=8, fstride=10, tstride=10, input_fdim=1025, input_tdim=31, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            #print("original_num_patches = ",self.original_num_patches)
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            #print("oringal_hw = ",self.oringal_hw)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            #print("original_embedding_dim = ",self.original_embedding_dim)
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x

    

# ### Model class ####
# Subclass the pretrained model and make it a binary classification
# Subclass the pretrained model and make it a binary classification

# Subclass the pretrained model and make it a binary classification

class MyModel(nn.Module):
    def __init__(self, mean = torch.tensor(2.6860e-05), std = torch.tensor(0.0094)):
        super().__init__()
        # num_classes=0 removes the pretrained head
        self.backbone = ASTModel()
        #####  This section is model specific
        #### It freezes some fo the layers by name
        #### you'll have to inspect the model to see the names
                #### end layer freezing
        #self.out = nn.Linear(self.backbone.num_features, 1)
        self.mean = mean
        self.std = std
        #self.sizer = VT.Resize((image_size,image_size),antialias = True)
        self.spec_layer = features.STFT(n_fft=int(config.NFFT), freq_bins=None, hop_length=int(config.n_hop),
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect',
                           sr=2*config.rate, output_format="Magnitude", trainable=False,verbose = False).to('cuda')
        self.batch_norm = nn.BatchNorm2d(num_features= 1)
        #self.augment_layer = augment_audio(trainable = True, sample_rate = config.rate)
        
    def forward(self, x,train = True):
        # first compute spectrogram
        spec_gram = self.spec_layer(x)
        #print("post spec gram shape = ",spec_gram.shape)
        #print("post norm shape = ",spec_gram.shape)
        spec_gram_nan_check = torch.isnan(spec_gram).any().item()
        assert not (spec_gram_nan_check) ,"Tensor contains NaN values after spec gram creation."
        
        with torch.no_grad():
            if train == True:
                #generate a random number and if condition is met apply aug
                ta_transformations_rndm_choice = VT.RandomChoice([AT.FrequencyMasking(freq_mask_param=100),AT.TimeMasking(time_mask_param=50)], p=[.4, .4])
                ta_transformations_rndm_apply = VT.RandomApply([AT.FrequencyMasking(freq_mask_param=50),AT.TimeMasking(time_mask_param=25)],p = .15)
                spec_gram = ta_transformations_rndm_choice(spec_gram)
                spec_gram = ta_transformations_rndm_apply(spec_gram)
                spec_gram_nan_check = torch.isnan(spec_gram).any().item()
                assert not (spec_gram_nan_check) ,"Tensor contains NaN values after augmentations  "
                
                
        #print("spec_gram shape = ",spec_gram.shape)    
        
        #x = self.sizer(spec_gram.squeeze(dim = 1))
        #print("post sizer shape = ",x.shape)
        #fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        x = (spec_gram - self.mean) / (self.std * 2)
        #print("post norm = ",x.shape)
        
        # then repeat channels
        del spec_gram,spec_gram_nan_check
        if DEBUG:
            print("Final shape that goes to backbone = " + str(x.shape))
                
        x = torch.transpose(x, 1, 2)
        x = self.backbone(x)
        backbone_op_nan_check = torch.isnan(x).any().item()
        assert not (backbone_op_nan_check) ,"Tensor contains NaN values in the backbone OP "
        #print("x shape = " + str(x.shape))
        #print("x = " +str(x))
        #pred = nn.Softmax(x)
        pred = x
        #print(np.argmax(pred.detach().cpu().numpy()))
        #print(pred)
        output = {"prediction": pred }
        #print(output)
        del x , backbone_op_nan_check
        return output


### Test Model####
### Test Model####
def test_model(model, loader, criterion,  classes = classes,device=None , call = "val"):
    softmax = nn.Softmax()
    if DEBUG:
        print("calling for ..." +str(call))
    with torch.no_grad():
        if device is None:
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        sigmoid = nn.Sigmoid()
        test_loss = 0.0
        model.eval()
        if DEBUG:
            print("inside test....")
        all_y = []
        all_y_pred = []
        counter = 1
        if DEBUG:
            print("length of loader = " + str(len(loader)))
        for idx,(x,y) in enumerate(loader):
            if DEBUG:
                print("loader index = " + str(idx))
                            
            x = x.to(device).float() 
            y = y.type(torch.LongTensor).to(device)
            if DEBUG:
                print("y = " + str(y))
            y_pred = model(x,train = False)['prediction']
            #y_pred_smax = softmax(y_pred)
            preds = torch.argmax(y_pred, axis = 1)
            y_pred_cpu = y_pred.cpu().detach()
            if DEBUG:
                print("y_pred_cpu = " + str(y_pred_cpu))
            #preds = torch.argmax(y_pred_cpu, axis = 1)
            if DEBUG:
                print("preds = " +str(preds))
            all_y_pred.append(preds.cpu().detach())
                                   
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            all_y.append(y.cpu().detach())
            #all_y_pred.append(np.argmax(y_pred.cpu().detach().numpy()))
            
            del x
            del y
            del y_pred
        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        if DEBUG:
            print("inside test....")
            print("y = " + str(all_y))
            print("y_pred  = " + str(all_y_pred))
        
        test_loss = test_loss/len(test_loader)
        test_f1 = f1_score(all_y.numpy(), all_y_pred.numpy(),average='weighted')
    
    
    return test_loss, test_f1 , all_y,all_y_pred
## Train_model ####
#train_loader, val_loader, test_loader,model,classes,df_train_offset ,num_epochs = num_epochs 

## Train_model ####
#train_loader, val_loader, test_loader,model,classes,df_train_offset ,num_epochs = num_epochs 
def train_model(train_loader, val_loader,test_loader, model ,classes,class_weights,num_epochs ):
    # Creates a GradScaler once at the beginning of training.
    torch.manual_seed(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')    
    
    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    model = model.to(device)
    weights_adj = torch.tensor(class_weights).type(torch.float).to(device)
    criterion_1 = nn.CrossEntropyLoss(weight=weights_adj,label_smoothing=.1)
    criterion_2 = nn.CrossEntropyLoss(weight=weights_adj)
    lr = .000015
    base_optimiser = timm.optim.AdamP(model.parameters(), lr= lr)
    look_optimiser = timm.optim.Lookahead(base_optimiser)
    cooldown_epoch = 50
    
    #optimiser = timm.optim.AdamW(model.parameters(), lr=config_pytorch.lr)
    #timm.optim.Lookahead(optimiser, alpha=0.5, k=6)
    scheduler = timm.scheduler.CosineLRScheduler(base_optimiser, t_initial= num_epochs,lr_min= lr/100,warmup_t = 5,warmup_lr_init= lr/10,noise_std=.075)
       
    #optimiser = timm.optim.RAdam(model.parameters(), lr=config_pytorch.lr/10)
    num_epochs = num_epochs
    all_train_loss = []
    all_train_f1 = []
    all_val_loss = []
    all_val_f1 = []
    best_val_loss = np.inf
    best_val_f1 = -np.inf
    best_train_f1 = -np.inf
    best_epoch = -1
    checkpoint_name = None
    overrun_counter = 0
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax()
    all_train_f1 = []
    all_val_f1 = []
    accumulation_steps = 4
    lr_log = []
    for e in range(num_epochs + cooldown_epoch):
        start_time = time.time()
        train_loss = 0.0
        model.train()
        running_loss = 0.0
        all_y = []
        all_y_pred = []
        #tk0 = tqdm(train_loader, total=int(len(train_loader)))
        for batch_i, inputs in enumerate(train_loader):
            if DEBUG:
                print("inside train loop.. batch_ind = " +str(batch_i))
            if batch_i % 100 == 0:
                bat_time = time.time()
                durn = (bat_time - start_time)/60
                print("epoch = " +str(e) + "batch = " +str(batch_i) + " of " + str(len(train_loader)) + "duraation = " + str(durn))
            x = inputs[0].to(device).float()
            if DEBUG:
                print("inside train loop.. x device = " +str(x.device))
                
            
            y = inputs[1].type(torch.LongTensor).to(device)
            x_sum = torch.sum(x,axis = 1)
            x_sum.unsqueeze(dim = 1)
                                  
            with autocast():
                y_pred = model(x,train = True)['prediction']
                #y_pred_smax = softmax(y_pred)
                preds = torch.argmax(y_pred, axis = 1)
                if e < 20 :
                    
                    loss = criterion_1(y_pred, y)
                else:
                    
                    loss = criterion_2(y_pred, y)
            
            if DEBUG:
                    print("y_pred  = " +str(y_pred))
                    print("preds = " +str(preds))
                   
            train_loss += loss.item()
            all_y.append(y.cpu().detach())
            y_pred_cpu = y_pred.cpu().detach()
            #preds = torch.argmax(y_pred_cpu, axis = 1)
            if DEBUG:
                print("batch_ind = " +str(batch_i))
                print("y_pred_cpu = " + str(y_pred_cpu))
                
            loss.backward()
            all_y_pred.append(preds.cpu().detach())
            base_optimiser.step()
            base_optimiser.zero_grad()
            #scheduler.step(e)
                
            
            torch.nn.utils.clip_grad_norm_(model.parameters(),error_if_nonfinite=False ,max_norm = 1.0 )
            base_optimiser.step()
            del x
            del y
            del y_pred,preds
        
        #lr_log.append(lr)
        look_optimiser.sync_lookahead()
        all_train_loss.append(train_loss/len(train_loader))
        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        if DEBUG:
            print("y = " + str(all_y))
            print("y_pred  = " + str(all_y_pred))
        
        train_f1 = f1_score(all_y.numpy(), all_y_pred.numpy(),average='weighted')
        all_train_f1.append(train_f1)
        if DEBUG:
            print("train acc = " +str(train_acc))
        all_train_f1.append(train_f1)
        val_loss, val_f1 , _,_ = test_model(model, val_loader, criterion = nn.CrossEntropyLoss(), classes = classes ,device=device, call = "val")
        all_val_f1.append(val_f1)
        all_val_loss.append(val_loss)
        if DEBUG:
            print("val F1 = " + str(val_f1))
        all_val_loss.append(val_loss)
        all_val_f1.append(val_f1)
        
        acc_metric = val_f1
        best_acc_metric = best_val_f1
        if acc_metric > best_acc_metric:  
            overrun_counter = -1
            checkpoint_name = f'model_e{e}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'
            torch.save(model.state_dict(), os.path.join(config.model_dir,  checkpoint_name))
            sys.stdout.flush()
            print('Epoch: %d, Train Loss: %.8f, Train f1: %.8f, Val Loss: %.8f, Val f1: %.8f, overrun_counter %i' % (e, train_loss/len(train_loader), train_f1, val_loss/len(val_loader), val_f1,  overrun_counter))
            print('Saving model to:', os.path.join(config.model_dir,  checkpoint_name)) 
            print("Now printing classification rport... ")
            print("********************************")
            current_lr = base_optimiser.param_groups[0]['lr']
            print("Current LR = " + '{0:.8f}'.format(current_lr))
            from sklearn.metrics import classification_report
            _, _ , all_y_test,all_y_pred_test = test_model(model, test_loader, criterion = nn.CrossEntropyLoss(), classes = classes ,device=device, call = "test")
            # at times output is not getting printed. Could be due to multi threading and hence adding sleep
            time.sleep(2)
            sys.stdout.flush()
            print(classification_report(all_y_test.numpy(), all_y_pred_test.numpy(), target_names= classes))
            print("********************************")
            time.sleep(2)
            plot_confusion_matrix(all_y_pred_test.numpy(), all_y_test.numpy() , classes)
            best_epoch = e
            best_val_f1 = val_f1
            best_val_loss = val_loss
            
        else:
            print("..Overrun....no improvement")
            overrun_counter += 1
            sys.stdout.flush()
            print('Epoch: %d, Train Loss: %.8f, Train f1: %.8f, Val Loss: %.8f, Val f1: %.8f, overrun_counter %i' % (e, train_loss/len(train_loader), train_f1, val_loss/len(val_loader), val_f1,  overrun_counter))
        scheduler.step(e+1)
        if overrun_counter > config_pytorch.max_overrun:
            break
            
    
    return model, lr_log,all_train_f1,all_train_loss,all_val_loss,all_val_f1


#### Dataste class #####

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
        if inp_rate != 2*config.rate:
            #print(" Original sample rate = " +str(inp_rate)+ " resampling ...")
            import torchaudio.transforms as T
            resampler = T.Resample(inp_rate, 2*config.rate, dtype=entire_aud.dtype)
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
            
        if DEBUG:
            print("idx = " + str(idx))
            #print("offset = " + str(offset))
            #print("shape of x post augmentation = " + str(x.shape))
            print("from get_item of train, returning  x of shape = " +str(f_out.shape))
        
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

#### Load model ####
def load_model(filepath, model=MyModel('convnext_xlarge_in22k')):
    # Instantiate model to inspect
    print("Filepath = " + str(filepath))
    print("model = " +str(model))
    device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device("cpu"))
    print(f'Training on {device}')
        
    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device)
    # Load trained parameters from checkpoint (may need to download from S3 first)


    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    checkpoint = model.load_state_dict(torch.load(filepath))

    return model

# ## 

if __name__ == '__main__':
    batch_size = args.batch_size
    pin_memory = args.pin_memory
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    short_audio=args.USE_SHORT_AUDIO
    DEBUG = args.DEBUG
    print("inside main.....")
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
    model =MyModel()
    min_length = (config.win_size * config.n_hop) / config.rate
    
    train_dataset = MozDataset(df_train_offset,  config.data_dir, min_length)
    val_dataset = MozDataset(df_val_offset,  config.data_dir, min_length)
    test_dataset = MozDataset(df_test_offset,  config.data_dir, min_length)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers,batch_size = batch_size,shuffle = True, pin_memory=True )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory, shuffle = True )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,num_workers= num_workers, pin_memory=pin_memory,shuffle = True)
    
    
    
    #train_loader, val_loader,test_loader, model ,class_weights, classes = classes, num_epochs = args.num_epochs ,n_channels = 1
    #train_loader, val_loader,test_loader, model, classes ,df,num_epochs = args.num_epochs ,n_channels = 1
    tr_model, lr_log,all_train_f1,all_train_loss,all_val_loss,all_val_f1 = train_model(train_loader, val_loader, test_loader,model,classes,class_weights ,num_epochs = args.num_epochs )
    
    print("ALL DONE!!!!")

    

    
