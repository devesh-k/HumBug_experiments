
import os
import pandas as pd
import sys
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
import torchaudio.transforms as T
import torchvision.transforms as VT
from torch.cuda.amp import autocast, GradScaler
from timm.scheduler.cosine_lr import CosineLRScheduler
import timm
import timm.optim
from timm.loss import BinaryCrossEntropy
from timm.utils import NativeScaler
from timm.models import model_parameters
from glob import glob
## nnAudio
from nnAudio import features , Spectrogram
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import argparse


# Parse input arguments
parser = argparse.ArgumentParser(description='Trainable_SpecAugment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers')

parser.add_argument('--pin_memory',default= True,
                    help='pin_memory')
parser.add_argument('--test_batch_size',type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--DEBUG',default= False,
                    help='whether or not to print error messages')

parser.add_argument('--num_epochs', type=int, default=300,
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


class ApplyAug(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug_flag_y = VT.Compose([
            VT.GaussianBlur(3),
            VT.RandomErasing(),
            #transforms.Normalize(mean=2.7360104e-05, std=.0061507192)
        ])
        
    def forward(self, spec_gram, aug_flag):
        if aug_flag == "Y":
            rgb_img_auto_aug = self.aug_flag_y(spec_gram)
            return rgb_img_auto_aug
        else:
            return spec_gram    


class SpecAug_All(nn.Module):
    def __init__(self):
        super(SpecAug_All, self).__init__()
        self.transformations = nn.Sequential(
            VT.transforms.RandomApply([
                torchaudio.transforms.FrequencyMasking(freq_mask_param=100),
                torchaudio.transforms.TimeMasking(time_mask_param=50)
                
            ], p=0.05),  # Example probability of applying the transformations
            # Additional transformations or layers can be added here
        )

    def forward(self, x):
        x = self.transformations(x)
        return x

class FreqMask(nn.Module):
    def __init__(self):
        super(FreqMask, self).__init__()
        self.transformations = nn.Sequential(
            VT.transforms.RandomApply([
                torchaudio.transforms.FrequencyMasking(freq_mask_param=80),
                ], p=0.50),  # Example probability of applying the transformations
            # Additional transformations or layers can be added here
        )
        

    def forward(self, x):
        x = self.transformations(x)
        return x

class TimeMask(nn.Module):
    def __init__(self):
        super(TimeMask, self).__init__()
        self.transformations = nn.Sequential(
            VT.transforms.RandomApply([
                torchaudio.transforms.TimeMasking(time_mask_param=25)
                ], p=0.40),  # Example probability of applying the transformations
            # Additional transformations or layers can be added here
        )
        

    def forward(self, x):
        x = self.transformations(x)
        return x



    

# ### Model class ####
# Subclass the pretrained model and make it a binary classification
# Subclass the pretrained model and make it a binary classification

class MyModel(nn.Module):
    def __init__(self, model_name ,input_size = 4, hidden_size = 768 , num_classes = 8 , image_size = 224 , batch_size = args.batch_size):
        super(MyModel, self).__init__()

        self.backbone = timm.create_model(model_name,
                        pretrained=True, num_classes=8, in_chans=1, 
                        drop_path_rate=0.2, global_pool='max',
                        drop_rate=0.25)
        self.spec_layer = features.STFT(n_fft=int(config.NFFT), freq_bins=None, hop_length=int(config.n_hop),
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect',
                           sr=config.rate, output_format="Magnitude", trainable=True,verbose = False,fmin = 100,fmax = 2000)
        #self.linear = nn.Linear(hidden_size , 1024)
        self.output = nn.Linear(1000, num_classes)
        self.sizer = VT.Resize((image_size,image_size),antialias = True)
        self.softmax = nn.Softmax(dim = 1)
        self.TimeMask_layer = TimeMask()
        self.FreqMask_layer = FreqMask()
        self.SpecAug_All_layer = SpecAug_All()
        self.norm_layer = Normalize_batch()
        self.aug_layer = ApplyAug()
        
        

    def forward(self, input_ids,train = True ,attention_mask = False):
        #this will hold the output
        output_dict = {'probs':None , 'preds':None}
        max_input_wav,_ = torch.max(input_ids,dim =2)
        min_input_wav,_ = torch.min(input_ids,dim =2)
        
        if DEBUG:
            print("shape of input = ",input_ids.shape)
            print("max_input_wav = ", max_input_wav)
            print("min_input_wav = ", min_input_wav)
            
        with torch.no_grad():
            spec_gram = self.spec_layer(input_ids)
            spec_gram_nan_check = torch.isnan(spec_gram).any().item()
            min_val_spec_gram = torch.min(spec_gram)
            max_val_spec_gram = torch.max(spec_gram)
            assert not(max_val_spec_gram < 0) , "neg value in the output of spectrogram" 
            assert not (spec_gram_nan_check) ,"Tensor contains NaN values after spec gram creation."
            if DEBUG:
                print("post STFT , spec gram shape = ", spec_gram.shape)
                print("post STFT , spec gram max val = ", torch.max(spec_gram))
                print("post STFT , spec gram min val = ", torch.min(spec_gram))
                print("post STFT , spec_gram_nan_check = ", spec_gram_nan_check)
                 
        
            if train== True:
                rand_aug_choice = torch.randint(low=0, high=100, size=(1,1),device = 'cuda',dtype=torch.int32)
                #print("rand_aug_choice = ",rand_aug_choice)
                if rand_aug_choice %2 == 0 :
                    spec_gram = self.aug_layer(spec_gram , aug_flag = "Y")
                    spec_gram_nan_check = torch.isnan(spec_gram).any().item()
                    assert not (spec_gram_nan_check) ,"Tensor contains NaN values after aug layer Y "
                    if DEBUG:
                        print("post aug_layer Y , spec gram shape = ", spec_gram.shape)
                        print("post aug_layer flag Y , spec gram max val = ", torch.max(spec_gram))
                        print("post aug_layer flag Y, spec gram min val = ", torch.min(spec_gram))
                        print("post aug_layer flag Y , spec_gram_nan_check = ", spec_gram_nan_check)
                        print("post aug_layer flag Y , spec_gram_nan_check = ", spec_gram_nan_check)
            
                spec_gram = self.FreqMask_layer(spec_gram)
                spec_gram = self.TimeMask_layer(spec_gram)
                spec_gram = self.SpecAug_All_layer(spec_gram)
        
        
            spec_gram = self.aug_layer(spec_gram , aug_flag = "N")
            if DEBUG:
                print("post aug_layer N , spec gram shape = ", spec_gram.shape)
            
            spec_gram_nan_check = torch.isnan(spec_gram).any().item()
            assert not (spec_gram_nan_check) ,"Tensor contains NaN values after aug layer Y "
            del spec_gram_nan_check,min_val_spec_gram,max_val_spec_gram,max_input_wav,min_input_wav
            if DEBUG:
                #print("post aug_layer N , spec gram = ", spec_gram)
                print("post aug_layer flag N , spec gram max val = ", torch.max(spec_gram))
                print("post aug_layer flag N, spec gram min val = ", torch.min(spec_gram))
        # now reshape to image_size
                
            if DEBUG:
                print("post reshape , spec gram shape = ", spec_gram.shape)
            spec_gram = torch.nn.functional.interpolate(spec_gram.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
            normalized_spec_gram = self.norm_layer(spec_gram)
        
                
            if DEBUG:
                print("normalized_spec_gram = ",normalized_spec_gram)
                print("post sizer shape of spec_gram = " , spec_gram.shape)
                print("post Norm  , normalized_spec_gram max = ", torch.max(normalized_spec_gram))
                print("post Norm  , normalized_spec_gram min = ", torch.min(normalized_spec_gram))
            #now make it 3 channel 
        #spec_gram = torch.cat((spec_gram, spec_gram, spec_gram), dim=1).to('cuda')
        #del mean,std
        backbone_op = self.backbone(normalized_spec_gram)
        #print("backbone_op = ",backbone_op)
        backbone_op_nan_check = torch.isnan(backbone_op).any().item()
        assert not (backbone_op_nan_check) ,"Tensor contains NaN values in the backbone OP "
        if DEBUG:
            print("backbone_op shape ",backbone_op.shape)
            print("backbone_op = ", backbone_op)
        out_smax = self.softmax(backbone_op)
        if DEBUG:
            print("out_smax = ",out_smax)
            print("out_smax shape = " , out_smax.shape)
        #out_smax = self.softmax(output)
        out = torch.argmax(out_smax , dim = 1)
        #print("out = ",out)
        
        output_dict['probs'] = out_smax
        output_dict['preds'] = out
        #print("^^^^^ inside forward^^^^^^^")
        del spec_gram
        if DEBUG:
            print("output_dict = ", output_dict)
        return output_dict










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
            print(" $$$$$$$$inside test$$$$$$$$....")
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
            output = model(x,train = False)
            y_pred = output['probs']
            #y_pred_smax = softmax(y_pred)
            preds = output['preds']
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
            print(" $$$$$$$$ exiting test$$$$$$$$....")
        
        test_loss = test_loss
        test_f1 = f1_score(all_y.numpy(), all_y_pred.numpy(),average='weighted')
    
    
    return test_loss, test_f1 , all_y,all_y_pred

## Train_model ####
#(train_loader, val_loader, test_loader,model, classes ,class_weights ,num_epochs = num_epochs )
#(train_loader, val_loader, test_loader,model, classes ,class_weights ,num_epochs = num_epochs )
def train_model(train_loader, val_loader,test_loader, model, classes ,df,num_epochs = args.num_epochs ,n_channels = 1):
    # Creates a GradScaler once at the beginning of training.
    
    global_step = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')    
    
    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    model = model.to(device)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    #class_weights_flat = torch.tensor(class_weights).view(-1)  # Reshape to a 1-dimensional tensor
    #weights_adj = class_weights_flat.type(torch.float)  # Convert to float tensor
    #print("inside train. Type of class_weights_flat = ",type(class_weights))
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced',classes=np.unique(np.array(df.specie_ind)),y=np.array(np.array(df.specie_ind)))
    
    weights_adj = torch.from_numpy(class_weights).type(torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_adj)
    #optimiser = timm.optim.create_optimizer_v2(model.parameters(), lr= .0001,opt = 'lookahead_adam')
    base_optimiser = timm.optim.AdamP(model.parameters(), lr= 1e-4)
    lookahead_optimiser = timm.optim.Lookahead(base_optimiser)
    #scheduler = CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min= 1e-6)
    scheduler = timm.scheduler.CosineLRScheduler(base_optimiser, t_initial=num_epochs/2,lr_min= 1e-6,cycle_limit=num_epochs//2 + 1,cycle_decay = .85)
    #timm.optim.Lookahead(optimiser, alpha=0.5, k=6)
    
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
    lr_log = []
    for e in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        model.train()
        loss_nan_flag = False
        #running_loss = 0.0
        all_y = []
        all_y_pred = []
        for batch_i, inputs in enumerate(train_loader):
            if DEBUG:
                print("inside train loop.. batch_ind = " +str(batch_i))
            if batch_i % 500 == 0:
                bat_time = time.time()
                durn = (bat_time - start_time)/60
                print("epoch = " +str(e) + "batch = " +str(batch_i) + " of " + str(len(train_loader)) + "duraation = " + str(durn))
            x = inputs[0].to(device).float()
            has_nan = torch.isnan(x).any().item()
            assert not(has_nan) ,"Tensor contains NaN values in TRAIN."
            if DEBUG:
                print("inside train loop.. x device = " +str(x.device))
            y = inputs[1].type(torch.LongTensor).to(device)
             
            with autocast():
                output = model(x,train = True)
                y_pred = output['probs']
                preds = output['preds']
                loss = criterion(y_pred, y)
            
            if DEBUG:
                    print("y_pred  = " +str(y_pred))
                    print("preds = " +str(preds))
                
            train_loss += loss.item()
            if DEBUG:
                print("loss = ",loss.item())
            is_loss_nan = torch.isnan(loss)
            current_lr = base_optimiser.param_groups[0]['lr']
            lr_log.append(current_lr)
            loss_nan_flag = True
            if is_loss_nan:
                print("NAN encountered returning ...")
                return model, lr_log,all_train_f1,all_train_loss,all_val_loss,all_val_f1
                
            
            all_y.append(y.cpu().detach())
            y_pred_cpu = y_pred.cpu().detach()
            if DEBUG:
                print("batch_ind = " +str(batch_i))
                print("y_pred_cpu = " + str(y_pred_cpu))
                
            all_y_pred.append(preds.cpu().detach())
            base_optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),error_if_nonfinite=False ,max_norm = 5.0 )
            base_optimiser.step()
            del x
            del y
            del y_pred,preds
        
        #lr_log.append(lr)
        lookahead_optimiser.sync_lookahead()
        all_train_loss.append(train_loss/len(train_loader))
        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        if DEBUG:
            print("y = " + str(all_y))
            print("y_pred  = " + str(all_y_pred))
        
        train_f1 = f1_score(all_y.numpy(), all_y_pred.numpy(),average='weighted')
        all_train_f1.append(train_f1)
        if DEBUG:
            print("train_f1 = " +str(train_f1))
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
        scheduler.step(e+1)
        
        if acc_metric > best_acc_metric:  
            overrun_counter = -1
            checkpoint_name = f'model_e{e}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'
            torch.save(model.state_dict(), os.path.join("models",  checkpoint_name))
            sys.stdout.flush()
            print('Epoch: %d, Train Loss: %.8f, Train f1: %.8f, Val Loss: %.8f, Val f1: %.8f, overrun_counter %i' % (e, train_loss/len(train_loader), train_f1, val_loss/len(val_loader), val_f1,  overrun_counter))
            
            print('Saving model to:', os.path.join("models",  checkpoint_name)) 
            print("Now printing classification rport... ")
            print("********************************")
            from sklearn.metrics import classification_report
            test_loss, test_f1 , all_y_test,all_y_pred_test = test_model(model, test_loader, criterion = nn.CrossEntropyLoss(), classes = classes ,device=device, call = "test")
            # at times output is not getting printed. Could be due to multi threading and hence adding sleep
            time.sleep(2)
            sys.stdout.flush()
            print("test_loss = ",str(test_loss/len(test_loader)))
            print("test_f1 = ",str(test_f1))
            print("LR log = ",lr_log[-1])
            
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
        
        if overrun_counter > config_pytorch.max_overrun:
            break
            
    
    return model, lr_log,all_train_f1,all_train_loss,all_val_loss,all_val_f1


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
    print("inside main. class_weigths type = ", type(class_weights))
    model =MyModel('convnext_xlarge_in22k',224)
    min_length = (config.win_size * config.n_hop) / config.rate
    
    train_dataset = MozDataset(df_train_offset,  config.data_dir, min_length)
    val_dataset = MozDataset(df_val_offset,  config.data_dir, min_length)
    test_dataset = MozDataset(df_test_offset,  config.data_dir, min_length)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers,batch_size = batch_size,shuffle = True, pin_memory=True,drop_last = True )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory,drop_last = True )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,num_workers= num_workers, pin_memory=pin_memory,drop_last = True)
    
    
    
    #train_loader, val_loader,test_loader, model ,class_weights, classes = classes, num_epochs = args.num_epochs ,n_channels = 1
    #train_loader, val_loader,test_loader, model, classes ,df,num_epochs = args.num_epochs ,n_channels = 1
    tr_model, lr_log,all_train_f1,all_train_loss,all_val_loss,all_val_f1 = train_model(train_loader, val_loader, test_loader,model,classes,df_train_offset ,num_epochs = num_epochs )
    
    print("ALL DONE!!!!")

    

    
