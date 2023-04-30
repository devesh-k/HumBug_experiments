
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
#import datetime

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
from timm.loss import BinaryCrossEntropy
from timm.utils import NativeScaler
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



# Parse input arguments
parser = argparse.ArgumentParser(description='Trainable_SpecAugment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers')

parser.add_argument('--pin_memory',default= True,
                    help='pin_memory')
parser.add_argument('--test_batch_size',type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--DEBUG',default= False,
                    help='whether or not to print error messages')

parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--USE_SHORT_AUDIO',  default= True,
                    help='number of epochs to train')
parser.add_argument('--num-nodes', type=int, default=1,
                    help='Number of available nodes/hosts')
parser.add_argument('--node-id', type=int, default=0,
                    help='Unique ID to identify the current node/host')
parser.add_argument('--num-gpus', type=int, default=4,
                    help='Number of GPUs in each node')

args = parser.parse_args()
#short_audio=USE_SHORT_AUDIO
args = parser.parse_args()
WORLD_SIZE = args.num_gpus * args.num_nodes

os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '9956' 

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



    

# ### Model class ####
# Subclass the pretrained model and make it a binary classification
# Subclass the pretrained model and make it a binary classification

# Subclass the pretrained model and make it a binary classification

class MyModel(nn.Module):
    def __init__(self, model_name, image_size,device):
        super().__init__()
        # num_classes=0 removes the pretrained head
        self.backbone = timm.create_model(model_name,
                        pretrained=True, num_classes=8, in_chans=1, 
                        drop_path_rate=0.2, global_pool='max',
                        drop_rate=0.25)
        #####  This section is model specific
        #### It freezes some fo the layers by name
        #### you'll have to inspect the model to see the names
                #### end layer freezing
        self.out = nn.Linear(self.backbone.num_features, 1)
        self.sizer = VT.Resize((image_size,image_size),antialias = True)
        self.spec_layer = features.STFT(n_fft=int(config.NFFT), freq_bins=None, hop_length=int(config.n_hop),
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect',
                           sr=config.rate, output_format="Magnitude", trainable=False,verbose = False)
        self.batch_norm = nn.BatchNorm2d(num_features= 1)
        #self.augment_layer = augment_audio(trainable = True, sample_rate = config.rate)
        
    def forward(self, x,train = True):
        # first compute spectrogram
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="nnAudio")
        spec_gram = self.spec_layer(x)
        #print("post spec gram shape = ",spec_gram.shape)
        spec_gram = self.batch_norm(spec_gram.unsqueeze(dim = 1))
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
                
                
            
        
        x = self.sizer(spec_gram.squeeze(dim = 1))
        #print("post sizer shape = ",x.shape)
        x = x.unsqueeze(dim = 1)
        #print("post unsqueeze shape = ",x.shape)
        
        # then repeat channels
        del spec_gram,spec_gram_nan_check
                  
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
#model, test_loader, criterion = nn.CrossEntropyLoss(), classes = classes ,device=device, call = "test"
def test_model(model, loader, device,criterion, call = "val"):
    
    with torch.no_grad():
        if device is None:
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        test_loss = 0.0
        model.eval()
        all_y = []
        all_y_pred = []
        for idx,(x,y) in enumerate(loader):
            x = x.to(device).float() 
            y = y.type(torch.LongTensor).to(device)
            y_pred = model(x,train = False)['prediction']
            preds = torch.argmax(y_pred, axis = 1)
            y_pred_cpu = y_pred.cpu().detach()
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
              
        test_loss = test_loss/len(loader)
        test_f1 = f1_score(all_y.numpy(), all_y_pred.numpy(),average='weighted')
               

        #dist.all_reduce(torch.tensor(test_f1), op=dist.ReduceOp.AVG)
 
    return test_loss, test_f1 , all_y,all_y_pred

## Train_model ####
#train_loader, model,classes,class_weights,train_sampler,device
def train_model(train_loader, model,classes,class_weights,train_sampler,device,e ,base_optimiser , look_optimiser  ):
    # Creates a GradScaler once at the beginning of training.
    torch.manual_seed(0)
    device = device
    print(f'Training on {device}')    
    weights_adj = torch.tensor(class_weights).type(torch.float).to(device)
    criterion_1 = nn.CrossEntropyLoss(weight=weights_adj,label_smoothing=.1)
    criterion_2 = nn.CrossEntropyLoss(weight=weights_adj)
    start_time = time.time()
    train_loss = 0.0
    model.train()
    all_y = []
    all_y_pred = []
    for batch_i, inputs in enumerate(train_loader):
        if batch_i % 200 == 0:
            bat_time = time.time()
            durn = (bat_time - start_time)/60
            print(" device = " + str(device) + "epoch = " +str(e) + "batch = " +str(batch_i) + " of " + str(len(train_loader)) + "duraation = " + str(durn) )
        x = inputs[0].to(device).float()
        y = inputs[1].type(torch.LongTensor).to(device)
        x_sum = torch.sum(x,axis = 1)
        x_sum.unsqueeze(dim = 1)
        with autocast():
            y_pred = model(x,train = True)['prediction']
            preds = torch.argmax(y_pred, axis = 1)
            if e < 20 :
                loss = criterion_1(y_pred, y)
            else:
                loss = criterion_2(y_pred, y)
                                           
        train_loss += loss.item()
        all_y.append(y.cpu().detach())
        y_pred_cpu = y_pred.cpu().detach()
        all_y_pred.append(preds.cpu().detach())
        base_optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),error_if_nonfinite=False ,max_norm = 1.0 )
        base_optimiser.step()
        del x
        del y
        del y_pred,preds
                
    look_optimiser.sync_lookahead()
    #all_train_loss.append(train_loss/len(train_loader))
    train_loss = train_loss/len(train_loader)
    all_y = torch.cat(all_y)
    all_y_pred = torch.cat(all_y_pred)
    train_f1 = f1_score(all_y.numpy(), all_y_pred.numpy(),average='weighted')
         
    return train_f1 , train_loss 
          
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

# ### Load model ####
# def load_model(filepath, model=MyModel('convnext_xlarge_in22k',224,device)):
#     # Instantiate model to inspect
#     print("Filepath = " + str(filepath))
#     print("model = " +str(model))
#     device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device("cpu"))
#     print(f'Training on {device}')

#     if torch.cuda.device_count() > 1:
#         print("Using data parallel")
#         model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
#     model = model.to(device)
#     # Load trained parameters from checkpoint (may need to download from S3 first)


#     if torch.cuda.is_available():
#         map_location=lambda storage, loc: storage.cuda()
#     else:
#         map_location='cpu'

#     checkpoint = model.load_state_dict(torch.load(filepath))

#     return model
### This function converts a np array into a tensor and performs reduction operation
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
                
    


def worker(local_rank, args):
    
    # TODO Step 4: Compute the global rank (global_rank) of the spawned process as:
    # =node_id*num_gpus + local_rank.
    # To properly initialize and synchornize each process, 
    # invoke dist.init_process_group with the approrpriate parameters:
    # backend='nccl', world_size=WORLD_SIZE, rank=global_rank
    from torch.nn.parallel import DistributedDataParallel as DDP
    warnings.filterwarnings("ignore")
    global_rank = args.node_id * args.num_gpus + local_rank 
    dist.init_process_group( backend='nccl', world_size=WORLD_SIZE, rank=global_rank ) 

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
    dist.barrier()
    
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
    device = torch.device("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu")
    print("device = ",device)
    
    model =MyModel('convnext_xlarge_in22k',224 , device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    min_length = (config.win_size * config.n_hop) / config.rate
    
    train_dataset = MozDataset(df_train_offset,  config.data_dir, min_length)
    val_dataset = MozDataset(df_val_offset,  config.data_dir, min_length)
    test_dataset = MozDataset(df_test_offset,  config.data_dir, min_length)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=WORLD_SIZE,rank=global_rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,num_replicas=WORLD_SIZE,rank=global_rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,num_replicas=WORLD_SIZE,rank=global_rank)

    
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=num_workers,batch_size = batch_size, pin_memory=True,sampler = train_sampler )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory, sampler = val_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,num_workers= num_workers, pin_memory=pin_memory, sampler = test_sampler)
      
    
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
    #model = DDP(model, device_ids=[torch.cuda.device])
    lr = .000015
    base_optimiser = timm.optim.AdamP(model.parameters(), lr= lr)
    look_optimiser = timm.optim.Lookahead(base_optimiser)
    scheduler = timm.scheduler.CosineLRScheduler(base_optimiser, t_initial= args.num_epochs,lr_min= lr/100,warmup_t = 5,warmup_lr_init= lr/10,noise_std=.075)

    
    num_epochs = args.num_epochs
    cooldown_epoch = 50
    best_val_loss = np.inf
    best_val_f1 = -np.inf
    best_train_f1 = -np.inf
    best_epoch = -1
    checkpoint_name = None
    overrun_counter = 0
    all_val_f1 = []
    all_val_loss = []
    all_train_loss = []
    all_train_f1 = []
    
     
    #(train_loader, val_loader,test_loader, model ,classes,class_weights,num_epochs,train_sampler,device )
    for e in range(num_epochs + cooldown_epoch):
        train_sampler.set_epoch(e)
        train_f1,train_loss = train_model(train_loader, model,classes,class_weights,train_sampler,device,e ,base_optimiser ,look_optimiser )
        dist.barrier()
        #averaging and reducing
        train_f1 = reduce_op(train_f1 , device)
        train_loss = reduce_op(train_loss , device)
        all_train_f1.append(train_f1)
        all_train_f1.append(train_f1)
        val_loss, val_f1 , _,_ = test_model(model, val_loader, device , criterion = nn.CrossEntropyLoss(), call = "val")
        
        dist.barrier()
        val_f1 = reduce_op(val_f1 , device)
        val_loss = reduce_op(val_loss , device)
        
        all_val_f1.append(val_f1)
        all_val_loss.append(val_loss)
        ## evaluate the validation results
        acc_metric = val_f1
        best_acc_metric = best_val_f1
        if acc_metric > best_acc_metric: 
            #dist.barrier()
            overrun_counter = -1
            checkpoint_name = f'model_e{e}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'
            torch.save(model.module.state_dict(), os.path.join(config.model_dir,  checkpoint_name))
            best_epoch = e
            best_val_f1 = val_f1
            best_val_loss = val_loss
            
            if global_rank == 0:
                print("Better val_score .." + " train_f1 = " +str(train_f1) + "val _f1 = " +str(val_f1))
                _, _ , all_y_test,all_y_pred_test = test_model(model, test_loader,device ,criterion = nn.CrossEntropyLoss(), call = "test")
                print("Device = " + str(device) + 'Epoch: %d, Train Loss: %.8f, Train f1: %.8f, Val Loss: %.8f, Val f1: %.8f, overrun_counter %i' % (e, train_loss/len(train_loader), train_f1, val_loss/len(val_loader), val_f1,  overrun_counter))
                print('Saving model to:', os.path.join(config.model_dir,  checkpoint_name)) 
                print_result(all_y_test,all_y_pred_test, classes = classes)
                
        else:
            print("on device..." +str(device) + "..Overrun....no improvement")
            overrun_counter += 1
            print("device = " + str(device) +'Epoch: %d, Train Loss: %.8f, Train f1: %.8f, Val Loss: %.8f, Val f1: %.8f, overrun_counter %i' % (e, train_loss/len(train_loader), train_f1, val_loss/len(val_loader), val_f1,  overrun_counter))
            print("current_time =",str(datetime.now()))
        
        dist.barrier()
        scheduler.step(e+1)
        
        if overrun_counter > config_pytorch.max_overrun:
            print()
            break
            
            ### print classification report#######
        scheduler.step(e + 1)
    print("Finished all epochs ALL DONE!!!!")

# #

if __name__ == '__main__':
    torch.multiprocessing.spawn(worker, nprocs=args.num_gpus, args=(args,))
