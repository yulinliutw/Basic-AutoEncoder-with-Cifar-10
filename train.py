import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      
import matplotlib.pyplot as plt
from load_data import load_data
import numpy as np
from autoencoder import AutoEncoder
from torch.autograd import Variable
import pickle
import os,sys

'''init setting'''
parser = argparse.ArgumentParser(description='Autoencoder_train: Inference Parameters')
parser.add_argument('--epoch',
                    type=int,
                    default=600,
                    help='training epoch setting')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.00005,
                    help='learning rate setting')
parser.add_argument('--save_weight_dir',
                    default = './train_weight1/',
                    help    = 'Path to folder of saving weight')
parser.add_argument('--load_weight_dir',
                    default = './better_weight/checkpoint_ep599_itir_999.pkl',
                    help    = 'Path to folder of saving weight')
parser.add_argument('--save_loss_figure_dir',
                    default = './loss_figure.pickle',
                    help    = 'Path to folder of saving loss figure, , if you dont rename it, it will combine current and previous training result')
parser.add_argument('--gpuid',
                    default = 0,
                    type    = int,
                    help    = 'GPU device ids (CUDA_VISIBLE_DEVICES)')

'''gobal setting'''
global args
args = parser.parse_args()
torch.manual_seed(0)

'''training setting'''
EPOCH = args.epoch
loss_iter = 0
try:
    with open(args.save_loss_figure_dir, 'rb') as file:
        total_loss =pickle.load(file)
except:  
    total_loss = {'losses_train':[],'losses_val':[]} 

'''set the training gpu''' 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

'''load_data'''
Load_data = load_data()
train_data = Load_data.train()
val_data = Load_data.val()

'''init model'''
autoencoder = AutoEncoder()
model_dict = autoencoder.state_dict()
try:
    pre_train_path = args.load_weight_dir
    pretrained_dict = torch.load(pre_train_path) #load pre train model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} #load the layer only same with the target model
    model_dict.update(pretrained_dict)
    print('===================================')
    print('load pre_train weight successfully')
    print('===================================')
except: 
    print('===================================')
    print('       random init the weight      ')
    print('===================================')
autoencoder.load_state_dict(model_dict) 
autoencoder.cuda()
autoencoder.train()


'''opt setting'''
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.learning_rate)   # optimize all cnn parameters
loss_func = nn.L1Loss()   

'''golder for saving  weight'''
save_path=args.save_weight_dir

for epoch in range(EPOCH):
    loss_iter=0
    for step, (x, b_label) in enumerate(train_data):    
        
        x_in = torch.tensor(x, dtype=torch.float32).cuda()        
        decoded = autoencoder(x_in)  
        
        loss = loss_func(decoded, x_in)   # L1 loss
        
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients        
        loss_iter = loss_iter+loss.data.cpu().numpy()
        '''save weight'''
        if(((step+1)%200)==0):
            filename = 'checkpoint_ep'+str(epoch)+'_itir_'+str(step)+'.pkl'
            filename = os.path.join(save_path, filename)  
            torch.save(autoencoder.state_dict(), filename) 
            
    '''caculate one epoch loss for training set'''   
    loss_iter = loss_iter/(step+1)
    total_loss['losses_train'].append(loss_iter)    
    
    '''caculate one epoch loss for val set'''
    with torch.no_grad(): #it can save the memory,prevent it allocated,we dont need to keep the grad during the evualation
        loss_val=0
        for index in range(0,val_data.size()[0],50):
            x_in = torch.tensor(val_data[index:index+49,:,:,:], dtype=torch.float32).cuda() 
            decoded = autoencoder(x_in)
            loss_val = loss_val+loss_func(decoded, x_in)   # L1 loss
        
    total_loss['losses_val'].append(loss/(val_data.size()[0]/50))    
    
    '''draw the loss figure'''
    plt.title('loss_figure(L1)')
    plt.plot(total_loss['losses_train'],label='training loss')
    plt.plot(total_loss['losses_val'],label='val loss')
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')    
    plt.show()
    
#saving the loss
file = open(args.save_loss_figure_dir, 'wb')
pickle.dump(total_loss, file)
file.close()
       
