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
import argparse

'''init setting'''
parser = argparse.ArgumentParser(description='Autoencoder_eval: Inference Parameters')
parser.add_argument('--load_weight_dir',
                    default = './better_weight/checkpoint_ep599_itir_999.pkl',
                    help    = 'Path to folder of saving weight')
parser.add_argument('--save_loss_figure_dir',
                    default = './loss_figure_1.pickle',
                    help    = 'Path to folder of saving loss figure')
parser.add_argument('--gpuid',
                    default = 0,
                    type    = int,
                    help    = 'GPU device ids (CUDA_VISIBLE_DEVICES)')
global args
args = parser.parse_args()

'''set the training gpu''' 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

'''load_data'''
Load_data=load_data()
test_data=Load_data.test()

'''init model'''
autoencoder = AutoEncoder()
autoencoder.load_state_dict(torch.load(args.load_weight_dir)) #load pre-train
autoencoder.cuda()
autoencoder.eval()

loss_func = nn.L1Loss() 
loss = 0

with torch.no_grad(): #it can save the memory,prevent it allocated,we dont need to keep the grad during the evualation
    for index in range(0,test_data.size()[0],50):
        x_in = torch.tensor(test_data[index:index+49,:,:,:], dtype=torch.float32).cuda() 
        decoded = autoencoder(x_in)
        loss = loss+loss_func(decoded, x_in)   # L1 loss 
        #pick some sample to check vision performance
        plt.title('autoencoder input')
        plt.imshow(np.transpose(x_in[0,:,:,:].data.cpu().numpy(),(1,2,0)))
        plt.show()         
        plt.title('autoencoder outoput')
        plt.imshow(np.transpose(decoded[0,:,:,:].data.cpu().numpy(),(1,2,0)))
        plt.show()         
       
loss = loss/(test_data.size()[0]/50)
result = loss.data.cpu().numpy()
print('average testing loss per pixel(L1):')
print(loss)

'''print training history'''
try:
    with open(args.save_loss_figure_dir, 'rb') as file:
        total_loss = pickle.load(file)    
        '''draw the loss figure'''
        plt.title('training_loss_figure(L1)')
        plt.plot(total_loss['losses_train'],label='training loss')
        plt.plot(total_loss['losses_val'],label='val loss')
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')    
        plt.show()  
except: 
    print("no file to show the training history")