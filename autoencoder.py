import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      
import matplotlib.pyplot as plt
from load_data import load_data
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,      # input height
                out_channels=16,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      
            ),      
            nn.LeakyReLU(),    # activation
            nn.Conv2d(
                in_channels=16,      # input height
                out_channels=32,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      
            ),     
            nn.LeakyReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),     
            nn.Conv2d(
                in_channels=32,      # input height
                out_channels=32,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),      
            nn.LeakyReLU(),    # activation
            nn.Conv2d(
                in_channels=32,      # input height
                out_channels=64,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),      
            nn.LeakyReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    
        )
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(
                in_channels=64,      # input height
                out_channels=32,    # n_filters
                kernel_size=2,      # filter size
                stride=2,           # filter movement/step
                padding=0,      
            ),     
            nn.LeakyReLU(),       # activation 
            nn.Conv2d(
                in_channels=32,      # input height
                out_channels=32,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),      
            nn.LeakyReLU(),    # activation
           nn.ConvTranspose2d(
                in_channels=32,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),
            nn.Conv2d(
                in_channels=16,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,     
            ),      
            nn.LeakyReLU(),    # activation       
             nn.ConvTranspose2d(
                in_channels=16,      # input height
                out_channels=16,    # n_filters
                kernel_size=2,      # filter size
                stride=2,           # filter movement/step
                padding=0,      
            ), 
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,      # input height
                out_channels=16,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      
            ),     
            nn.LeakyReLU(),    # activation            
           nn.ConvTranspose2d(
                in_channels=16,      # input height
                out_channels=3,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),
            nn.Conv2d(
                in_channels=3,      # input height
                out_channels=3,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      
            ),    
            nn.ReLU(), # activation 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

