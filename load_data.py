import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision     
import matplotlib.pyplot as plt

torch.manual_seed(1)    
class load_data:
    def __init__(self,epoch=1,batch_size=50):
        self.EPOCH = epoch           
        self.BATCH_SIZE = batch_size
    def train(self):
        
        train_data = torchvision.datasets.CIFAR10(
            root='./CIFAR10/',   
            train=True,  # this is training data
            transform=torchvision.transforms.ToTensor(),
            download=True                                                      
        )        
        train_loader = Data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True)
        return train_loader
        
    def val(self):
        test_data = torchvision.datasets.CIFAR10(root='./CIFAR10/', train=False,download=True)        
        test_x = torch.tensor(test_data.data[:4000].astype('float')/255.)   
        test_x = test_x.permute(0,3,1,2)
        return test_x
        
    def test(self):
        test_data = torchvision.datasets.CIFAR10(root='./CIFAR10/', train=False,download=True)        
        test_x = torch.tensor(test_data.data[4000:].astype('float')/255.)   
        test_x = test_x.permute(0,3,1,2)    
        return test_x
        

#
    