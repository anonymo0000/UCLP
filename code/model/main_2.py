import time
from myDataset import MyDataset
from torch.utils.data import DataLoader
from PMA import PMA_2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#    
# path = "/home/user/Programs/PMA_lihang/data/data_att_type_.csv"
path = ""
dataset = MyDataset(path)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
#    
model = PMA_2()
#    
criterion = nn.CrossEntropyLoss()
#   
optimizer = optim.Adam(model.parameters(),lr=0.00001)
#  
print("start training")
loss_all = []
for epoch in range(10):
    start_time = time.time()
    loss_ = 0
    for i,data in enumerate(dataloader):
        inputs,labels = data
        # labels  one-hot  ，  6 ，       
        #labels = labels[0]
        # print("inshape",inputs.shape)
        # labels      
        labels = torch.squeeze(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print("outsahpe",outputs.shape)
        # print("labelshape",labels.shape)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        loss_ += loss.item()
    end_time = time.time()
    loss_all.append(loss_)
    print("epoch:%d,step:%d,loss:%f"%(epoch,i,loss_))
    print("time:%f"%(end_time-start_time))
    
# loss    
path= "/home/user/Programs/PMA_lihang/code_lihang/model/train_log_2.txt"
f = open(path,'w')
for i in range(len(loss_all)):
    f.write(str(loss_all[i]))
    f.write("\n")

#    
torch.save(model.state_dict(),"/home/user/Programs/PMA_lihang/code_lihang/model/model_2.pth")
