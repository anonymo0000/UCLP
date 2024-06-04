import time

from matplotlib import pyplot as plt
from myDataset import MyDataset
from torch.utils.data import DataLoader
from PMA import PMA_3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#    
path = "/home/user/Programs/PMA_lihang/data/data_att_vec_train.csv"
dataset = MyDataset(path)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
#    
model = PMA_3()
#    
criterion = nn.CrossEntropyLoss()
#   
optimizer = optim.Adam(model.parameters(),lr=0.00001)
#  
print("start training")
loss_all = []
#      
acc = 0
for epoch in range(20):
    start_time = time.time()
    loss_ = 0
    model.train()

    #    batch loss
    loss_batch = []

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
        # break

        loss_batch.append(loss.item())
    plt.plot(loss_batch)
    plt.show()

    # print("loss_batch:",loss_batch)
    #   batch loss    
    print("    ")
    path = "/home/user/Programs/PMA_lihang/code_lihang/model/loss.txt"
    with open(path,"a") as f:
        for i in range(len(loss_batch)):
            f.write(str(loss_batch[i])+"\n")
            

    end_time = time.time()
    loss_all.append(loss_)
    print("epoch:%d,step:%d,loss:%f"%(epoch,i,loss_))
    print("time:%f"%(end_time-start_time))
    #    ，     
    model.eval()
    path = "/home/user/Programs/PMA_lihang/data/data_att_vec_eval.csv"
    dataset = MyDataset(path)
    dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
    correct = 0
    total = 0
    
    print("start eval")
    for i,data in enumerate(dataloader):
        inputs,labels = data
        labels = torch.squeeze(labels)
        outputs = model(inputs)
        predict = torch.max(outputs,1)[1]
        total += labels.size(0)
        # print("outputs:",outputs)
        # print("size:",labels.size(0))
        # print("labels:",labels)
        # print("predict:",predict)
        correct += (predict == labels).sum()
        break
    print("correct:%d,total:%d,acc:%f"%(correct,total,correct/total))
    # break
    #                ，    
    if correct/total > acc:
        acc = correct/total
        torch.save(model.state_dict(),"/home/user/Programs/PMA_lihang/code_lihang/model/model_3.pth")
        print("*************epoch:",epoch,"****save model***********")
    # break
# loss    
# path= "/home/user/Programs/PMA_lihang/code_lihang/model/train_log.txt"
# f = open(path,'w')
# for i in range(len(loss_all)):
#     f.write(str(loss_all[i]))
#     f.write("\n")

# #    
# torch.save(model.state_dict(),"/home/user/Programs/PMA_lihang/code_lihang/model/model.pth")


#    
# exit()
#    ，    
model.load_state_dict(torch.load("/home/user/Programs/PMA_lihang/code_lihang/model/model_3.pth"))
model.eval()
#      
path = "/home/user/Programs/PMA_lihang/data/data_att_vec_test.csv"
dataset = MyDataset(path)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)
#  ，     
correct = 0
total = 0
print("start predict")
for i,data in enumerate(dataloader):
    inputs,labels = data
    labels = torch.squeeze(labels)
    outputs = model(inputs)
    predict = torch.max(outputs,1)[1]
    total += labels.size(0)
    correct += (predict == labels).sum()
print("correct:%d,total:%d,acc:%f"%(correct,total,correct/total))

