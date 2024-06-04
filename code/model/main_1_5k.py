import time
from myDataset import MyDataset
from torch.utils.data import DataLoader
from PMA import PMA_1_5kernel
import torch
import torch.nn as nn

import torch.optim as optim



#1 5         

#    
# path = "/home/user/Programs/PMA_lihang/data/data_v_type_train.csv"
path = "/home/user/Programs/PMA_lihang/data/myData_v_type_train.csv"

dataset = MyDataset(path)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
#    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PMA_1_5kernel()
model.to(device)
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
    for i,data in enumerate(dataloader):
        inputs,labels = data
        # labels  one-hot  ，  6 ，       
        #labels = labels[0]
        # print("inshape",inputs.shape)
        # labels      
        inputs = inputs.to(device)
        labels = labels.to(device)
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
    #    ，     
    model.eval()
    # path = "/home/user/Programs/PMA_lihang/data/data_v_type_eval.csv"
    path = "/home/user/Programs/PMA_lihang/data/myData_v_type_eval.csv"
    dataset = MyDataset(path)
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True,drop_last=True)
    correct = 0
    total = 0
    
    print("start eval")
    for i,data in enumerate(dataloader):
        inputs,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels)
        outputs = model(inputs)
        predict = torch.max(outputs,1)[1]
        total += labels.size(0)
        correct += (predict == labels).sum()
    print("correct:%d,total:%d,acc:%f"%(correct,total,correct/total))
    #                ，    
    if correct/total > acc:
        acc = correct/total
        torch.save(model.state_dict(),"/home/user/Programs/PMA_lihang/code_lihang/model/myModel_1_5k.pth")
        print("*************epoch:",epoch,"****save model***********")


# # loss    
# path= "/home/user/Programs/PMA_lihang/code_lihang/model/train_log_1.txt"
# f = open(path,'w')
# for i in range(len(loss_all)):
#     f.write(str(loss_all[i]))
#     f.write("\n")
    
# #    
# torch.save(model.state_dict(),"/home/user/Programs/PMA_lihang/code_lihang/model/model_1.pth")



model = PMA_1_5kernel()
model.load_state_dict(torch.load("/home/user/Programs/PMA_lihang/code_lihang/model/myModel_1_5k.pth"))
model.to(device)
model.eval()
#      
# path = "/home/user/Programs/PMA_lihang/data/data_v_type_test.csv"
path = "/home/user/Programs/PMA_lihang/data/myData_v_type_test.csv"
dataset = MyDataset(path)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
#  ，
correct = 0
total = 0
print("start predict")
resoults = []
#  13 。             
#           
class_num = 14
corrects = [0 for i in range(class_num)]
#           
errors = [0 for i in range(class_num)]
#           
labels = [0 for i in range(class_num)]
for i,data in enumerate(dataloader):
    inputs,labels_ = data
    inputs = inputs.to(device)
    labels_ = labels_.to(device)
    labels_ = torch.squeeze(labels_)
    outputs = model(inputs)
    predict = torch.max(outputs,1)[1]
    resoults.append(predict)
    total += labels_.size(0)
    correct += (predict == labels_).sum()
    #             
    for i in range(len(predict)):
        if predict[i] == labels_[i]:
            corrects[predict[i]] += 1
        else:
            errors[predict[i]] += 1
    #        
    for i in range(len(labels_)):
        labels[labels_[i]] += 1
    # break
print("correct:%d,total:%d,acc:%f"%(correct,total,correct/total))
# print("resoults:",resoults)
print("corrects:",corrects)
print("errors:",errors)
print("labels:",labels)
#             
for i in range(len(corrects)):
    if corrects[i] == 0:
        precision = 0
    else:
        precision = corrects[i]/(corrects[i]+errors[i])
    if labels[i] == 0:
        recall = 0
    else:
        recall = corrects[i]/labels[i]
    print(" %d      %f,    %f"%(i,precision,recall))
#  f1 ，             0  
f1 = 0
count = 0
for i in range(len(corrects)):
    if corrects[i] == 0:
        precision = 0
    else:
        count += 1
        precision = corrects[i]/(corrects[i]+errors[i])
    if labels[i] == 0:
        recall = 0
    else:
        recall = corrects[i]/labels[i]
    if precision == 0 or recall == 0:
        continue
    f1 += 2*precision*recall/(precision+recall)
f1 = f1/count
print("f1:",f1)

