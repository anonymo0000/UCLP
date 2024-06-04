import time

from matplotlib import pyplot as plt
from myDataset import MyDataset
from torch.utils.data import DataLoader
from PMA import PMA_4#PMA_4_drop0_5
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torch.optim as optim
from sklearn.metrics import f1_score

mod = "_0"#""#"_1e-4"#
name = "root_cause"
model_index = "4"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##1 5         

#    
# path = "/home/user/Programs/PMA_lihang/data/data_v_type_train.csv"#    ,       ，    118       
# path = "/home/user/Programs/PMA_lihang/data/data118_att_type_train.csv"


# path = f"/home/user/Programs/PMA_lihang/data/myData118_{name}_train{mod}.csv" #   

# dataset = MyDataset(path)
# dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
# #    
# #   cuda    
# model = PMA_4()
# #    
# #     GPU 
# model.to(device)
# criterion = nn.CrossEntropyLoss()
# #   
# optimizer = optim.Adam(model.parameters(),lr=0.00001)
# #  
# print("start training")
# loss_all = []
# #      
# acc = 0
# f1_eval = 0.0

# for epoch in range(2101):
#     start_time = time.time()
#     loss_ = 0

#     for i,data in enumerate(dataloader):
#         inputs,labels = data
#         # labels  one-hot  ，  6 ，       
#         #labels = labels[0]
#         # print("inshape",inputs.shape)
#         # labels      
#         labels = torch.squeeze(labels)
#         optimizer.zero_grad()
#         #     GPU 
#         inputs = inputs.to(device)
#         labels = labels.to(device)
        
#         outputs = model(inputs)
#         # print("outsahpe",outputs.shape)
#         # print("labelshape",labels.shape)
#         loss = criterion(outputs,labels)
#         loss.backward()
#         optimizer.step()
#         loss_ += loss.item()


#     end_time = time.time()
#     loss_all.append(loss_)
#     print("epoch:%d,step:%d,loss:%f"%(epoch,i,loss_))
#     print("time:%f"%(end_time-start_time))


#     #    ，     

#     if epoch % 50 == 0:

#         model.eval()
#         # path = "/home/user/Programs/PMA_lihang/data/data_v_type_eval.csv"
#         # path = "/home/user/Programs/PMA_lihang/data/data118_att_type_eval.csv"
#         path = f"/home/user/Programs/PMA_lihang/data/myData118_{name}_eval{mod}.csv"
#         dataset = MyDataset(path)
#         dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
#         correct = 0
#         total = 0

#         ##    f1
#         y_true = []
#         y_pred = []
        

    
#         print("start eval")
#         for i,data in enumerate(dataloader):
#             inputs,labels = data

#             for label in labels:
#                 y_true.append(label[0])

#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             labels = torch.squeeze(labels)
#             outputs = model(inputs)
#             predict = torch.max(outputs,1)[1]

#             for i in range(len(predict)):
#                 y_pred.append(predict.cpu()[i])

#             total += labels.size(0)
#             correct += (predict == labels).sum()
#         print("correct:%d,total:%d,acc:%f"%(correct,total,correct/total))
#         #                ，    

#         f1_eval_now = f1_score(y_true,y_pred,average = 'weighted')
#         print("f1",f1_eval_now)
#         # if correct/total > acc:
#         if f1_eval_now > f1_eval:
#             f1_eval = f1_eval_now
#             acc = correct/total
#             torch.save(model.state_dict(),f"/home/user/Programs/PMA_lihang/code_lihang/model/myModel118_{model_index}{mod}.pth")
#             print("*************epoch:",epoch,"****save model***********")




# # loss    
# path= "/home/user/Programs/PMA_lihang/code_lihang/model/train_log_1.txt"
# f = open(path,'w')
# for i in range(len(loss_all)):
#     f.write(str(loss_all[i]))
#     f.write("\n")
    
# #    
# torch.save(model.state_dict(),"/home/user/Programs/PMA_lihang/code_lihang/model/model_1.pth")



model = PMA_4()
# model.load_state_dict(torch.load("/home/user/Programs/PMA_lihang/code_lihang/model/model_1_1k.pth"))
model.load_state_dict(torch.load(f"/home/user/Programs/PMA_lihang/code_lihang/model/myModel118_{model_index}{mod}.pth"))
#     GPU 
model.to(device)

model.eval()
#      
# path = "/home/user/Programs/PMA_lihang/data/data118_att_type_test.csv"

# path = f"/home/user/Programs/PMA_lihang/data/myData118_{name}_test{mod}.csv"
path = "/home/user/Programs/PMA_lihang/data/myData118_root_cause_test.csv"

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


##    f1
y_true = []
y_pred = []


for i,data in enumerate(dataloader):
    inputs,labels_ = data

    for label in labels_:
        y_true.append(label[0])


    labels_ = torch.squeeze(labels_)
    #     GPU 
    inputs = inputs.to(device)
    labels_ = labels_.to(device)

    outputs = model(inputs)
    predict = torch.max(outputs,1)[1]
    resoults.append(predict)
    total += labels_.size(0)
    correct += (predict == labels_).sum()
    #             
    for i in range(len(predict)):
        y_pred.append(predict.cpu()[i])
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

from sklearn.metrics import f1_score
# print(y_true[0:10])
# print(y_pred[0:10])

f1_ = f1_score(y_true, y_pred, average='weighted')

print("  f1",f1_)

