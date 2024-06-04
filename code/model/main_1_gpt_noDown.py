from transformers import BertTokenizer
import time
from myDataset import MyDatasetForGpt2 as MyDataset
from torch.utils.data import DataLoader
from PMA import Text2Transformer3  # Text2norLstm #Text2norLstm #Text2norLstm #
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import f1_score
from transformers import GPT2Tokenizer

mod = "_1e-4"  # ""#"_0"#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 1 5         

#     
# path = "/home/user/Programs/PMA_lihang/data/data_v_type_train.csv"#    ,       ，    118       
# path = "/home/user/Programs/PMA_lihang/data/data118_v_type_train.csv"
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<pad>')  # Specify pad_token here
# model = Text2Transformer3("gpt2")


# model_path = '/home/user/Programs/PMA_lihang/code_lihang/pretrained_gpt2'
#          
# model_path = "/path/to/local/directory"  # Replace with the actual path
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  #   PAD

# path = f"/home/user/Programs/PMA_lihang/data/myData118_v_type_train{mod}_v2.csv"  #    
path = "/home/user/Programs/PMA_lihang/data/data118_v_type_train.csv"
dataset = MyDataset(path, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

# path_eval = f"/home/user/Programs/PMA_lihang/data/myData118_v_type_eval{mod}_v2.csv"  #    
path_eval = "/home/user/Programs/PMA_lihang/data/data118_v_type_eval.csv"
dataset_eval = MyDataset(path_eval, tokenizer)
dataloader_eval = DataLoader(dataset_eval, batch_size=8, shuffle=True, drop_last=True)

# path_test = f"/home/user/Programs/PMA_lihang/data/myData118_v_type_test{mod}_v2.csv"  #    
path_test = "/home/user/Programs/PMA_lihang/data/data118_v_type_test.csv"
dataset_test = MyDataset(path_test, tokenizer)
dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=True, drop_last=True)

#         batch
# print("batch:", dataloader.__len__())
# print("batch_eval:", dataloader_eval.__len__())
# print("batch_test:", dataloader_test.__len__())


# print(dataset.__len__())
# print(dataset_eval.__len__())
# print(dataset_test.__len__()/64)

# exit()
#     
#   cuda    


# model = Text2Transformer3("gpt2")
#     
#      GPU 
model.to(device)
criterion = nn.CrossEntropyLoss()
#    
optimizer = optim.Adam(model.parameters(), lr=0.00001)


#   
print("start training")
loss_all = []
#       
acc = 0

f1_eval = 0

flag = 0  #     f1      ，     

for epoch in range(10):
    print("epoch:", epoch)
    start_time = time.time()
    loss_ = 0
    model.train()
    print("start train")
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # print("inputs:", inputs)
        # print("labels:", labels)

        optimizer.zero_grad()
        outputs = model(inputs, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_ += loss.item()

        #  100 batch    loss
        if i % 50 == 0:
            print(i, " / ", dataloader.__len__(), " train batch")

    end_time = time.time()
    loss_all.append(loss_)
    print("epoch:%d,step:%d,loss:%f" % (epoch, i, loss_))
    print("time:%f" % (end_time - start_time))
    # exit()
    #     ，     

    # if epoch % 50 == 0:
    model.eval()
    y_true = []
    y_pred = []

    for i, data in enumerate(dataloader_eval):
        inputs = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)

        for label in labels:
            y_true.append(label.item())

        outputs = model(inputs, attention_mask=attention_mask)
        predict = torch.max(outputs, 1)[1]

        for i in range(len(predict)):
            y_pred.append(predict.cpu()[i].item())

        if i % 50 == 0:
            print(i, " / ", dataloader_eval.__len__(), " eval batch")

    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / total
    f1_eval_now = f1_score(y_true, y_pred, average='weighted')

    print("correct:%d,total:%d,acc:%f" % (correct, total, accuracy))
    print("f1", f1_eval_now)

    if f1_eval_now > f1_eval + 0.005:
        flag = 0
        f1_eval = f1_eval_now
        # torch.save(model.state_dict(), f"/home/user/Programs/PMA_lihang/code_lihang/model/myModel118_1_bert.pth")
        torch.save(model.state_dict(), f"/home/user/Programs/PMA_lihang/code_lihang/model/model118_1_bert_temp.pth")
        print("*************epoch:", epoch, "****save model***********")
    else:
        print("*************epoch:", epoch, "***************************not save model************************")
        flag += 1
        if flag == 3:
            print("*************epoch:", epoch, "****break***********")
            break


#     
exit(0)
model = Text2Transformer("bert-base-uncased")  #         
model.load_state_dict(torch.load(f"/home/user/Programs/PMA_lihang/code_lihang/model/model118_1_bert.pth"))
model.to(device)
model.eval()

#       
# path = "/home/user/Programs/PMA_lihang/data/data118_v_type_test.csv"
# dataset = MyDataset(path, tokenizer)  #      tokenizer
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

#     
correct = 0
total = 0
print("start predict")
resoults = []

#              
class_num = 14
corrects = [0 for _ in range(class_num)]
errors = [0 for _ in range(class_num)]
labels = [0 for _ in range(class_num)]

y_true = []
y_pred = []

with torch.no_grad():
    for i, data in enumerate(dataloader_test):
        inputs = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels_ = data['label'].to(device)

        for label in labels_:
            y_true.append(label.item())

        outputs = model(inputs, attention_mask=attention_mask)
        predict = torch.max(outputs, 1)[1]
        resoults.append(predict)

        total += labels_.size(0)
        correct += (predict == labels_).sum()

        for i in range(len(predict)):
            y_pred.append(predict.cpu()[i].item())

        for i in range(len(predict)):
            if predict[i] == labels_[i]:
                corrects[predict[i]] += 1
            else:
                errors[predict[i]] += 1
            labels[labels_[i]] += 1

print("correct:%d,total:%d,acc:%f" % (correct, total, correct/total))
print("corrects:", corrects)
print("errors:", errors)
print("labels:", labels)

#              
for i in range(len(corrects)):
    if corrects[i] == 0:
        precision = 0
    else:
        precision = corrects[i] / (corrects[i] + errors[i])
    if labels[i] == 0:
        recall = 0
    else:
        recall = corrects[i] / labels[i]
    print(" %d      %f,    %f" % (i, precision, recall))

#   f1 ，             0  
f1 = 0
count = 0
for i in range(len(corrects)):
    if corrects[i] == 0:
        precision = 0
    else:
        count += 1
        precision = corrects[i] / (corrects[i] + errors[i])
    if labels[i] == 0:
        recall = 0
    else:
        recall = corrects[i] / labels[i]
    if precision == 0 or recall == 0:
        continue
    f1 += 2 * precision * recall / (precision + recall)
f1 = f1 / count
print("f1:", f1)

f1_ = f1_score(y_true, y_pred, average='weighted')
print("  f1", f1_)
