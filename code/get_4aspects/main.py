from module.pt1_ import devide1
from module.pt2_ import devide2
from module.pt3_ import devide3
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from module.myDataset import MyDataset
from module.PMA import PMA_1, PMA_2, PMA_3, PMA_4
from tqdm import tqdm


path_in = "data/row_test.csv"



path_temp = "data/temp.csv"
path_temp2 = "data/temp2.csv"
path_temp3 = "data/temp3.csv"
path_temp4 = "data/temp4.csv"

path_out = "data/out.csv"



path_result = "data/result.csv"



f_temp = open(path_temp, 'w', encoding='utf-8')
f_temp.truncate()
f_temp.close()
f_temp = open(path_temp2, 'w', encoding='utf-8')
f_temp.truncate()
f_temp.close()
f_temp = open(path_temp3, 'w', encoding='utf-8')
f_temp.truncate()
f_temp.close()
f_temp = open(path_temp4, 'w', encoding='utf-8')
f_temp.truncate()
f_temp.close()
f_temp = open(path_out, 'w', encoding='utf-8')
f_temp.truncate()
f_temp.close()
f_temp = open(path_result, 'w', encoding='utf-8')
f_temp.truncate()
f_temp.close()

# exit()

f_in = open(path_in, 'r', encoding='utf-8')
lines_in = csv.reader(f_in)
f_temp = open(path_temp, 'a', encoding='utf-8')

for line in lines_in:
    data = ""
    data = "CVE-YYYY-xxxx"+"\t##=divide=##\t"+line[0] + "\n"
    f_temp.write(data)
    # break
devide1(path_temp, path_out,path_temp2)
devide2(path_temp2, path_out,path_temp3)
devide3(path_temp3, path_out,path_temp4)

v_type = ["Cross site scripting","SQL injection","Buffer overflow","Directory traversal","Cross-site request forgery","PHP file inclusion","Use-after-free","Integer overflow","Untrusted search path","Format string","CRLF injection","XML External Entity","Others"]
root_cause = ["Input Validation Error","Boundary Condition Error","Failure to Handle Exceptional Conditions","Design Error","Access Validation Error","Atomicity Error","Race Condition Error","Serialization Error","Configuration Error","Origin Validation Error","Environment Error"]
att_vector = ["Environment Error","Via some crafted data","By executing the script","HTTP protocol correlation","Call API","Others"]
att_type = ["Remote attacker","Local attacker","Authenticated user","Context-dependent","Physically proximate attacker","Others"]




f_out = open(path_out, 'r', encoding='utf-8')
lines_out = csv.reader(f_out)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model2 = PMA_2()
model2.load_state_dict(torch.load("module/model_2.pth"))
model2.to(device)
model2.eval()

model3 = PMA_3()
model3.load_state_dict(torch.load("module/model_3.pth"))
model3.to(device)
model3.eval()

model4 = PMA_4()
model4.load_state_dict(torch.load("module/model_4.pth"))
model4.to(device)
model4.eval()

model1 = PMA_1()
model1.load_state_dict(torch.load("module/model_1.pth"))
model1.to(device)
model1.eval()

ii = 0
for line in tqdm(lines_out):
    aspect_4 = []
    for i in range(5):
        aspect_4.append("")
    
    # print(line[0])
    # exit()
    #1:vt 2:att_type 3:att_vector 4:root_cause
    aspect_4[0] = line[0]
    if(line[2] == ""):

        model = PMA_2()
        model.load_state_dict(torch.load("module/model_2.pth"))
        model.eval()

        path_train = "data/toTrain.csv"
        f_now = open(path_train, 'w', encoding='utf-8')
        f_now.truncate()
        
        data = []
        temp = ""
        for i in range(6):
            temp += line[i+1]
        data.append(line[0])
        data.append(temp)
        data.append("0")
        writer = csv.writer(f_now)
        writer.writerow(data)
        f_now.close()
        dataset = MyDataset(path_train)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        correct = 0
        total = 0
        # print("start predict")

        for i,data in enumerate(dataloader):
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print("inputs:",inputs)
            # print("labels:",labels)
            labels = torch.squeeze(labels)
            outputs = model2(inputs)
            predict = torch.max(outputs,1)[1]
            # print("predict*************:",predict)
            predict = predict.cpu()
            predict = predict.numpy()
            predict = predict[0]
            res = att_type[predict]
            aspect_4[4] = res
            break
    else:
        aspect_4[4] = line[2]

        
    if(line[4] == ""):#

        path_train = "data/toTrain.csv"
        f_now = open(path_train, 'w', encoding='utf-8')
        f_now.truncate()
        
        data = []
        temp = ""
        for i in range(6):
            temp += line[i+1]
        data.append(line[0])
        data.append(temp)
        data.append("0")
        writer = csv.writer(f_now)
        writer.writerow(data)
        f_now.close()
        dataset = MyDataset(path_train)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        correct = 0
        total = 0

        for i,data in enumerate(dataloader):
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print("inputs:",inputs)
            # print("labels:",labels)
            labels = torch.squeeze(labels)
            outputs = model3(inputs)
            predict = torch.max(outputs,1)[1]
            # print("predict*************:",predict)
            predict = predict.cpu()
            predict = predict.numpy()
            predict = predict[0]
            res = att_vector[predict]
            aspect_4[3] = res
            break
    else:
        aspect_4[3] = line[4]

    if(line[5] == ""):

        path_train = "data/toTrain.csv"
        f_now = open(path_train, 'w', encoding='utf-8')
        f_now.truncate()
        
        data = []
        temp = ""
        for i in range(6):
            temp += line[i+1]
        data.append(line[0])
        data.append(temp)
        data.append("0")
        writer = csv.writer(f_now)
        writer.writerow(data)
        f_now.close()
        dataset = MyDataset(path_train)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        correct = 0
        total = 0

        for i,data in enumerate(dataloader):
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print("inputs:",inputs)
            # print("labels:",labels)
            labels = torch.squeeze(labels)
            outputs = model4(inputs)
            predict = torch.max(outputs,1)[1]

            predict = predict.cpu()
            predict = predict.numpy()
            predict = predict[0]
            res = root_cause[predict]
            aspect_4[2] = res
            break
    else:
        aspect_4[2] = line[5]
    
    if(line[6] == ""):


        path_train = "data/toTrain.csv"
        f_now = open(path_train, 'w', encoding='utf-8')
        f_now.truncate()
        
        data = []
        temp = ""
        for i in range(6):
            temp += line[i+1]
        data.append(line[0])
        data.append(temp)
        data.append("0")
        writer = csv.writer(f_now)
        writer.writerow(data)
        f_now.close()
        dataset = MyDataset(path_train)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        correct = 0
        total = 0

        for i,data in enumerate(dataloader):

            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print("inputs:",inputs)
            # print("labels:",labels)
            labels = torch.squeeze(labels)
            outputs = model1(inputs)
            predict = torch.max(outputs,1)[1]
            # print("predict*************:",predict)

            predict = predict.cpu()
            predict = predict.numpy()
       
            predict = predict[0]
            res = v_type[predict]
            aspect_4[1] = res
            break
    else:
        aspect_4[1] = line[6]

   

    f = open(path_result, 'a', encoding='utf-8')

    writer = csv.writer(f)
    writer.writerow(aspect_4)


    ii += 1
    if(ii == 100):
        break
    # break