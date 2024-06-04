from module.pt1_ import devide1
from module.pt2_ import devide2
from module.pt3_ import devide3
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from module.myDataset import MyDataset
from module.PMA import PMA_1, PMA_2, PMA_3, PMA_4
from tqdm import tqdm

def get_4aspects(description):
    print("开始处理")
    resoult_list = []
        
    path_in = "data/row_test.csv"
    path_temp = "data/temp.csv"
    path_temp2 = "data/temp2.csv"
    path_temp3 = "data/temp3.csv"
    path_temp4 = "data/temp4.csv"
    path_out = "data/out.csv"
    path_result = "data/result.csv"

    

        # break
    # return resoult_list
    # ftemp = open(path_temp2, 'w', encoding='utf-8')
    # ftemp.truncate()
    # ftemp.close()


    # ftemp = open(path_temp, 'w', encoding='utf-8')
    # ftemp.truncate()
    # ftemp.close()
    ftemp2 = open(path_temp2, 'w', encoding='utf-8')
    ftemp2.truncate()
    ftemp2.close()

    ftemp3 = open(path_temp3, 'w', encoding='utf-8')
    ftemp3.truncate()
    ftemp3.close()

    ftemp4 = open(path_temp4, 'w', encoding='utf-8')
    ftemp4.truncate()
    ftemp4.close()

    ftemp5 = open(path_out, 'w', encoding='utf-8')
    ftemp5.truncate()
    ftemp5.close()

    ftemp6 = open(path_result, 'w', encoding='utf-8')
    ftemp6.truncate()
    ftemp6.close()

    f_temp = open(path_temp, 'w', encoding='utf-8')
    data = "CVE-YYYY-xxxx"+"\t##=divide=##\t"+description
    f_temp.write(data)
    f_temp.close()
    # exit()
    
    
    # return resoult_list
    devide1(path_temp, path_out,path_temp2)
    devide2(path_temp2, path_out,path_temp3)
    devide3(path_temp3, path_out,path_temp4)

    # return resoult_list

    v_type = ["Cross site scripting","SQL injection","Buffer overflow","Directory traversal","Cross-site request forgery","PHP file inclusion","Use-after-free","Integer overflow","Untrusted search path","Format string","CRLF injection","XML External Entity","Others"]
    root_cause = ["Input Validation Error","Boundary Condition Error","Failure to Handle Exceptional Conditions","Design Error","Access Validation Error","Atomicity Error","Race Condition Error","Serialization Error","Configuration Error","Origin Validation Error","Environment Error"]
    att_vector = ["via field argument or parameter","Via some crafted data","By executing the script","HTTP protocol correlation","Call API","Others"]
    att_type = ["Remote attacker","Local attacker","Authenticated user","Context-dependent","Physically proximate attacker","Others"]



    f_out = open(path_out, 'r', encoding='utf-8')
    lines_out = csv.reader(f_out)
    flag = False
    for line in lines_out:
        flag = True
        break
        # print(line)#['CVE-YYYY-xxxx', 'BSD-derived TCP/IP implementations ', 'remote attackers', 'cause a denial of service (crash or hang)', ' via crafted packets.', '', 'ip_input.c']
    if flag == False:
        return resoult_list
    else:
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

        # ii = 0
        f_out = open(path_out, 'r', encoding='utf-8')
        lines_out = csv.reader(f_out)
        for line in lines_out:
            # print("数据",line)
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
                    #
                    predict = predict.cpu()
                    predict = predict.numpy()
                    predict = predict[0]
                    res = att_type[predict]
                    aspect_4[4] = res
                    break
            else:
                aspect_4[4] = line[2]

                
            if(line[4] == ""):

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
                    # print("predict*************:",predict)
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
                # print("start predict")
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

            resoult_list = aspect_4[1:]
        

            # f = open(path_result, 'a', encoding='utf-8')
            # writer = csv.writer(f)
            # writer.writerow(aspect_4)

            return resoult_list
        # break




#description = "A flaw was found in the hivex library. This flaw allows an attacker to input a specially crafted Windows Registry (hive) file, which would cause hivex to recursively call the _get_children() function, leading to a stack overflow. The highest threat from this vulnerability is to system availability."
description2 = "ASP.NET Core 2.0 allows an attacker to steal log-in session information such as cookies or authentication tokens via a specially crafted URL aka ""ASP.NET Core Elevation Of Privilege Vulnerability""."
description3 = "Buffer overflow in POP servers based on BSD/Qualcomm's qpopper allows remote attackers to gain root access using a long PASS command."
list = []
list2 = []
list3 = []
# list = get_4aspects(description)
list2 = get_4aspects(description2)
# list3 = get_4aspects(description3)
