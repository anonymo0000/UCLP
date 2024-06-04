import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import re
from gensim.models import Word2Vec
import numpy as np
# ###   att_type   
# path_att_type_label = "/home/user/Programs/PMA_lihang/data/data_att_type_.csv"
# path_v_type_label ="/home/user/Programs/PMA_lihang/data/data_v_type_.csv"
# print("  word2vec  ")
model = Word2Vec.load("module/word2vec.model")

#model.wv.add("<PAD>", np.zeros(300))

print("    ")

# #           -     
def build_vocab():
    word2vec = {}
    i = 0
    #      
    print(len(model.wv.key_to_index))
    #         padding
    word2vec['<PAD>'] = 0
    word2vec['<UNK>'] = 1
    i = 1
    for word in model.wv.key_to_index:
        i += 1
        word2vec[word] = i

    #    
    path_voc = "module/vocab.txt"

    for word in word2vec:
        with open(path_voc, 'a') as f:
            f.write(word + "\t" + str(word2vec[word]) + "\n")
            
        



def modify_str(content):
    content = re.sub("[^\u4e00-\u9fa5^a-z^A-Z^0-9]", " ", str(content))
    #       
    content = re.sub("\s+", " ", content)
    #     
    content = re.sub("\n", " ", content)
    #          
    content = content.strip()
    #print(content)
    content = content.split(" ")
    #            
    content = [word.lower() for word in content]
    if len(content) < 64:
        content = content + ['<PAD>'] * (64 - len(content))
    elif len(content) > 64:
        content = content[:64]
    # res = []
    # for i in range(len(content)):
    #     try:
    #         res.append(model[content[i]])
    #     except:
    #         #           
    #         res.append(np.random.random((300,)))
    return content
##            
import pandas as pd
class MyDataset(Dataset):
    def __init__(self, path):
        # print("     ")
        # path = "module/toTrain.csv"
        
        ff= open(path, 'r', encoding='utf-8')  
        data_label = csv.reader(ff)
        
        # f.close()
       
        
        self.data = []
        self.label = []
        #    
        self.vocab = {}
        with open("module/vocab.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split("\t")
                self.vocab[line[0]] = int(line[1])
        #print(len(self.vocab))
        for i in data_label:
            # print("     ")
            # print(i)
            self.data.append(i[1])
            self.label.append(int(i[2]))
        # print("label***************",self.label)
            #break
        # print(self.data[0])
        # print("       ")
        self.len = len(self.data)
        # print("     ：",self.len)
        # for mmm in self.label:
        #     print(type(mmm))
        # self.data = torch.Tensor(np.array(self.data))
        # self.label = torch.tensor(np.array(self.label))
        # print(self.data.shape)

    def __getitem__(self, index):
        
        data = self.data[index]
        # print("data***************",data)
        label = []
        label.append(self.label[index])
        #print("label***************",type(label))
        label = torch.LongTensor(np.array(label))
        data = modify_str(data)
        data_ = []
        #print("SDCCSDC",self.vocab.get("<UNK>"))
        for i in data:
            data_.append(self.vocab.get(i,1))
        # if len(data_) != 64:
        #     print(data_)
        #     print("error")
        #  long  
        data_ = torch.LongTensor(data_)
        # label = torch.LongTensor(label)
        # print("data_:",data_.shape)
        # print("label:",label.shape)
        return data_, label
    def __len__(self):
        return self.len

# #  
# print("  ")
# mydataset = MyDataset("/home/user/Programs/PMA_lihang/data/aaaatemp.csv")
# print(len(mydataset))
# print(mydataset[0])
# dataloader = DataLoader(mydataset,batch_size=8,shuffle=True)
# print(len(dataloader))
# for data in dataloader:
#     input, label = data
#     print(input.shape)
#     print(label.shape)
#     break
