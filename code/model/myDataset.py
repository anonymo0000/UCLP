from transformers import GPT2Tokenizer
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import re
from gensim.models import Word2Vec
import numpy as np
# 构建att_type数据集
path_att_type_label = "/home/user/Programs/PMA_lihang/data/data_att_type_.csv"
path_v_type_label = "/home/user/Programs/PMA_lihang/data/data_v_type_.csv"
print("加载word2vec模型")
# model = Word2Vec.load("/home/user/Programs/PMA_lihang/code_lihang/word2vec/model/word2vec.model")
model = Word2Vec.load("/home/user/lh/PMA_lihang/code_lihang/word2vec/model/word2vec.model")
# model.wv.add("<PAD>", np.zeros(300))


# #从预训练模型中构建词语-词向量字典
def build_vocab():
    word2vec = {}
    i = 0
    # 输出词表长度
    print(len(model.wv.key_to_index))
    # 在字典前面添加一个padding
    word2vec['<PAD>'] = 0
    word2vec['<UNK>'] = 1
    i = 1
    for word in model.wv.key_to_index:
        i += 1
        word2vec[word] = i

    # 写入文件
    path_voc = "/home/user/Programs/PMA_lihang/code_lihang/model/vocab.txt"

    for word in word2vec:
        with open(path_voc, 'a') as f:
            f.write(word + "\t" + str(word2vec[word]) + "\n")


def modify_str(content):
    content = re.sub("[^\u4e00-\u9fa5^a-z^A-Z^0-9]", " ", str(content))
    # 去除连续的空格
    content = re.sub("\s+", " ", content)
    # 删除回车符
    content = re.sub("\n", " ", content)
    # 删除开头和结尾的空格
    content = content.strip()
    # print(content)
    content = content.split(" ")
    # 将所有大写字母转换为小写
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
    #         #使用随机的词向量来代替
    #         res.append(np.random.random((300,)))
    return content
# 直接返回一个词嵌入的向量


class MyDataset(Dataset):
    def __init__(self, path):
        # print("加载数据集")
        f = open(path, 'r', encoding='utf-8')
        data_label = csv.reader(f)

        self.data = []
        self.label = []
        # 读取字典
        self.vocab = {}
        with open("/home/user/Programs/PMA_lihang/code_lihang/model/vocab.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split("\t")
                self.vocab[line[0]] = int(line[1])
        # print(len(self.vocab))
        for i in data_label:
            self.data.append(i[1])
            self.label.append(int(i[2]))
        # print("label***************",self.label)
            # break
        # print(self.data[0])
        # print("数据集加载完成")
        self.len = len(self.data)
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
        # print("label***************",type(label))
        label = torch.LongTensor(np.array(label))
        data = modify_str(data)
        data_ = []
        # print("SDCCSDC",self.vocab.get("<UNK>"))
        for i in data:
            data_.append(self.vocab.get(i, 1))
        # if len(data_) != 64:
        #     print(data_)
        #     print("error")
        # 转成long类型
        data_ = torch.LongTensor(data_)
        # label = torch.LongTensor(label)
        # print("data_:",data_.shape)
        # print("label:",label.shape)
        return data_, label

    def __len__(self):
        return self.len


class MyDatasetForTransformer(Dataset):
    def __init__(self, path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = []
        self.label = []
        with open(path, 'r', encoding='utf-8') as f:
            data_label = csv.reader(f)
            for i in data_label:
                self.data.append(i[1])
                self.label.append(int(i[2]))
        self.len = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        inputs = self.tokenizer(data, padding='max_length', truncation=True,
                                max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label)
        }

    def __len__(self):
        return self.len


# 数据构建类
class MyDatasetForGPT2(Dataset):
    def __init__(self, path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = []
        self.label = []

        with open(path, 'r', encoding='utf-8') as f:
            data_label = csv.reader(f)
            for i in data_label:
                text = f"Classify this text: {i[1]}"  # Add a classification prompt
                self.data.append(text)
                self.label.append(int(i[2]))

        self.len = len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        inputs = self.tokenizer(data, padding='max_length', truncation=True,
                                max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label)
        }
    
    def __len__(self):
        return self.len




# class MyDatasetForGPT2(Dataset):
#     def __init__(self, path, tokenizer, max_length=128):
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#         self.data = []
#         self.label = []
#         with open(path, 'r', encoding='utf-8') as f:
#             data_label = csv.reader(f)
#             for i in data_label:
#                 self.data.append(i[1])
#                 self.label.append(int(i[2]))
#         self.len = len(self.data)

#     def __getitem__(self, index):
#         data = self.data[index]
#         label = self.label[index]

#         # Tokenize the text using GPT-2 style (no 'input_ids' and 'attention_mask' for GPT-2)
#         inputs = self.tokenizer(data, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

#         return {
#             'input_ids': inputs['input_ids'].squeeze(),
#             'label': torch.tensor(label)
#         }

#     def __len__(self):
#         return self.len


# #测试
# print("测试")
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
