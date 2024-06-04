
from transformers import RobertaModel, RobertaForSequenceClassification, RobertaTokenizer
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from gensim.models import Word2Vec


# model = Word2Vec.load("/home/user/Programs/PMA_lihang/code_lihang/word2vec/model/word2vec.model")
model = Word2Vec.load("/home/user/lh/PMA_lihang/code_lihang/word2vec/model/word2vec.model")
# 在模型中插入一个单词“<PAD>”，并且设置其词向量为0

# 300维6类 att_type


class PMA(nn.Module):
    def __init__(self):
        super(PMA, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 300, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(300, 1024, 3)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # print("8xshape",x.shape)
        # print(x)
        # 增加一个维度
        # x = x.unsqueeze(0)
        # x = torch.argmax(x)
        # print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x

# 300维13类 v_type


class PMA_1(nn.Module):
    def __init__(self):
        super(PMA_1, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 300, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(300, 1024, 3)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 13)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # print("8xshape",x.shape)
        # print(x)
        # 增加一个维度
        # x = x.unsqueeze(0)
        # x = torch.argmax(x)
        # print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x


class PMA_11kernel(nn.Module):
    def __init__(self):
        super(PMA_11kernel, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 300, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(300, 1024, 1)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 13)
        self.softmax = nn.Softmax(dim=1)
        self.flag = True

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # print("8xshape",x.shape)
        # print(x)
        # 增加一个维度
        # x = x.unsqueeze(0)
        # x = torch.argmax(x)
        # print(x)
        # if(x.shape != torch.Size([4,13])):
        #    x = x.unsqueeze(0)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x


class PMA_1_5kernel(nn.Module):
    def __init__(self):
        super(PMA_1_5kernel, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 300, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(300, 1024, 5)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 13)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # print("8xshape",x.shape)
        # print(x)
        # 增加一个维度
        # x = x.unsqueeze(0)
        # x = torch.argmax(x)
        # print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x


# 128维6类att_type
class PMA_2(nn.Module):
    def __init__(self):
        super(PMA_2, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 128, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(128, 1024, 3)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # print("8xshape",x.shape)
        # #print(x)
        # # 增加一个维度
        # # x = x.unsqueeze(0)
        # #x = torch.argmax(x)
        # print(x.shape)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x


class PMA_2_7(nn.Module):
    def __init__(self):
        super(PMA_2_7, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 128, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(128, 1024, 3)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # print("8xshape",x.shape)
        # #print(x)
        # # 增加一个维度
        # # x = x.unsqueeze(0)
        # #x = torch.argmax(x)
        # print(x.shape)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x


# 300维6类 att_vec
class PMA_3(nn.Module):
    def __init__(self):
        super(PMA_3, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 300, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(300, 1024, 3)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # print("8xshape",x.shape)
        # print(x)
        # 增加一个维度
        # x = x.unsqueeze(0)
        # x = torch.argmax(x)
        # print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x

# 300维11类 att_vec


class PMA_4(nn.Module):
    def __init__(self):
        super(PMA_4, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 300, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(300, 1024, 3)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 13)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # dropout

        # print("8xshape",x.shape)
        # print(x)
        # 增加一个维度
        # x = x.unsqueeze(0)
        # x = torch.argmax(x)
        # print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x


class PMA_4_drop0_5(nn.Module):
    def __init__(self):
        super(PMA_4_drop0_5, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 300, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(300, 1024, 3)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 12)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.dropout(x, 0.5)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # dropout

        # print("8xshape",x.shape)
        # print(x)
        # 增加一个维度
        # x = x.unsqueeze(0)
        # x = torch.argmax(x)
        # print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x


class PMA_4_drop1_5(nn.Module):
    def __init__(self):
        super(PMA_4_drop1_5, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 300, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(300, 1024, 3)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 13)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        x = F.dropout(x, 0.5)
        # dropout

        # print("8xshape",x.shape)
        # print(x)
        # 增加一个维度
        # x = x.unsqueeze(0)
        # x = torch.argmax(x)
        # print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x


# 300维11类 att_vec
class PMA_5(nn.Module):
    def __init__(self):
        super(PMA_5, self).__init__()
        # 词表长度为46795，映射为300维的词向量,pading_idx为0
        self.embedding = nn.Embedding(46795, 300, padding_idx=0)
        # 设置embedding层的padding_idx为300维的全0向量
        # self.embedding.padding_idx = 0
        # 设计一个 一层卷积神经网络，对输入进行6分类，softmax，输入是300维的词向量，输出是6维的向量
        self.conv1 = nn.Conv1d(300, 1024, 3)
        # self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # embedding层
        # print("xshape",x.shape)
        x = self.embedding(x)
        # 输入是300维的词向量，输出是6维的向量
        # print("1xshape",x.shape)
        x = x.transpose(1, 2)
        # print("2xshape",x.shape)
        x = self.conv1(x)
        # print("3xshape",x.shape)
        x = self.relu(x)
        # print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        # print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        # print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # print("8xshape",x.shape)
        # print(x)
        # 增加一个维度
        # x = x.unsqueeze(0)
        # x = torch.argmax(x)
        # print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        # print(x)
        return x


class Text2CNN(nn.Module):
    def __init__(self, vocab_size=46795, embedding_dim=300, num_classes=6):
        super(Text2CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(100, 100, kernel_size=3)
        self.fc = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.max(dim=2)[0]
        x = self.fc(x)
        return x


class Text2CNN_7(nn.Module):
    def __init__(self, vocab_size=46795, embedding_dim=300, num_classes=7):
        super(Text2CNN_7, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(100, 100, kernel_size=3)
        self.fc = nn.Linear(100, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.max(dim=2)[0]
        x = self.fc(x)
        return x
    # def __init__(self, vocab_size=46795, embedding_dim=300, num_classes=13):
    #     super(Text2CNN, self).__init__()
    #     self.embedding = nn.Embedding(vocab_size, embedding_dim)
    #     self.conv1 = nn.Conv1d(embedding_dim, 100, 5)
    #     self.conv2 = nn.Conv1d(100, 100, 5)
    #     self.fc = nn.Linear(100, num_classes)
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     x = self.embedding(x)
    #     x = x.transpose(1, 2)
    #     x = self.conv1(x)
    #     x = self.relu(x)
    #     x = self.conv2(x)
    #     x = self.relu(x)
    #     x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
    #     x = self.fc(x)
    #     return x


class Text1Lstm(nn.Module):
    def __init__(self, vocab_size=46795, embed_dim=300, hidden_dim=100, num_classes=6):
        super(Text1Lstm, self).__init__()
        # create an embedding layer to map words to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # create a lstm layer to process the sequence of vectors
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # create a linear layer to map the final hidden state to the output logits
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len) containing the indices of the words
        # embed the words to get a tensor of shape (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        # pass the embedded sequence to the lstm and get the final hidden state of shape (batch_size, hidden_dim)
        _, (h_n, _) = self.lstm(x)
        # squeeze the first dimension of h_n to get a tensor of shape (batch_size, hidden_dim)
        h_n = h_n.squeeze(0)
        # pass the final hidden state to the linear layer and get the output logits of shape (batch_size, num_classes)
        out = self.linear(h_n)
        return out


class Text1Lstm_7(nn.Module):
    def __init__(self, vocab_size=46795, embed_dim=300, hidden_dim=100, num_classes=7):
        super(Text1Lstm_7, self).__init__()
        # create an embedding layer to map words to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # create a lstm layer to process the sequence of vectors
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # create a linear layer to map the final hidden state to the output logits
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len) containing the indices of the words
        # embed the words to get a tensor of shape (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        # pass the embedded sequence to the lstm and get the final hidden state of shape (batch_size, hidden_dim)
        _, (h_n, _) = self.lstm(x)
        # squeeze the first dimension of h_n to get a tensor of shape (batch_size, hidden_dim)
        h_n = h_n.squeeze(0)
        # pass the final hidden state to the linear layer and get the output logits of shape (batch_size, num_classes)
        out = self.linear(h_n)
        return out


class Text2norLstm(nn.Module):
    def __init__(self, vocab_size=46795, embed_dim=300, hidden_dim=100, num_classes=13):
        super(Text2norLstm, self).__init__()
        # create an embedding layer to map words to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # create a lstm layer to process the sequence of vectors
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        # create a linear layer to map the final hidden state to the output logits
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len) containing the indices of the words
        # embed the words to get a tensor of shape (batch_size, seq_len, embed_dim)
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        # pass the embedded sequence to the lstm and get the final hidden state of shape (batch_size, hidden_dim)
        _, (h_n, _) = self.lstm(x)
        # print(h_n.shape)
        # squeeze the first dimension of h_n to get a tensor of shape (batch_size, hidden_dim)
        h_n = h_n[-1]
        # print(h_n.shape)
        # pass the final hidden state to the linear layer and get the output logits of shape (batch_size, num_classes)
        out = self.linear(h_n)
        # print(out.shape)
        return out


class Text2norLstm_7(nn.Module):
    def __init__(self, vocab_size=46795, embed_dim=300, hidden_dim=100, num_classes=7):
        super(Text2norLstm_7, self).__init__()
        # create an embedding layer to map words to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # create a lstm layer to process the sequence of vectors
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        # create a linear layer to map the final hidden state to the output logits
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len) containing the indices of the words
        # embed the words to get a tensor of shape (batch_size, seq_len, embed_dim)
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)
        # pass the embedded sequence to the lstm and get the final hidden state of shape (batch_size, hidden_dim)
        _, (h_n, _) = self.lstm(x)
        # print(h_n.shape)
        # squeeze the first dimension of h_n to get a tensor of shape (batch_size, hidden_dim)
        h_n = h_n[-1]
        # print(h_n.shape)
        # pass the final hidden state to the linear layer and get the output logits of shape (batch_size, num_classes)
        out = self.linear(h_n)
        # print(out.shape)
        return out

# class Text1Lstm(nn.Module):
#     def __init__(self, vocab_size = 46795, embedding_dim = 300, hidden_dim = 100, num_classes = 13):
#         super(Text1Lstm, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)

#     def forward(self, text):
#         embedded = self.embedding(text)
#         output, (hidden, cell) = self.lstm(embedded)
#         hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
#         return self.fc(hidden.squeeze(0))


class Text2Lstm(nn.Module):
    def __init__(self,  vocab_size=46795, embedding_dim=300, hidden_dim=100, num_classes=6):
        super(Text2Lstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.fc(x)
        return x


class Text2Lstm_7(nn.Module):
    def __init__(self,  vocab_size=46795, embedding_dim=300, hidden_dim=100, num_classes=7):
        super(Text2Lstm_7, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.fc(x)
        return x


class Text1biLstm(nn.Module):
    def __init__(self,  vocab_size=46795, embedding_dim=300, hidden_dim=100, num_classes=6):
        super(Text1biLstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.fc(x)
        return x


class Text1biLstm_7(nn.Module):
    def __init__(self,  vocab_size=46795, embedding_dim=300, hidden_dim=100, num_classes=7):
        super(Text1biLstm_7, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.fc(x)
        return x

# bert的模型


class Text2Transformer(nn.Module):
    def __init__(self, model_name, num_classes=13):
        super(Text2Transformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.transformer.config.hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        # x is a tensor of shape (batch_size, seq_len) containing the indices of the words
        outputs = self.transformer(x, attention_mask=attention_mask)  # Pass attention_mask here
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Using mean pooling over sequence length
        out = self.linear(pooled_output)
        # attention_weights = outputs.attentions

        return out


# class Text2Transformer(nn.Module):
#     def __init__(self, model_name, num_classes=13):
#         super(Text2Transformer, self).__init__()
#         self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
#         self.transformer = RobertaModel.from_pretrained(model_name)
#         self.classifier = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

#     def forward(self, input_ids, attention_mask=None):
#         outputs = self.transformer(input_ids, attention_mask=attention_mask)
#         pooler_output = outputs.pooler_output
#         logits = self.classifier(pooler_output).logits
#         return logits

# roberta的模型
# class Text2Transformer(nn.Module):
#     def __init__(self, model_name, num_classes=13):
#         super(Text2Transformer, self).__init__()
#         self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
#         self.transformer = RobertaModel.from_pretrained(model_name)
#         self.classifier = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         outputs = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         logits = self.classifier(outputs.pooler_output).logits
#         return logits

class Text2norGPT(nn.Module):
    def __init__(self, model_name, num_classes=13):
        super(Text2norGPT, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.transformer.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.transformer(x)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Using mean pooling over sequence length
        out = self.linear(pooled_output)
        return out


class Text2norTransformer2(nn.Module):
    def __init__(self, model_name, num_classes):
        super(Text2norTransformer2, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.transformer.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Using mean pooling over sequence length
        out = self.linear(pooled_output)
        return out


class Text2GPT2(nn.Module):
    def __init__(self, model_name, num_classes=13):
        super(Text2GPT2, self).__init__()
        self.gpt2 = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.linear = nn.Linear(self.gpt2.config.hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        # x is a tensor of shape (batch_size, seq_len) containing the indices of the words
        outputs = self.gpt2(x, attention_mask=attention_mask)  # Pass attention_mask here
        pooled_output = outputs.logits  # GPT-2 models use 'logits' for classification
        out = self.linear(pooled_output)
        return out


class Text2Transformer3(nn.Module):
    def __init__(self, model_name, num_classes=13):
        super(Text2Transformer3, self).__init__()
        self.transformer = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        # self.linear = nn.Linear(self.transformer.config.hidden_size, num_classes) # No need for this

    def forward(self, x, attention_mask=None):
        # x is a tensor of shape (batch_size, seq_len) containing the indices of the words
        outputs = self.transformer(x, attention_mask=attention_mask,
                                   token_type_ids=None)  # Pass token_type_ids=None here
        # pooled_output = outputs.last_hidden_state.mean(dim=1)  # No need for this
        out = outputs.logits  # Use logits instead of mean pooling
        return out
