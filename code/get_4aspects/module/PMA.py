
import torch.nn as nn
import torch.nn.functional as F

from gensim.models import Word2Vec




model = Word2Vec.load("module/word2vec.model")
#          “<PAD>”，         0

#300 6  att_type
class PMA(nn.Module):
    def __init__(self):
        super(PMA,self).__init__()
        #     46795，   300     ,pading_idx 0
        self.embedding = nn.Embedding(46795,300,padding_idx=0)
        #  embedding  padding_idx 300   0  
        #self.embedding.padding_idx = 0
        #             ，     6  ，softmax，   300     ，   6    
        self.conv1 = nn.Conv1d(300,1024,3)
        #self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024,6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        #embedding 
        # print("xshape",x.shape)
        x = self.embedding(x)
        #   300     ，   6    
        #print("1xshape",x.shape)
        x = x.transpose(1,2)
        #print("2xshape",x.shape)
        x = self.conv1(x)
        #print("3xshape",x.shape)
        x = self.relu(x)
        #print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        #print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        #print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        #print("8xshape",x.shape)
        #print(x)
        #      
        #x = x.unsqueeze(0)
        #x = torch.argmax(x)
        #print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        #print(x)
        return x

#300 13  v_type
class PMA_1(nn.Module):
    def __init__(self):
        super(PMA_1,self).__init__()
        #     46795，   300     ,pading_idx 0
        self.embedding = nn.Embedding(46795,300,padding_idx=0)
        #  embedding  padding_idx 300   0  
        #self.embedding.padding_idx = 0
        #             ，     6  ，softmax，   300     ，   6    
        self.conv1 = nn.Conv1d(300,1024,3)
        #self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024,13)
        self.softmax = nn.Softmax(dim=1)





    
    def forward(self,x):
        #embedding 
        # print("xshape",x.shape)
        x = self.embedding(x)
        #   300     ，   6    
        #print("1xshape",x.shape)
        x = x.transpose(1,2)
        #print("2xshape",x.shape)
        x = self.conv1(x)
        #print("3xshape",x.shape)
        x = self.relu(x)
        #print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        #print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        #print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        #print("8xshape",x.shape)
        #print(x)
        #      
        x = x.unsqueeze(0)
        #x = torch.argmax(x)
        #print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        #print(x)
        return x

class PMA_11kernel(nn.Module):
    def __init__(self):
        super(PMA_11kernel,self).__init__()
        #     46795，   300     ,pading_idx 0
        self.embedding = nn.Embedding(46795,300,padding_idx=0)
        #  embedding  padding_idx 300   0  
        #self.embedding.padding_idx = 0
        #             ，     6  ，softmax，   300     ，   6    
        self.conv1 = nn.Conv1d(300,1024,1)
        #self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024,13)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        #embedding 
        # print("xshape",x.shape)
        x = self.embedding(x)
        #   300     ，   6    
        #print("1xshape",x.shape)
        x = x.transpose(1,2)
        #print("2xshape",x.shape)
        x = self.conv1(x)
        #print("3xshape",x.shape)
        x = self.relu(x)
        #print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        #print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        #print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        #print("8xshape",x.shape)
        #print(x)
        #      
        #x = x.unsqueeze(0)
        #x = torch.argmax(x)
        #print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        #print(x)
        return x

class PMA_1_5kernel(nn.Module):
    def __init__(self):
        super(PMA_1_5kernel,self).__init__()
        #     46795，   300     ,pading_idx 0
        self.embedding = nn.Embedding(46795,300,padding_idx=0)
        #  embedding  padding_idx 300   0  
        #self.embedding.padding_idx = 0
        #             ，     6  ，softmax，   300     ，   6    
        self.conv1 = nn.Conv1d(300,1024,5)
        #self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024,13)
        self.softmax = nn.Softmax(dim=1)





    
    def forward(self,x):
        #embedding 
        # print("xshape",x.shape)
        x = self.embedding(x)
        #   300     ，   6    
        #print("1xshape",x.shape)
        x = x.transpose(1,2)
        #print("2xshape",x.shape)
        x = self.conv1(x)
        #print("3xshape",x.shape)
        x = self.relu(x)
        #print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        #print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        #print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        #print("8xshape",x.shape)
        #print(x)
        #      
        #x = x.unsqueeze(0)
        #x = torch.argmax(x)
        #print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        #print(x)
        return x


#128 6 att_type
class PMA_2(nn.Module):
    def __init__(self):
        super(PMA_2,self).__init__()
        #     46795，   300     ,pading_idx 0
        self.embedding = nn.Embedding(46795,128,padding_idx=0)
        #  embedding  padding_idx 300   0  
        #self.embedding.padding_idx = 0
        #             ，     6  ，softmax，   300     ，   6    
        self.conv1 = nn.Conv1d(128,1024,3)
        #self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024,6)
        self.softmax = nn.Softmax(dim=1)





    
    def forward(self,x):
        #embedding 
        # print("xshape",x.shape)
        x = self.embedding(x)
        #   300     ，   6    
        #print("1xshape",x.shape)
        x = x.transpose(1,2)
        #print("2xshape",x.shape)
        x = self.conv1(x)
        #print("3xshape",x.shape)
        x = self.relu(x)
        #print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        #print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        #print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        # print("8xshape",x.shape)
        #print(x)
        #       
        x = x.unsqueeze(0)
        #x = torch.argmax(x)
        #print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        #print(x)
        return x



#300 6  att_vec
class PMA_3(nn.Module):
    def __init__(self):
        super(PMA_3,self).__init__()
        #     46795，   300     ,pading_idx 0
        self.embedding = nn.Embedding(46795,300,padding_idx=0)
        #  embedding  padding_idx 300   0  
        #self.embedding.padding_idx = 0
        #             ，     6  ，softmax，   300     ，   6    
        self.conv1 = nn.Conv1d(300,1024,3)
        #self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024,6)
        self.softmax = nn.Softmax(dim=1)





    
    def forward(self,x):
        #embedding 
        # print("xshape",x.shape)
        x = self.embedding(x)
        #   300     ，   6    
        #print("1xshape",x.shape)
        x = x.transpose(1,2)
        #print("2xshape",x.shape)
        x = self.conv1(x)
        #print("3xshape",x.shape)
        x = self.relu(x)
        #print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        #print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        #print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        #print("8xshape",x.shape)
        #print(x)
        #      
        x = x.unsqueeze(0)
        #x = torch.argmax(x)
        #print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        #print(x)
        return x
    
#300 11  att_vec
class PMA_4(nn.Module):
    def __init__(self):
        super(PMA_4,self).__init__()
        #     46795，   300     ,pading_idx 0
        self.embedding = nn.Embedding(46795,300,padding_idx=0)
        #  embedding  padding_idx 300   0  
        #self.embedding.padding_idx = 0
        #             ，     6  ，softmax，   300     ，   6    
        self.conv1 = nn.Conv1d(300,1024,3)
        #self.conv2 = nn.Conv1d(1024,300,3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024,11)
        self.softmax = nn.Softmax(dim=1)





    
    def forward(self,x):
        #embedding 
        # print("xshape",x.shape)
        x = self.embedding(x)
        #   300     ，   6    
        #print("1xshape",x.shape)
        x = x.transpose(1,2)
        #print("2xshape",x.shape)
        x = self.conv1(x)
        #print("3xshape",x.shape)
        x = self.relu(x)
        #print("4xshape",x.shape)
        # x = self.conv2(x)
        # #print("5xshape",x.shape)
        # x = self.relu(x)
        #print("6xshape",x.shape)
        x = F.max_pool1d(x, x.size(2))
        #print("7xshape",x.shape)
        x = self.fc(x.squeeze())
        #print("8xshape",x.shape)
        #print(x)
        #      
        x = x.unsqueeze(0)
        #x = torch.argmax(x)
        #print(x)
        x = self.softmax(x)
        # print("9xshape",x.shape)
        #print(x)
        return x