import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class V_type_classifer(nn.Module):
    def __init__(self):
        super(V_type_classifer,self).__init__()
        path = "/home/user/Programs/PMA_lihang/mypaper/module/bert-base-uncased/pytorch_model.bin"
        #  bert  
        self.bert = BertModel.from_pretrained(path)
        #  bert       embedding 
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        


