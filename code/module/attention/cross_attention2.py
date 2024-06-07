import torch
import torch.nn as nn
from transformers import BertModel

class TextMatchModel(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese') # use BERT to encode text sequences
        self.cross_attention = CrossAttention(dim, heads) # use CrossAttention to compute similarity
        self.fc = nn.Linear(dim, 1) # use a fully connected layer to predict match score

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        # encode text sequences using BERT
        output1 = self.bert(input_ids1, attention_mask1)
        output2 = self.bert(input_ids2, attention_mask2)
        
        # get the last hidden states of BERT
        x1 = output1.last_hidden_state # shape (batch_size, seq_len1, dim)
        x2 = output2.last_hidden_state # shape (batch_size, seq_len2, dim)

        # get the global features by averaging over the sequence dimension
        x1 = x1.mean(dim=1) # shape (batch_size, dim)
        x2 = x2.mean(dim=1) # shape (batch_size, dim)

        # compute cross attention between text sequences
        x = self.cross_attention(x1.unsqueeze(0),x2.unsqueeze(0)) 
            # shape (batch_size,h,dim/h), where h is number of heads


#定义一个交叉注意力机制，计算两个文本序列的交叉注意力
