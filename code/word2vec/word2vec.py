#1         
import gensim
import logging
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load word2vec model (trained on an enormous Google corpus)
model = gensim.models.KeyedVectors.load_word2vec_format('/home/user/Programs/PMA_lihang/code_lihang/word2vec/model/GoogleNews-vectors-negative300.bin', binary = True) 

path_data = '/home/user/Programs/PMA_lihang/data/pt1_file2.csv'

#         
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_vectorOOV1(s):
  try:
    return np.array(model.get_vector(s))
  except KeyError:
    return np.random.random((300,))

def get_aspect_vec1(words):
    vector = []
    for word in words:
        word_vector = get_vectorOOV1(word)
        vector.append(word_vector)
    # print(len(vector))
    tensor = torch.Tensor(np.array(vector))
    return tensor    

f = open(path_data,'r',encoding='utf-8')
vec = []
lines = csv.reader(f)
for line in lines:
    print(len(line))
    words = line[1].split(" ")#product embedding
    # print(words)
    embedding = get_aspect_vec1(words)
    print(embedding.shape)
    break


