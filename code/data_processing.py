#       ，        
#  tqdm 
from tqdm import tqdm
path = "/home/user/Programs/PMA_lihang/data/allitems.csv"
path1 = "/home/user/Programs/PMA_lihang/data/allitems_v2.csv"

# i = 0
# data = []
# with open(path, 'r', encoding = 'utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         # i+=1
#         #               
#         if line.find('** RESERVED ** This candidate has been reserved by') != -1:
#             i+=1
#             continue
#         data.append(line)
#     #   data          
#     print(i)
#     with open(path1, 'w', encoding = 'utf-8') as f1:
#         f1.writelines(data)
path2 = "/home/user/Programs/PMA_lihang/data/allitems_v3.csv"
# i = 0
# data = []   
# with open(path1, 'r', encoding = 'utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         # i+=1
#         #               
#         if line.find('** REJECT ** DO NOT USE THIS CANDIDATE NUMBER.') != -1:
#             i+=1
#             continue
#         data.append(line)
#     #   data          
#     print(i)
#     with open(path2, 'w', encoding = 'utf-8') as f1:
#         f1.writelines(data)
    
path3 = "/home/user/Programs/PMA_lihang/data/allitems_v4.csv"
# i = 0
# data = []   
# with open(path2, 'r', encoding = 'utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         # i+=1
#         #               
#         if line.find('DO NOT USE THIS CANDIDATE NUMBER.') != -1:
#             i+=1
#             continue
#         data.append(line)
#     #   data          
#     print(i)
#     with open(path3, 'w', encoding = 'utf-8') as f1:
#         f1.writelines(data)        

#     
# data = []   
# m = 0
path4 = "/home/user/Programs/PMA_lihang/data/allitems_v5.csv"    
# with open(path3, 'r', encoding = 'utf-8') as f:
#     lines = f.readlines()
#     for line in tqdm(lines):
#         content = ''
#         #               
#         for i in range(len(line)):
#             if line[i] == ',':
#                 break
#             content += line[i]
        
#         content += '\t##=divide=##\t'
#         #     "ab" "kjl"     
#         for i in range(len(line)):
#             if line[i] == ',' and line[i+1] == '"':
#                 break
#         for j in range(i, len(line)):
#             if line[j] == '"' and line[j+1] == ',' and line[j+2] == '"':
#                 break
#         for k in range(i+2, j):
#             content += line[k]
#         content += '\n'
        
#         data.append(content)
        
# with open(path4, 'w', encoding = 'utf-8') as f1:
#     f1.writelines(data)

#      jupyter notebook     
