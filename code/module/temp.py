#          
path = '/home/user/Programs/PMA_lihang/mypaper/data/data_context_v_type.csv'
path1 = "/home/user/Programs/PMA_lihang/mypaper/data/data_context_v_type_.csv"
import csv

f = open(path, 'r', encoding='utf-8')
f1 = open(path1, 'a', encoding='utf-8')

f_reader = csv.reader(f)
for line in f_reader:
    data = []
    data.append(line[0])
    data.append(line[1])
    head_tail = line[2].split(line[1])
    description = ""
    for i in range(len(head_tail)):
        description += head_tail[i]
    data.append(description)
    #    
    writer = csv.writer(f1)
    writer.writerow(data)


