
import re
import csv
import random
from tqdm import tqdm

# file1 = r'/home/user/Programs/PMA_lihang/data/allitems_v5.csv'#'/home/user/Programs/PMA_lihang/data/allitems_v5.csv'#CVE
# file2 = r'/home/user/Programs/PMA_lihang/data/pt1_file2_1.csv' # in the following parteen
# file3 = r'/home/user/Programs/PMA_lihang/data/pt1_file3.csv' # not in >>>

def devide1(f_in, f_out, f_next):# f_in:    ，f_out:    ，f_next:       

    file1 = f_in
    file2 = f_out
    file3 = f_next

    # f_temp = open(file3, 'w', encoding='utf-8')
    # f_temp.truncate()
    # f_temp.close()



    f = open(file1, "r",encoding='utf-8')
    f2 = open(file2, 'a', encoding='utf-8')
    f3 = open(file3, 'a', encoding='utf-8')

    flag = 0
    #        vt     
    for row in csv.reader(f):
    #                       ，       ，
    #            ，       ，         ，      。
        # print(row[0])
        # break
        head = row[0].split('\t##=divide=##\t')[0]
        row = row[0].split('\t##=divide=##\t')[1]
        # print(head)
        # print(row)
        # break
        if len(re.findall('allow[^ ]* .*? to ', row)):#  allow  # 1.       allow to  ['allowed  acd to ']
            # print("$")
            # break
            aat = re.findall('allow[^ ]* (.*?) to ', row)[0]#0.       
            pd = row.split(re.findall('allow[^ ]* .*? to ', row)[0])[0]
            a1 = row.split(re.findall('allow[^ ]* .*? to ', row)[0])[1]
            rootc = ''
            #          ，  pd    " in "     ，   " in "     
            if len(re.findall('(.*?) in ',pd)) != 0:
                vt = re.findall('(.*?) in ',pd)[0]
                vt = str(vt).replace('[','').replace(']','')
            #  vt  []
            else:
                vt = ''
            if len(re.findall('(?:(?: via .*)|(?: by [a-z]+ing .*)|(?: uses? .*))', a1)):   # via    ，    2.        via,by doing,uses，  3           ，    
                at = re.findall('(?:(?: via .*)|(?: by [a-z]+ing .*)|(?: uses? .*))', a1)[0]#  ：    through/by something/used  
                                                                                            #  '?:'      ，    '('  ，    
                im = a1.replace(at, '')
            else:#   if        
                im = a1
                at = ''
            if len(re.findall('(?:(?:(?:(?:uffer)|(?:teger)) overflow)|(?:vulnerability?(?:ies)?)) in (.*) allow', pd,#3. pd   overflow vulnerability
                            re.DOTALL)):#re.DOTALL '.'     '\n'，      
                
                

                
                # vt = re.findall('((?:(?:(?:uffer)|(?:teger)) overflow)|(?:vulnerability?(?:ies)?)) in (?:.*) allow', pd,
                #                 re.DOTALL)[0]#     (?:(?:uffer)|(?:teger)) overflow)|(?:vulnerability?(?:ies)?
                pd = re.findall('(?:(?:(?:(?:uffer)|(?:teger)) overflow)|(?:vulnerability?(?:ies)?)) in (.*) allow', pd,
                                re.DOTALL)[0]#     (.*)
            if len(re.findall(
                    '(?:(?: fails? to )|(?: automatically )|(?: when )|(?: have )|(?: has )|(?: does )|(?: do )|(?: uses? )|(?: returns? )|(?: creates? )|(?: provides? )|(?: relies? )|(?: places? )|(?: generates? )|(?: advertises )|(?: mishandles )|(?: store SSH )|(?: lets? )|(?: handles )|(?: using )|(?: lets? )).*',
                    pd)):#4.         ，    rootcause
                rootc = re.findall(
                    '(?:(?: fails? to )|(?: automatically )|(?: when )|(?: have )|(?: has )|(?: does )|(?: do )|(?: uses? )|(?: returns? )|(?: creates? )|(?: provides? )|(?: relies? )|(?: places? )|(?: generates? )|(?: advertises )|(?: mishandles )|(?: store SSH )|(?: lets? )|(?: handles )|(?: using )|(?: lets? )).*',
                    pd)[0]
                pd = pd.split(rootc)[0]
            if at == '':
                    if len(re.findall(
                            '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct )|(?:compromise )).*? to',
                            im)):#5.       ， to  ，      ，  impact attacker vector
                        at = re.findall(
                            '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct )|(?:compromise )).*? to',
                            im)[0]#send     to  
                        im = im.split(at)[-1]#  
                                            #  ：cause  ，lead  ，       
            dt = []
            if " in " in pd:
                pd = pd.split(" in ")[1]
            if len(vt) == 1:
                vt = ''
            dt.append(head)
            dt.append(pd)#product     
            dt.append(aat)#attacker type    
            dt.append(im)#impact
            dt.append(at)#attacker vector
            dt.append(rootc)#rootcause 
            dt.append(vt)#vulnerability type
            writer = csv.writer(f2)
            writer.writerow(dt)

            # flag += 1
            # if flag == 10:
            #     break
        else:#   allow 
            dt = []
            # str = ""
            # str += head + "\t##=divide=##\t"
            dt.append(head)
            # str += row + "\n"
            #dt.append(":=:")
            dt.append(row)
            # f3.writelines(str)
            writer = csv.writer(f3)
            writer.writerow(dt)

            # flag += 1
            # if flag == 10:
            #     break
