import re
import csv
from tqdm import tqdm
# #         pt1_file3    
# file3 = r'/home/user/Programs/PMA_lihang/data/pt1_file3.csv'#CVE
# file4 = r'/home/user/Programs/PMA_lihang/data/pt2_file4.csv' # in the following parteen
# file5 = r'/home/user/Programs/PMA_lihang/data/pt2_file5.csv' # not in >>>
#        rootcause     

def devide2(f_in, f_out, f_next):

    file3 = f_in
    file4 = f_out
    file5 = f_next

    # f_temp = open(file5, 'w', encoding='utf-8')
    # f_temp.truncate()
    # f_temp.close()

    f3 = open(file3,encoding='utf-8')
    for row in tqdm(csv.reader(f3)):
    # try:
        head = row[0]
        # row = row[1]
        # #print(row)#print(
        # #print(head)
        # break
        vt=''
        pd=''
        im=''
        at=''
        aat = ''
        does=''
        pdd=''
        a1 = row[1]#  

        if  len(re.findall('\AAn? [^,]*? (?:(?:in)|(?:exists? when ))', a1)):
            #print('1'*20)
            #print(row[1])
            vt = re.findall('\A(An? [^,]*?) (?:(?:in)|(?:exists? when )).*?(?:(?:\. )|(?:\Z))', a1)[0] # 
            pd = re.findall('\AAn? [^,]*? (?:(?:in)|(?:exists? when ))(.*?)(?:(?:\. )|(?:\Z))', a1)[0]
            pd = pd.split('. ')[0]
            # if len(re.findall('when .*?, ',pd)):
            #     does =re.findall('(when .*?) ',pd)[0]
            #     pd=pd.split(does)[0]
            # if len(re.findall('due to .*?(?:,|\.) ',a1)):
            #     does =re.findall('due to (.*?)(?:,|\.) ',a1)[0]
            if len(re.findall(' A .*? could (?:potentially )?exploit this by',row[1])):
                aat=re.findall(' (A .*?) could (?:potentially )?exploit this by',a1)[0]
                at=re.findall(' A .*? could (?:potentially )?exploit this (by .*?)\. ',a1)
                if(len(at)):
                    at=at[0]
                else:
                    at=''
                if len(re.findall('by .*? to ',at)):
                    im=re.findall('by .*? to (.*)',at)[0]
                    at=at.split(im)[0]

            if len(re.findall('is due to .*?\. ',pd)):
                does =re.findall('is due to (.*?)\. ',pd)[0]
            elif len(re.findall('because of .*?\. ',pd)):
                does =re.findall('because of (.*?)\. ',pd)[0]
            elif len(re.findall('(?:(?: improperly )|(?: fails? )|(?: automatically )|(?: does )|(?: do )|(?: uses? )|(?: returns? )|(?: creates? )|(?: provides? )|(?: relies? )|(?: places? )|(?: generates? )|(?: advertises )|(?: mishandles )|(?: store SSH )|(?: lets? )|(?: handles )|(?: using )|(?: lets? )).*', pd)):
                does = re.findall('(?:(?: improperly )|(?: fails? )|(?: automatically )|(?: does )|(?: do )|(?: uses? )|(?: returns? )|(?: creates? )|(?: provides? )|(?: relies? )|(?: places? )|(?: generates? )|(?: advertises )|(?: mishandles )|(?: store SSH )|(?: lets? )|(?: handles )|(?: using )|(?: lets? )).*', pd)[0]
                pd = pd.split(does)[0]
            if len(re.findall('a .{0,20}(?:(?:attacker? )|(?:users? ))',row[1],re.IGNORECASE)):
                aat = re.findall('a .{0,20}(?:(?:attacker? )|(?:users? ))',row[1],re.IGNORECASE)[0]

            elif len(re.findall('\. .{0,20}(?:(?:attacker? )|(?:users? ))',row[1],re.IGNORECASE)):
                aat = re.findall('\. (.{0,20}(?:(?:attacker? )|(?:users? )))',row[1],re.IGNORECASE)[0]

            if len(re.findall('a .*? vulnera',row[1],re.IGNORECASE)):
                vt = re.findall('a (.*?) vulnera',row[1],re.IGNORECASE)[0]
            if len(re.findall('a .*? overflow',row[1],re.IGNORECASE)):
                vt = re.findall('a (.*? overflow)',row[1],re.IGNORECASE)[0]


            if len(re.findall(' result in .*?(?:(?:, )|(?:\. )|(?:\Z))',row[1])):
                im = re.findall(' (result in .*?)(?:(?:, )|(?:\. )|(?:\Z))',row[1])[0]
            elif len(re.findall(' leads?(?:ing)? to .*?(?:(?:, )|(?:\. )|(?:\Z))',row[1])):
                im = re.findall(' (leads?(?:ing)? to.*?)(?:(?:, )|(?:\. )|(?:\Z))',row[1])[0]
            elif len(re.findall(' able to .*?(?:(?:, )|(?:\. )|(?:\Z))', row[1])):
                im = re.findall(' able to (.*?)(?:(?:, )|(?:\. )|(?:\Z))', row[1])[0]
            elif len(re.findall(' enables .{0,20} to .*?(?:(?:, )|(?:\. )|(?:\Z))', row[1])):
                im = re.findall(' enables .{0,20} to(.*?)(?:(?:, )|(?:\. )|(?:\Z))', row[1])[0]
                if at == '':
                    if len(re.findall('by [a-z]+ing ', im)):
                        at = re.findall('by [a-z]+ing .*', im)[0]
                        im = im.split(at)[0]
            elif len(re.findall(' could [^be] .*?(?:(?:, )|(?:\. )|(?:\Z))', row[1])):
                im = re.findall(' could ([^be].*?)(?:(?:, )|(?:\. )|(?:\Z))', row[1])[0]
            elif len(re.findall('(?:(?:can )|(?:could )|(?:may ))causes? .*?(?:(?:, )|(?:\. )|(?:\Z))', row[1])):
                im = re.findall('(?:(?:can )|(?:could )|(?:may ))(causes? .*?)(?:(?:, )|(?:\. )|(?:\Z))', row[1])[0]
            elif len(re.findall('(?:(?:can )|(?:could )|(?:may )) allows? .*?(?:(?:, )|(?:\. )|(?:\Z))', row[1])):
                im = re.findall('(?:(?:can )|(?:could )|(?:may )) (allows? .*?)(?:(?:, )|(?:\. )|(?:\Z))', row[1])[0]
            if len(re.findall('\.A [^\.]*? can causes?',a1)):
                im =re.findall('\.A [^\.]*? can (causes?.*)',a1)[0]
                aat=re.findall('\.(A [^\.]*?) can',a1)[0]
            if len(re.findall('a .*?(?:(?:attacker? )|(?:users? ))(?:(?:can )|(?:could )|(?:may ))(?:(?:uses? )|(?:sends? )|(?:supplys? )|(?:speciallys? )|(?:uploads? )|(?:creates? )|(?:constructs? )|(?:compromises? )).*? to (?:(?:, )|(?:\. )|(?:\Z))',row[1],re.IGNORECASE)):
                if at=='':
                    at =re.findall('a .*?(?:(?:attacker? )|(?:users? ))(?:(?:can )|(?:could )|(?:may ))((?:(?:uses? )|(?:sends? )|(?:supplys? )|(?:speciallys? )|(?:uploads? )|(?:creates? )|(?:constructs? )|(?:compromises? )).*?) to (?:(?:, )|(?:\. )|(?:\Z))',row[1],re.IGNORECASE)[0]
            if does!='':
                pd = pd.split(does)[0]

            if len(re.findall(' Product:',row[1])):
                pdd=re.findall(' Product:(.*)',row[1])[0]
            elif len(re.findall('.*\. .* (?:(?:is )|(?:are ))affected.',row[1])):
                pdd=re.findall('.*\. (.*) (?:(?:is )|(?:are ))affected.',row[1])[0]
            elif len(re.findall('This [^ ]* affected ',row[1])):
                
                pdd=re.findall('This [^ ]* affected (.*?)(?:(?:, )|(?:\. )|(?:\Z))',row[1])
                if(len(pdd)):
                    pdd=pdd[0]
                else:
                    pdd=''

            elif len(re.findall('is affected by ',row[1])):
                pdd=re.findall('is affected by (.*?)(?:(?:, )|(?:\. )|(?:\Z))',row[1])[0]
            if len(re.findall(' could enable an? .*? to ',row[1])):
                aat = re.findall(' could enable an? (.*?) to',row[1])[0]
                # print('1'*20)
                # print(row)
                #print(row[1])
                im = re.findall(' could enable an? .*? to (.*?)(?:(?:, )|(?:\. )|(?:\Z))',row[1])
                if(len(im)):
                    im=im[0]
                else:
                    im=''
                pd =pd.split('could enable a')[0]
                does=''
            if len(re.findall('\Aissue .*as',vt)):
                vt=''
            if vt =='':
                if len(re.findall(' There is .*? in ', a1)):
                    vt = re.findall(' There is (.*?) in ', a1)[0]
            if vt =='A flaw was found':
                vt =''
            if len(re.findall('A vulnerability',vt)):
                vt=''
            if len(re.findall('An error',vt)):
                vt=''
            if len(re.findall('An issue',vt)):
                vt=''
            av=at
            if av == '':
                if len(im.split(' via ')) > 1:
                    av = 'via '+im.split(' via ')[-1]
                    im = im.split(' via ')[0]
            if av == '':
                if len(re.findall('by [a-z]+ing ',row[1])):
                    av = re.findall('by [a-z]+ing .*',a1)[0]
                    im = im.split(av)[0]

            dt = []
            pd=pd.split('. ')[0]
            if pdd!='':
                pd+='***'+pdd
            dt.append(row[0])
            dt.append(pd)#  
            dt.append(aat)#     
            dt.append(im)#  
            dt.append(av)#    
            dt.append(does)
            dt.append(vt)#    
            # dt.append(pdd)
            f4 = open(file4, 'a', newline='',encoding='utf-8')
            writer = csv.writer(f4)
            writer.writerow(dt)
        # elif len(re.findall('(?: allows? )|(?: has )', a1)):
        #     #print('2' * 20)
        #     im = re.findall('(?:(?: allows? )|(?: has )).*', a1)[0]
        #     pd = a1.replace(im, '')
        #     if len(re.findall('due to .*?(?:,|\.) ', a1)):
        #         does = re.findall('due to (.*?)(?:,|\.) ', a1)[0]

        elif len(re.findall(', which makes it easier for.*? to ', a1)):
            #print('2' * 20)
            aat = re.findall(', which makes it easier for (.*?) to ', a1)[0]
            im = re.findall(', which makes it easier for (?:.*?) to (.*)', a1)[0]
            a2 = a1.split(', which makes it easier for')[0]
            pd=a2
            if len(re.findall('(?:(?: when )|(?: improperly )|(?: fails? )|(?: automatically )|(?: does )|(?: do )|(?: uses? )|(?: returns? )|(?: creates? )|(?: provides? )|(?: relies? )|(?: places? )|(?: generates? )|(?: advertises )|(?: mishandles )|(?: store SSH )|(?: lets? )|(?: handles )|(?: using )|(?: lets? )).*', pd)):
                does = re.findall('(?:(?: when )|(?: improperly )|(?: fails? )|(?: automatically )|(?: does )|(?: do )|(?: uses? )|(?: returns? )|(?: creates? )|(?: provides? )|(?: relies? )|(?: places? )|(?: generates? )|(?: advertises )|(?: mishandles )|(?: store SSH )|(?: lets? )|(?: handles )|(?: using )|(?: lets? )).*', pd)[0]
                pd = pd.split(does)[0]
            if at == '':
                if len(im.split(' via ')) > 1:
                    at = 'via '+im.split(' via ')[-1]
                    im = im.split(' via ')[0]
            dt = []
            dt.append(row[0])
            dt.append(pd)
            dt.append(aat)
            dt.append(im)
            dt.append(at)
            dt.append(does)
            dt.append(vt)
            dt.append(pdd)
            f4 = open(file4, 'a', newline='',encoding='utf-8')
            writer = csv.writer(f4)
            writer.writerow(dt)
            # if len(re.findall('due to .*?(?:,|\.) ', a1)):
            #     does = re.findall('due to (.*?)(?:,|\.) ', a1)[0]
        elif len(re.findall(' Successful exploitation could lead to ',row[1])):
            #print('4' * 20)
            im = re.findall(' Successful exploitation could lead to (.*)', row[1])[0]
            pd = re.findall('(.*?) Successful exploitation could lead to ', row[1])[0]
            if len(re.findall('have an ',row[1])):
                vt = re.findall('have an? (.*?)\. ',row[1])[0]
                im = re.findall('Successful exploitation could lead to (.*)',row[1])[0]
                pd = re.findall('(.*?) have an ',row[1])[0]
                if len(re.findall('due to .*?(?:,|\.) ', a1)):
                    does = re.findall('due to (.*?)(?:,|\.) ', a1)[0]
            if at == '':
                if len(im.split(' via ')) > 1:
                    at = 'via '+im.split(' via ')[-1]
                    im = im.split(' via ')[0]
            dt=[]
            dt.append(row[0])
            dt.append(pd)
            dt.append(aat)
            dt.append(im)
            dt.append(at)
            dt.append(does)
            dt.append(vt)
            dt.append(pdd)
            f4 = open(file4, 'a', newline='',encoding='utf-8')
            writer = csv.writer(f4)
            writer.writerow(dt)

        else:
            f5 = open(file5, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f5)
            writer.writerow(row)
    # except BaseException:
    #     #print(row[0])
