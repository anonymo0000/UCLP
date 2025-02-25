import re
import csv
from tqdm import tqdm

# file5 = r'/home/user/Programs/PMA_lihang/data/pt2_file5.csv'#CVE
# file6 = r'/home/user/Programs/PMA_lihang/data/pt3_file6.csv' # in the following patten
# file7 = r'/home/user/Programs/PMA_lihang/data/pt3_file7.csv' # not in >>>

# count=0

def devide3(f_in, f_out, f_next):

    file5 = f_in
    file6 = f_out
    file7 = f_next

    # f_temp = open(file7, 'w', encoding='utf-8')
    # f_temp.truncate()
    # f_temp.close()

    f5=open(file5,encoding='utf-8')
    for row in tqdm(csv.reader(f5)):
    #   try:
        # print(row)
        # break
        pd=''
        does=''
        at=''
        av=''
        vt=''
        im=''
        cve=row[0]
        if len(re.findall('In .* there is a?n? (?:possible)? .* due to ',row[1])):
            pd =re.findall('In (.*?) there is a?n? (?:possible)? .*',row[1])[0]
            vt=re.findall('In .* there is a?n? (?:possible)? (.*?) due ',row[1])[0]
            does=re.findall('In .* there is a?n? (?:possible)? .* due to (.*?)\.',row[1])[0]
            if len(re.findall('lead to ',row[1])):
                im= re.findall('lead to (.*)\.',row[1])[0]
                aat = re.findall('lead to .*\.(.*?)(?:(?:Product:)|(?:\Z))',row[1])[0]
                if len(re.findall('Product:',row[1])):
                    pd+='***'+re.findall('Product:(.*)',row[1])[0]
            if av == '':
                if len(im.split(' via ')) > 1:
                    av = 'via '+im.split(' via ')[-1]
                    im = im.split(' via ')[0]
            if av == '':
                if len(re.findall('by [a-z]+ing ',im)):
                    av = re.findall('by [a-z]+ing .*',im)[0]
                    im = im.split(av)[0]
            if av == '':
                if len(re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)):
                    av = re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)[0]
                    im = im.split(av)[-1]
            dt = []
            dt.append(row[0])
            dt.append(pd)
            dt.append(at)
            dt.append(im)
            dt.append(av)
            dt.append(does)
            dt.append(vt)

            f6 = open(file6, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f6)
            writer.writerow(dt)
        elif len(re.findall('In .* there is a (?:possible)? .* that can be caused by', row[1])):
            pd = re.findall('In (.*?) there is a .*', row[1])[0]
            aat=re.findall('In .* there is a (?:possible)? .* that can be caused by(.*?) in', row[1])[0]
            # does = re.findall('In .* there is a possible .* due to (.*?)\. ', row[1])[0]
            # if len(re.findall('lead to ', row[1])):
            #     im = re.findall('lead to (.*)\.', row[1])[0]
            #     aat = re.findall('lead to .*\.(.*?)Product:', row[1])[0]
            pd += '***' + re.findall('In .* there is a (?:possible)? .* that can be caused by.*? in (.*)', row[1])[0]
            if av == '':
                if len(im.split(' via ')) > 1:
                    av = 'via '+im.split(' via ')[-1]
                    im = im.split(' via ')[0]
            if av == '':
                if len(re.findall('by [a-z]+ing ',im)):
                    av = re.findall('by [a-z]+ing .*',im)[0]
                    im = im.split(av)[0]
            if av == '':
                if len(re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)):
                    av = re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)[0]
                    im = im.split(av)[-1]
            dt = []
            dt.append(row[0])
            dt.append(pd)
            dt.append(at)
            dt.append(im)
            dt.append(av)
            dt.append(does)
            dt.append(vt)

            f6 = open(file6, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f6)
            writer.writerow(dt)
        elif len(re.findall('.*?(?:(?:is )|(?: are ))affected by:',row[1])):
            vt=re.findall('.*?(?:(?:is )|(?: are ))affected by:(.*?)',row[1])[0]

            pd=re.findall('(.*)(?:(?:is )|(?: are ))affected by:',row[1])[0]
            if len(re.findall('.*? The impact is:(.*?)',row[1])):
                im=re.findall('.*? The impact is:(.*?)',row[1])[0]
                if len(re.findall('The component is:(.*)',row[1])):
                    pd += '***'+re.findall('The component is:(.*)',row[1])[0]
            if len(re.findall('The attack vector is:',row[1])):
                aat=re.findall('The attack vector is:',row[1])[0]
            if av == '':
                if len(im.split(' via ')) > 1:
                    av = 'via '+im.split(' via ')[-1]
                    im = im.split(' via ')[0]
            if av == '':
                if len(re.findall('by [a-z]+ing ',im)):
                    av = re.findall('by [a-z]+ing .*',im)[0]
                    im = im.split(av)[0]
            if av=='':
                if len (re.findall('\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',im)):
                    av=re.findall('\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',im)[0]
                    im=im.split(av)[-1]
            dt = []
            dt.append(row[0])
            dt.append(pd)
            dt.append(at)
            dt.append(im)
            dt.append(av)
            dt.append(does)
            dt.append(vt)

            f6 = open(file6, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f6)
            writer.writerow(dt)
        elif len(re.findall('(?:(?:is)|(?:are)) vulnerable to.*? result(?:ing)? in ',row[1])):
            pd =re.findall('(.*)(?:(?:is)|(?:are)) vulnerable to',row[1])[0]
            vt =re.findall('(?:(?:is)|(?:are)) vulnerable to(.*?) result(?:ing)? in ',row[1])[0]
            im=re.findall('(?:(?:is)|(?:are)) vulnerable to .*? result(?:ing)? in (.*?)\.',row[1])
            if(len(im)):
                im=im[0]
            else:
                im=''
            if av == '':
                if len(im.split(' via ')) > 1:
                    av = 'via '+im.split(' via ')[-1]
                    im = im.split(' via ')[0]
            if av == '':
                if len(re.findall('by [a-z]+ing ',im)):
                    av = re.findall('by [a-z]+ing .*',im)[0]
                    im = im.split(av)[0]
            if av == '':
                if len(re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)):
                    av = re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)[0]
                    im = im.split(av)[-1]
            dt = []
            dt.append(row[0])
            dt.append(pd)
            dt.append(at)
            dt.append(im)
            dt.append(av)
            dt.append(does)
            dt.append(vt)

            f6 = open(file6, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f6)
            writer.writerow(dt)
        elif len(re.findall('\AThere is(?:.*?) in ', row[1])):
            vt =re.findall('\AThere is(.*?) in ', row[1])[0]
            pd =re.findall('\AThere is(?:.*?) in .*?(?:(?:, )|(?:\. )|(?:\Z))', row[1])[0]
            if len(re.findall('\AThere is(?:.*?) in (?:.*?)\.(?:(?:\Z)|(?: ))(.*) lead to ', row[1])):
                av =re.findall('\AThere is(?:.*?) in (?:.*?)\.(?:(?:\Z)|(?: ))(.*) lead to ', row[1])[0]
                im=re.findall('\AThere is(?:.*?) in (?:.*?)\.(?:(?:\Z)|(?: ))(?:.*) lead to (.*)', row[1])[0]
            elif len(pd.split(' lead to '))>1:
                im=pd.split('lead to')[-1]
                pd = pd.split(' lead to ')[0]
            if av == '':
                if len(im.split(' via ')) > 1:
                    av = 'via '+im.split(' via ')[-1]
                    im = im.split(' via ')[0]
            if av == '':
                if len(re.findall('by [a-z]+ing ',im)):
                    av = re.findall('by [a-z]+ing .*',im)[0]
                    im = im.split(av)[0]
            if av == '':
                if len(re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)):
                    av = re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)[0]
                    im = im.split(av)[-1]
            dt = []
            dt.append(row[0])
            dt.append(pd)
            dt.append(at)
            dt.append(im)
            dt.append(av)
            dt.append(does)
            dt.append(vt)

            f6 = open(file6, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f6)
            writer.writerow(dt)
        elif  len(re.findall('(?: allows? )|(?: has )',row[1])):
            im = re.findall('(?:(?: allows? )|(?: has )).*',row[1])[0]
            pd = row[1].replace(im,'')
            if len(re.findall(
                    '(?:(?: when )|(?: improperly )|(?: fails? )|(?: automatically )|(?: does )|(?: do )|(?: uses? )|(?: returns? )|(?: creates? )|(?: provides? )|(?: relies? )|(?: places? )|(?: generates? )|(?: advertises )|(?: mishandles )|(?: store SSH )|(?: lets? )|(?: handles )|(?: using )|(?: lets? )).*',
                    pd)):
                does = re.findall(
                    '(?:(?: when )|(?: improperly )|(?: fails? )|(?: automatically )|(?: does )|(?: do )|(?: uses? )|(?: returns? )|(?: creates? )|(?: provides? )|(?: relies? )|(?: places? )|(?: generates? )|(?: advertises )|(?: mishandles )|(?: store SSH )|(?: lets? )|(?: handles )|(?: using )|(?: lets? )).*',
                    pd)[0]
                pd = pd.split(does)[0]
            if av == '':
                if len(im.split(' via ')) > 1:
                    av = 'via '+im.split(' via ')[-1]
                    im = im.split(' via ')[0]
            if av == '':
                if len(re.findall('by [a-z]+ing ',im)):
                    av = re.findall('by [a-z]+ing .*',im)[0]
                    im = im.split(av)[0]
            if av == '':
                if len(re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)):
                    av = re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
                        im)[0]
                    im = im.split(av)[-1]
            dt = []
            dt.append(row[0])
            dt.append(pd)
            dt.append(at)
            dt.append(im)
            dt.append(av)
            dt.append(does)
            dt.append(vt)

            f6 = open(file6, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f6)
            writer.writerow(dt)
        elif len(re.findall('\AAn? .*? occur when ',row[1])):
            vt=re.findall('\A(An? .*?) occur when ',row[1])[0]
            if len (re.findall('\AAn? .*? occur when .*?(?:(?:\.)|(?:result(?:ing) in)) ',row[1])):
                does=re.findall('\AAn? .*? occur (when .*?)(?:(?:\.)|(?:result(?:ing) in)) ',row[1])[0]
                im=re.findall('\AAn? .*? occur when .*?(?:(?:\.)|(?:result(?:ing) in))(.*) ',row[1])[0]
            if av == '':
                if len(re.findall('by [a-z]+ing ',im)):
                    av = re.findall('by [a-z]+ing .*',im)[0]
                    im = im.split(av)[0]
            if av == '':
                if len(re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct )|(?:compromise )).*? to',
                        im)):
                    av = re.findall(
                        '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct )|(?:compromise )).*? to',
                        im)[0]
                    im = im.split(av)[-1]
            dt = []
            dt.append(row[0])
            dt.append(pd)
            dt.append(at)
            dt.append(im)
            dt.append(av)
            dt.append(does)
            dt.append(vt)
            
            f6 = open(file6, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f6)
            writer.writerow(dt)
        # elif len(re.findall('\AAn? .*? exists in ',row[1])):
        #     vt=re.findall('\A(An? .*?) exists in ',row[1])[0]
        #     if len (re.findall('\AAn? .*? exists in .*?(?:(?:\.)|(?:result(?:ing) in)) ',row[1])):
        #         does=re.findall('\AAn? .*? occur (when .*?)(?:(?:\.)|(?:result(?:ing) in)) ',row[1])[0]
        #         im=re.findall('\AAn? .*? occur when .*?(?:(?:\.)|(?:result(?:ing) in))(.*) ',row[1])[0]
        #     if len(re.findall(' affect',row[1])):
        #         pd=re.findall(' affect[^ ]*? (.*)',row[1])
        #     if av == '':
        #         if len(re.findall(
        #                 '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
        #                 im)):
        #             av = re.findall(
        #                 '\A(?:(?:use )|(?:send )|(?:supply )|(?:specially )|(?:upload )|(?:create )|(?:construct  )|(?:compromise )).*? to',
        #                 im)[0]
        #             im = im.split(av)[-1]
        #     dt = []
        #     dt.append(row[0])
        #     dt.append(pd)
        #     dt.append(at)
        #     dt.append(im)
        #     dt.append(av)
        #     dt.append(does)
        #     dt.append(vt)
        #     f2 = open(file1, 'a', newline='', encoding='utf-8')
        #     writer = csv.writer(f2)
        #     writer.writerow(dt)
        else:

            f7 = open(file7, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f7)
            writer.writerow(row)
    #   except BaseException:
    #       count += 1

    # print(count)