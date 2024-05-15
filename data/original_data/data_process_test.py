filename = "./data/trnTweet"

# import os
# print("当前工作目录:", os.getcwd())


with open(filename, encoding = 'utf-8') as f:
    datalist,taglist=[],[]
    for line in f:
        line=line.strip()
        datalist.append(line.split('\t')[0])
        taglist.append(line.split('\t')[1])
