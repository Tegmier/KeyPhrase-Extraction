# -*- coding: utf-8 -*-
import numpy as np
import pickle
from collections import Counter
from gensim.models import KeyedVectors
import time


def getlist(filename):
    # 打开文件名
    with open(filename, encoding = 'utf-8') as f:
        datalist,taglist=[],[]
        for line in f:
            line=line.strip()
            # 获得的推特内容 做成一个列表
            datalist.append(line.split('\t')[0])
            # 获得的hashtag内容 做成的一个列表
            taglist.append(line.split('\t')[1])
    # print(taglist)
    # print(datalist)
    return datalist,taglist

# 这个函数是建立词库，输入是两个：训练集和测试集
def get_dict(filenames):
    trnTweet,testTweet=filenames
    # getlist(XXXX)[0]是sentence，getlist(XXXX)[1]是hashtag
    sentence_list=getlist(trnTweet)[0]+getlist(testTweet)[0]

    words2idx=1,{}
    words=[]

    # 把句子给拆分成单词，得到全部单词
    for sentence in sentence_list:
        word_list=sentence.split()
        words.extend(word_list)
    
    # Counter函数主要来用来统计每个元素出现的次数
    word_counts=Counter(words)

    # 生成元组：字典（词，index） ,然后对每个出现的词按照index编号，index越小说明该词出现程度越频繁
    # {'the': 1, 'to': 2, 'in': 3, 'for': 4, 'a': 5, 
    words2idx={word[0]:i+1 for i,word in enumerate(word_counts.most_common())}

    # 这个字典是key word对于key phrase位置的flag
    labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}

    # 获得两个字典，一个是 {词库index：词}， 一个是 {词对于hashtag的关系：id}
    dicts = {'words2idx': words2idx, 'labels2idx': labels2idx}

    # 返回一个字典dicts，里头两个字典，一个是 'words2idx'：{词库index：词}， 一个是 'labels2idx'：{词对于hashtag的关系：id}
    return dicts

def get_train_test_dicts(filenames):
    """
    Args:
    filenames:trnTweet,testTweet,tag_id_cnt

    Returns:
    dataset:train_set,test_set,dicts

    train_set=[train_lex,train_y,train_z]
    test_set=[test_lex,test_y,test_z]
    dicts = {'words2idx': words2idx, 'labels2idx': labels2idx}


    """
    # 先获取两个字典： 一个是 {词index：词}， 一个是 {词对于hashtag的关系：id}
    trnTweetCnn, testTweetCnn= filenames
    dicts=get_dict([trnTweetCnn,testTweetCnn])

    # 把训练集和测试集的sentence和hashtag获取到
    trn_data=getlist(trnTweetCnn)
    test_data=getlist(testTweetCnn)

    # 得到训练集的sentence和hashtag
    trn_sentence_list,trn_tag_list=trn_data

    # 得到测试集的sentence和hashtag
    test_sentence_list,test_tag_list=test_data

    # 把两个字典取出来
    words2idx=dicts['words2idx']
    labels2idx=dicts['labels2idx']

    # 这个函数获取并返回了三个列表 lex[] y[] z[],lex装载的是map结构的单词的embedding值，y装载的是key word在原句子中所在的位置，z装载的是key word里每个单词的位置flag
    def get_lex_y(sentence_list,tag_list,words2idx):
        # lex是放embding的地方
        lex,y,z=[],[],[]
        # bad_cnt是没有出现在tweet里的key phrase的tweet
        bad_cnt=0
        for s,tag in zip(sentence_list,tag_list):
            word_list=s.split()
            t_list=tag.split()
            # 这里的emb只是字母的代号
            # map(function, iterable) 函数将函数 function 应用于可迭代对象 iterable 中的每个元素，并返回一个迭代器
            emb = map(lambda x:words2idx[x],word_list)

            # python2里map返回的是一个列表，这里迁移到python3以后手动加个列表
            emb = list(emb)
            begin=-1

            # 写的真的很垃圾的一个循环，首先是word_list一个一个往后推，然后用t_list去靠
            # 如果在word list里找到了重复的key word， 则用begin记录这个词在sentence里的位置
            for i in range(len(word_list)):
                ok=True
                for j in range(len(t_list)):
                    if word_list[i+j]!=t_list[j]:
                        ok=False
                        break
                if ok==True:
                    begin=i
                    break
            
            # 计算有几个句子不满足keygraph出现在句子中
            if begin==-1:
                bad_cnt+=1
                continue

            # 把embedding装进lex里头，非常土的embedding
            lex.append(emb)

            # 创建一个元素为0的长度为这个句子单词数量的列表， 然后把所有和tag一样的位置变成1
            labels_y=[0]*len(word_list)

            for i in range(len(t_list)):
                labels_y[begin+i]=1
            y.append(labels_y)

            # 创建一个元素为0的长度为这个句子单词数量的列表
            labels_z=[0]*len(word_list)

            # 如果key word长度为1，则其label_z的开始位置设置为'S'
            if len(t_list)==1:
                labels_z[begin]=labels2idx['S']

            # 如果key word的长度大于1，则暂且将其label_z的开始位置设置为'B'
            elif len(t_list)>1:
                labels_z[begin]=labels2idx['B']
            
                # key word中间的部分会被设置成为'I'
                for i in range(len(t_list)-2):
                    labels_z[begin+i+1]=labels2idx['I']

                # key word最后一个位置会被设置为'E'
                labels_z[begin+len(t_list)-1]=labels2idx['E']

            z.append(labels_z)
        return lex,y,z
    
    # 对train，test分别求 lex,y,z
    train_lex, train_y, train_z = get_lex_y(trn_sentence_list,trn_tag_list, words2idx)
    test_lex, test_y, test_z = get_lex_y(test_sentence_list,test_tag_list,words2idx)

    train_set = [train_lex, train_y, train_z]
    test_set = [test_lex, test_y, test_z]

    # 全部打包
    data_set = [train_set, test_set, dicts]

    # print(train_set[0])

    # 把处理好的数据给装进pkl里
    with open('data_set.pkl', 'wb') as f:
        pickle.dump(data_set, f)
    return data_set



def load_bin_vec(frame,vocab):
    k=0
    word_vecs={}
    with open(frame) as f:
        for line in f:
            word=line.strip().split(' ',1)[0]
            embeding=line.strip().split(' ',1)[1].split()
            if word in vocab:
                word_vecs[word]=np.asarray(embeding,dtype=np.float32)
            k+=1
            if k%10000==0:
                print ("load_bin_vec %d" % k)

    return word_vecs

def load_pretrained_model(model_path):
    start_time = time.time()
    print("Loading pretrained model...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    end_time = time.time()
    print("Pretrained model loaded. Time taken: {:.2f} seconds.".format(end_time - start_time))
    return model

def vector_transformation(model_path, vocab):
    k = 0
    word_vecs={}
    pretrained_model = load_pretrained_model(model_path)
    for word in vocab:
        word_vector = pretrained_model[word]
        word_vecs[word]=np.asarray(word_vector,dtype=np.float32)
        k+=1
        if k%10000==0:
            print ("Vextor transformation %d" % k)
    
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, dim=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    k=0
    for w in vocab:
        if w not in word_vecs:
            word_vecs[w]=np.asarray(np.random.uniform(-0.25,0.25,dim),dtype=np.float32)
            k+=1
            if k % 10000==0:
                print ("add_unknow_words %d" % k)
    return word_vecs

def get_embedding(w2v,words2idx,k=300):
    embedding = np.zeros((len(w2v) + 2, k), dtype=np.float32)
    for (w,idx) in words2idx.items():
        embedding[idx]=w2v[w]
    #embedding[0]=np.asarray(np.random.uniform(-0.25,0.25,k),dtype=np.float32)
    with open('embedding.pkl','w') as f:
        pickle.dump(embedding,f)
    return embedding


if __name__ == '__main__':

    # Modified by Tegmier
    # data_folder = ["original_data/trnTweet","original_data/testTweet"]
    trnTweet = "./data/trnTweet"
    testTweet = "./data/trnTweet"
    data_folder = [trnTweet,testTweet]
    # Modification end

    data_set = get_train_test_dicts(data_folder)
    print ("data_set complete!")

    # data_set = [train_set, test_set, dicts]
    dicts = data_set[2]
    # 创建一个无序的集合，这个集合是词库的集合
    vocab = set(dicts['words2idx'].keys())

    print ("total num words: " + str(len(vocab)))
    print ("vocabulary dataset created!")

    train_set, test_set, dicts=data_set
    print ("The length of training data: " + str(len(train_set[0])))

    #GoogleNews-vectors-negative300.txt为预先训练的词向量 

    # w2v_file='original_data/GoogleNews-vectors-negative300.txt' 
    # w2v=load_bin_vec(w2v_file,vocab)
    w2v_file = "./GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
    vector_transformation(w2v_file,vocab)


    print ("word2vec loaded")
    # add_unknown_words(w2v,vocab)
    # embedding=get_embedding(w2v,dicts['words2idx'])
    print ("embedding created")