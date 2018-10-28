import csv
import pandas as pd
import jieba.analyse
import time
import jieba
import jieba.posseg
import os, sys
from gensim.models import word2vec


def get_train_words():
    #原始数据存储路径
    data_path = './data/user_tag_query.10W.TRAIN'

    #生成数据路径
    csvfile = open(data_path + '-1w.csv', 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'age', 'Gender', 'Education', 'QueryList'])

    #转换成utf-8编码的格式
    with open(data_path, 'r',encoding='gb18030',errors='ignore') as f:
        lines = f.readlines()
        for line in lines[0:10000]:
            try:
                line.strip()
                data = line.split("\t")
                writedata = [data[0], data[1], data[2], data[3]]
                querystr = ''
                data[-1]=data[-1][:-1]
                for d in data[4:]:
                    try:
                        cur_str = d.encode('utf8')
                        cur_str = cur_str.decode('utf8')
                        querystr += cur_str + '\t'
                    except:
                        continue
                        #print (data[0][0:10])
                querystr = querystr[:-1]
                writedata.append(querystr)
                writer.writerow(writedata)
            except:
                #print (data[0][0:20])
                continue


def get_test_words():
    data_path = './data/user_tag_query.10W.TEST'

    csvfile = open(data_path + '-1w.csv', 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'QueryList'])

    with open(data_path, 'r',encoding='gb18030',errors='ignore') as f:
        lines = f.readlines()
        for line in lines[0:10000]:
            try:
                data = line.split("\t")
                writedata = [data[0]]
                querystr = ''
                data[-1]=data[-1][:-1]
                for d in data[1:]:
                    try:
                        cur_str = d.encode('utf8')
                        cur_str = cur_str.decode('utf8')
                        querystr += cur_str + '\t'
                    except:
                        #print (data[0][0:10])
                        continue
                querystr = querystr[:-1]
                writedata.append(querystr)
                writer.writerow(writedata)
            except:
                #print (data[0][0:20])
                continue


def user_info():
    # 编码转换完成的数据，取的是1W的子集
    trainname = './data/user_tag_query.10W.TRAIN-1w.csv'
    testname = './data/user_tag_query.10W.TEST-1w.csv'
    data = pd.read_csv(trainname, encoding='gbk')

    # 分别生成三种标签数据（性别，年龄，学历）
    data.age.to_csv("./data/train_age.csv", index=False)
    data.Gender.to_csv("./data/train_gender.csv", index=False)
    data.Education.to_csv("./data/train_education.csv", index=False)

    # 将搜索数据单独拿出来
    data.QueryList.to_csv("./data/train_querylist.csv", index=False)
    data = pd.read_csv(testname, encoding='gbk')
    data.QueryList.to_csv("./data/test_querylist.csv", index=False)

    return trainname


def user_input(trainname):
    traindata = []
    with open(trainname, 'rb') as f:
        line = f.readline()
        count = 0
        while line:
            try:
                traindata.append(line)
                count += 1
            except:
                print ("error:", line, count)
            line = f.readline()
    return traindata


def decomposition_train_words():
    start = time.clock()

    filepath = './data/train_querylist.csv'
    QueryList = user_input(filepath)

    writepath = './data/train_querylist_writefile-1w.csv'
    csvfile = open(writepath, 'w')

    POS = {}
    for i in range(len(QueryList)):
        # print (i)
        if i % 2000 == 0 and i >= 1000:
            print (i, 'finished')
        s = []
        str = ""
        words = jieba.posseg.cut(QueryList[i])  # 带有词性的精确分词模式
        allowPOS = ['n', 'v', 'j']
        for word, flag in words:
            POS[flag] = POS.get(flag, 0) + 1
            if (flag[0] in allowPOS) and len(word) >= 2:
                str += word + " "

        cur_str = str.encode('utf8')
        cur_str = cur_str.decode('utf8')
        s.append(cur_str)

        csvfile.write(" ".join(s) + '\n')
    csvfile.close()

    end = time.clock()
    print ("total time: %f s" % (end - start))


def save_model():
    # 将数据变换成list of list格式
    train_path = './data/train_querylist_writefile-1w.csv'
    with open(train_path, 'r') as f:
        My_list = []
        lines = f.readlines()
        for line in lines:
            cur_list = []
            line = line.strip()
            data = line.split(" ")
            for d in data:
                cur_list.append(d)
            My_list.append(cur_list)

        model = word2vec.Word2Vec(My_list, size=300, window=10, workers=4)
        savepath = '1w_word2vec_' + '300' + '.model'  # 保存model的路径

        model.save(savepath)
        print('保存成功')


def demo():
    get_train_words()
    get_test_words()
    trainname = user_info()
    user_input(trainname)
    decomposition_train_words()
    save_model()



if __name__ == "__main__":
    demo()