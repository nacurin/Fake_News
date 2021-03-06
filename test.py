import pandas as pd
import sys
import csv
import gensim
import scipy
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import nltk
from nltk.stem import *

from numpy import *
import numpy as np
from random import shuffle

LabeledSentence = gensim.models.doc2vec.LabeledSentence
Doc2Vec = gensim.models.doc2vec
TaggedDocument = gensim.models.doc2vec.TaggedDocument
#pycharm script is "main.py test_bodies.csv test_stances.csv answers.csv"
file_name= sys.argv[1]
body_path = sys.argv[2]
train_path = sys.argv[3]
dev_path=sys.argv[4]
answer_path = sys.argv[5]

'''
iris = datasets.load_iris()
X = iris.data.tolist()
X = np.array(X)
X,y =iris.data,iris.target

a = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X,y).predict(X)
'''

dev = pd.read_csv(dev_path)
dev = dev.T
dev = dev.to_dict(orient = 'list')


#read train datas which are in project folder
train_data = pd.read_csv(train_path)
train_data = train_data.T
train = train_data.to_dict(orient = 'list')


body_data=pd.read_csv(body_path)
#body_data=body_data.head(20)
body_data=body_data.T
body=body_data.to_dict(orient = 'list')  #train data's structure is dict[]:0:headline 1:body id 2:stance 3:body
temp1 = []
temp2 = []

for key in dev:
    temp1.append(dev[key][0])
    temp2.append(dev[key][1])

def cleanword(a):
    a = nltk.tokenize.word_tokenize(a)  # tokenize
    a = [word.lower() for word in a]  # lower words
    a = [w for w in a if (w not in stopwords.words('english'))]  # remove stopwords
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '``']
    a = [word for word in a if (word not in english_punctuations)]
    #  a = [word for word in a if word.isalpha()]
    stemmer = SnowballStemmer("english")
    a = [stemmer.stem(word) for word in a]
    str = ""
    for s in a:
        str = str + " " + s

    return str

for key in dev:
    dev[key][0] = cleanword(dev[key][0])

for key in body:
    body[key][1] = cleanword(body[key][1])

for key in train:
    train[key][0] = cleanword(train[key][0])

body_dict = dict()
for k,v in body.items():
    body_dict[body[k][0]]=body[k][1]

#input
# data = pd.read_csv('bodies.csv')
# titleID = data[[0]]
# body = data[[1]]
# print(titleID)
# print(body)

def labelizeReviews(texts, label_type):
    labelized = []
    for i, v in enumerate(texts):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

corpora_documents = []
temp = ["test"]
for key in body:
    temp.append(body[key][1])

raw_documents = temp

print("start lebal documents")

for i, item_text in enumerate(raw_documents):
    words_list = raw_documents[i].split()
    document = TaggedDocument(words=words_list, tags=[i])
    corpora_documents.append(document)

print("start establish model")

#establish model
model = gensim.models.Doc2Vec(size=5, min_count=1, window=10, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(size=5, min_count=1, window=10, sample=1e-3, negative=5, dm=0, workers=3)

#build_vocab
print("start build vocab")
model.build_vocab(corpora_documents)
model_dbow.build_vocab(corpora_documents)

#first train
print("start first train")
model.train(corpora_documents)
model_dbow.train(corpora_documents)

# corpora_documents = numpy.array(corpora_documents)
# for epoch in range(10):
#     perm = numpy.random.permutation(corpora_documents.shape[0])
#     model.train(corpora_documents[perm])
#     model_dbow.train(corpora_documents[perm])


def calculateVec(text):
    return model_dbow.infer_vector(text)
#print(calculateVec("a apple"))


# build classifier
input = []
for key in train:
    index = train[key][1]
    val1 = calculateVec(train[key][0]).tolist()
    val2 = calculateVec(body_dict[index]).tolist()
    val1 = np.array(val1)
    val2 = np.array(val2)
    temp = scipy.spatial.distance.cosine(val1, val2)
    input.append([temp])
input = np.array(input)
#input = reshape(input, 40000)

target = []
for key in train:
    label = train[key][2]
    if label == 'unrelated':
        temp = 0
    elif label == 'agree':
        temp = 1
    elif label == 'disagree':
        temp = 2
    elif label == 'discuss':
        temp = 3
    target.append(temp)
target = np.array(target)
#target = reshape(target, 40000)
X,y = input,target

dev_data = []
for key in dev:
    index = dev[key][1]
    val1 = calculateVec(dev[key][0]).tolist()
    val2 = calculateVec(body_dict[index]).tolist()
    val1 = np.array(val1)
    val2 = np.array(val2)
    temp = scipy.spatial.distance.cosine(val1, val2)

    dev_data.append([temp])
dev_data = np.array(dev_data)
#dev_data = reshape(dev_data, 4000)

cls = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X,y)
answer = cls.predict(dev_data)

ans = answer.tolist()
result = []
for k in ans:
    label = ans[k]
    if label == 0:
        temp = 'unrelated'
    elif label == 1:
        temp = 'agree'
    elif label == 2:
        temp = 'disagree'
    elif label == 3:
        temp = 'discuss'
    result.append(temp)

temp = result

j=0
result = []
result.append(["headline" ,"Body ID" ,"stance" ])
while j < 4000:
    tempUnit = [temp1[j], temp2[j], temp[j]]
    result.append(tempUnit)
    j += 1

def creatCsv(fileName = "", data =[]):
    with open(fileName,"w") as csvFile:
        csvWriter = csv.writer(csvFile)
        k=0
        while k<len(result):
            str = result[k]
            csvWriter.writerow(str)
            k += 1
        csvFile.close()

creatCsv(answer_path,result)

print ("test finished")
ans=1
