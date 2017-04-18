from input import data
from input import testdata
import random
from collections import defaultdict
import numpy as np
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features, clean
from sklearn.ensemble import GradientBoostingClassifier
from score import report_score, score_submission
import lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction
import re
import csv
from collections import OrderedDict

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.body[stance['Body ID']])
        BodyIds = []
        BodyIds.append(stance['Body ID'])

    X_lda = gen_or_load_feats(lda_features, h, b, "features/lda." + name + ".npy")
    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_lda]
    return X,y

def generate_test_features(stances,dataset,name):
    h, b = [],[]

    for stance in stances:
        h.append(stance['Headline'])
        b.append(dataset.body[stance['Body ID']])

    X_lda = gen_or_load_feats(lda_features, h, b, "features/lda" + name + ".npy")
    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_lda]
    return X

def lda_generate_model(headlines, bodies):

    X = []

    # get train data
    templist = []
    # clean_headlines = []
    # clean_bodies = []
    # for headline in headlines:
    #     clean_headlines.append(clean(headline))
    # for body in bodies:
    #     clean_bodies.append(clean(body))
    clean_headlines = headlines
    clean_bodies = bodies

    # get test data
    test_stances = test.test
    test_dataset = data
    test_headlines, test_bodies = [], []
    for stance in test_stances:#test has not been cleaned
        test_headlines.append(stance['Headline'])
        test_bodies.append(test_dataset.body[stance['Body ID']])

    # add train & test
    clean_headlines = list(set(clean_headlines))
    clean_bodies = list(set(clean_bodies))
    test_headlines = list(set(test_headlines))
    templist = clean_headlines + clean_bodies + test_headlines

    cv = CountVectorizer()
    cv_fit = cv.fit_transform(templist)
    cv_fit = cv_fit.toarray()

    model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    model.fit(cv_fit)

    # get lda dict
    vec_dict = OrderedDict()
    doc_topic = model.doc_topic_
    i = 0
    for i in range(len(templist)):
        vec_dict[templist[i]] = doc_topic[i]

    print("lda_generate_model complete!")
    return vec_dict

def lda_features(headlines, bodies):

    # calculate the dot
    lda_feature_result = []
    for i in range(len(headlines)):
        headline_vec = vec_dict[headlines[i]]
        body_vec = vec_dict[bodies[i]]
        headline_vec = np.array(headline_vec)
        body_vec = np.array(body_vec)

        lda_feature_result.append(np.inner(headline_vec, body_vec))

    return np.array(lda_feature_result)

def kfold(data, n_folds):
    folds = []

    r = random.Random()
    r.seed(1)
    trainidx = list(data.body.keys())
    r.shuffle(trainidx)

    for k in range(n_folds):
        folds.append(trainidx[int(k*len(trainidx)/n_folds):int((k+1)*len(trainidx)/n_folds)]) #one of methods to generate k-folds

    train_data = defaultdict(list)
    for line in data.train:
        fold_id = 0
        for fold in folds:
            if line['Body ID'] in fold:
                train_data[fold_id].append(line)
            fold_id += 1

    return folds,train_data

#main
test = testdata()   #load test data
data = data()  #load data
all_headlines = []
all_bodies = []

for line in data.train:
    all_headlines.append(line['Headline'])

for key in data.body:
    all_bodies.append(data.body[key])

vec_dict = lda_generate_model(all_headlines, all_bodies)
folds, fold_data = kfold(data,8)
Xs = dict()#feature
ys = dict()#stance

X_testset = generate_test_features(test.test,data,"test")  #new
for fold in fold_data:
    Xs[fold], ys[fold] = generate_features(fold_data[fold], data, str(fold))

best_score = 0
best_fold = None

# Classifier for each fold
for fold in fold_data:
    ids = list(range(len(folds)))
    del ids[fold]

    X_train = np.vstack(tuple([Xs[i] for i in ids]))
    y_train = np.hstack(tuple([ys[i] for i in ids]))

    X_test = Xs[fold]
    y_test = ys[fold]

    clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
    clf.fit(X_train, y_train)

    predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
    actual = [LABELS[int(a)] for a in y_test]

    fold_score, _ = score_submission(actual, predicted)
    max_fold_score, _ = score_submission(actual, actual)

    score = fold_score / max_fold_score

    print("Score for fold " + str(fold) + " was - " + str(score))
    if score > best_score:
        best_score = score
        best_fold = clf

predicted = [LABELS[int(a)] for a in best_fold.predict(X_testset)]

j=0
result = []
result.append(["Headline" ,"Body ID" ,"Stance" ])
while j < 4000:
    tempUnit = [test.test[j]['Headline'], test.test[j]['Body ID'], predicted[j]]
    result.append(tempUnit)
    j += 1

def creatCsv(fileName = "", data =[]):
    with open(fileName,"w",encoding = "UTF-8") as csvFile:
        csvWriter = csv.writer(csvFile)
        k=0
        while k<len(result):
            str = result[k]
            csvWriter.writerow(str)
            k += 1
        csvFile.close()

creatCsv("answer.csv",result)

print ("test finished")

print(best_score)


