from input import data
import random
from collections import defaultdict
import numpy as np
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from sklearn.ensemble import GradientBoostingClassifier
from score import report_score, score_submission

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.body[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y



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
data = data()  #load data
folds, fold_data = kfold(data,8)
Xs = dict()#feature
ys = dict()#stance
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

print(best_score)
