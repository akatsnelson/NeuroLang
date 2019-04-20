# coding=utf-8
import itertools
# import sys
import warnings

import pymorphy2

# reload(sys)
# sys.setdefaultencoding('utf-8')

global morph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, \
    precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm
from joblib import dump

morph = pymorphy2.MorphAnalyzer()

warnings.filterwarnings('ignore')

# mpl.use('Agg')
#
# from pymystem3 import Mystem

# stemmer = Mystem()

plt.ioff()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.utils import shuffle

# def human_tokenize(text):
#     params = []
#     REGEX = re.compile(u'[^=[А-Яа-я ]+]*')
#     text = REGEX.sub("", text)
#     for word in nltk.word_tokenize(text):
#         parse = morph.parse(word)
#         # params.append(parse[0].normal_form)
#         params.append(str(parse[0].tag.POS))
#     return params


# def reShape(item1, item2):
#     result = [[], []]
#     for i in range(len(item1.shape[0])):
#         result[i] = [item1[i][0], [item1[i][1] + item2[i][1]]]
#     return result


# тут загрузте свои собственные данные
data = pd.read_csv(r'data.csv', sep=',', encoding='utf8')
data = shuffle(data)

y = np.array(data['class'])
tx1 = data['targTxt1']
tx2 = data['targTxt2']

X = [data.iloc[i]['targTxt1'] + ' ' + data.iloc[i]['targTxt2'] for i in range(len(data))]

dataE = pd.read_csv(r'essay.csv', sep=',', encoding='utf8')
dataE = shuffle(dataE)

yE = np.array(dataE['class'])
tx1E = dataE['targTxt1']
tx2E = dataE['targTxt2']

XE = [dataE.iloc[i]['targTxt1'] + ' ' + dataE.iloc[i]['targTxt2'] for i in range(len(dataE))]

# dataE = pd.read_csv(r'essay.csv', sep=',', encoding='utf8')
# dataE = shuffle(dataE)
#
# yE = np.array(dataE['class'])
# tx1E = dataE['targTxt1']
# tx2E = dataE['targTxt2']
#
# xE = [dataE.iloc[i]['targTxt1'] + ' ' + dataE.iloc[i]['targTxt2'] for i in range(len(dataE))]

count_vect = CountVectorizer(ngram_range=(1, 5), analyzer='char', lowercase=False,
                             max_features=11500).fit(X)

X1_train_counts = count_vect.transform(tx1)
X2_train_counts = count_vect.transform(tx2)

X_train_counts = count_vect.transform(X)

tfidf_transformer = TfidfTransformer(sublinear_tf=True).fit(X_train_counts)

X1_train_tfidf = tfidf_transformer.transform(X1_train_counts)
X2_train_tfidf = tfidf_transformer.transform(X2_train_counts)

X1 = [np.array_split(i.toarray().reshape((11500, 1)), 10) for i in X1_train_tfidf]
X2 = [np.array_split(i.toarray().reshape((11500, 1)), 10) for i in X2_train_tfidf]

items1 = []
items2 = []
for t in range(len(X1)):
    items1S = []
    items2S = []
    for r in itertools.product(X1[t], X2[t]):
        items1S.append(r[0])
        items2S.append(r[1])
    items1.append(items1S)
    items2.append(items2S)

items = []
for i in tqdm(range(len(items1))):
    vec = [cosine_similarity(np.array(items1[i][j]).reshape(1150, ), np.array(items2[i][j]).reshape(1150, ))[0][0] for j
           in range(len(items1[i]))]
    items.append(vec)

items_train, items_test, y_train, y_test = train_test_split(items, y, test_size=0.2, random_state=7)

clf = SVC(class_weight="balanced", kernel="linear", random_state=42).fit(items_train, y_train)

# essay

X1_train_countsE = count_vect.transform(tx1E)
X2_train_countsE = count_vect.transform(tx2E)

X1_train_tfidfE = tfidf_transformer.transform(X1_train_countsE)
X2_train_tfidfE = tfidf_transformer.transform(X2_train_countsE)

XE1 = [np.array_split(i.toarray().reshape((11500, 1)), 10) for i in X1_train_tfidfE]
XE2 = [np.array_split(i.toarray().reshape((11500, 1)), 10) for i in X2_train_tfidfE]

items1E = []
items2E = []
for t in range(len(XE1)):
    items1SE = []
    items2SE = []
    for r in itertools.product(XE1[t], XE2[t]):
        items1SE.append(r[0])
        items2SE.append(r[1])
    items1E.append(items1SE)
    items2E.append(items2SE)

itemsE = []
for i in tqdm(range(len(items1E))):
    vec = [cosine_similarity(np.array(items1E[i][j]).reshape(1150, ), np.array(items2E[i][j]).reshape(1150, ))[0][0] for
           j in range(len(items1E[i]))]
    itemsE.append(vec)


# end

predictions = clf.predict(items_test)
print("Precision: {0:6.2f}".format(precision_score(y_test, predictions, average='macro')))
print("Recall: {0:6.2f}".format(recall_score(y_test, predictions, average='macro')))
print("F1-measure: {0:6.2f}".format(f1_score(y_test, predictions, average='macro')))
print("Accuracy: {0:6.2f}".format(accuracy_score(y_test, predictions)))
print(classification_report(y_test, predictions) + '\n\n')
print(str(confusion_matrix(y_test, predictions)) + '\n\n')
print("Essay")
predictionsE = clf.predict(itemsE)
print("Precision: {0:6.2f}".format(precision_score(yE, predictionsE, average='macro')))
print("Recall: {0:6.2f}".format(recall_score(yE, predictionsE, average='macro')))
print("F1-measure: {0:6.2f}".format(f1_score(yE, predictionsE, average='macro')))
print("Accuracy: {0:6.2f}".format(accuracy_score(yE, predictionsE)))
print(classification_report(yE, predictionsE) + '\n\n')
print(str(confusion_matrix(yE, predictionsE)) + '\n\n')

dump(count_vect, 'сV.joblib')
dump(tfidf_transformer, 'tfidf.joblib')
dump(clf, 'clfSVM.joblib')
# predictions = clf.predict(xE)
# print("Precision: {0:6.2f}".format(precision_score(yE, predictions, average='macro')))
# print("Recall: {0:6.2f}".format(recall_score(yE, predictions, average='macro')))
# print("F1-measure: {0:6.2f}".format(f1_score(yE, predictions, average='macro')))
# print("Accuracy: {0:6.2f}".format(accuracy_score(yE, predictions)))
# print(classification_report(yE, predictions) + '\n\n')
# print(str(confusion_matrix(yE, predictions)) + '\n\n')
