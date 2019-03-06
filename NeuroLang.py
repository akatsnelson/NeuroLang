import warnings

from nltk.corpus.reader import nltk
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings('ignore')
import re
import pymorphy2

global morph
morph = pymorphy2.MorphAnalyzer()

import pandas as pd
import numpy as np
from nltk import word_tokenize
import matplotlib.pyplot as plt

# get_ipython().magic('matplotlib inline')
from sklearn.svm import SVC

# mpl.use('Agg')

from pymystem3 import Mystem

stemmer = Mystem()

# from pymystem3 import Mystem
# m = Mystem()
# A = []
#
# for i in titles:
#     #print(i)
#     lemmas = m.lemmatize(i)
#     A.append(lemmas)
#
# #Этот массив можно сохранить в файл либо "забэкапить"
# import pickle
# with open("mystem.pkl", 'wb') as handle:
#                     pickle.dump(A, handle)

plt.ioff()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import *
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def human_tokenize(text):
    REGEX = re.compile(r"[\._\-:;\/\[\]\(\\'<>)\`\"\{\}#]*")
    pattern = re.compile(r'\d+')
    text = [word_tokenize(tok.strip()) for tok in REGEX.split(text)]
    text = [item for sublist in text for item in sublist]
    text = [t for t in text if pattern.findall(t) == []]
    # text = stopwords(text, mystopwords)
    return text


def token_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def token_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# тут загрузте свои собственные данные
data = pd.read_csv(r'../csv.csv', sep=',', encoding='utf8')
data = shuffle(data)

y = np.array(data['class'])
X = np.array(data['tagged_texts'])

# X y мы разбиваем в соотношении 80 на 20, чтобы обучаться на 80% и тестироваться на 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
print("X_train:", len(X_train), "X_test:", len(X_test), "y_train:", len(y_train), "y_test:", len(y_test), "X:", len(X),
      "y:", len(y))

clf = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 5), analyzer='word', lowercase=False, tokenizer=human_tokenize)),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
    ('reducer', TruncatedSVD(n_components=100)),
    ('clf', SVC(class_weight="balanced", kernel="linear", random_state=42))])

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Precision: {0:6.2f}".format(precision_score(y_test, predictions, average='macro')))
print("Recall: {0:6.2f}".format(recall_score(y_test, predictions, average='macro')))
print("F1-measure: {0:6.2f}".format(f1_score(y_test, predictions, average='macro')))
print("Accuracy: {0:6.2f}".format(accuracy_score(y_test, predictions)))
print(classification_report(y_test, predictions) + '\n\n')
print(str(confusion_matrix(y_test, predictions)) + '\n\n')


def visualize_coefficients(classifier, feature_names, n_top_features=25):
    coef = classifier
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefficients], rotation=60, ha="right")
    plt.show()


visualize_coefficients(clf.named_steps['clf'].coef_[0, :], clf.named_steps['vect'].get_feature_names())
