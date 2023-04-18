import json
import re

import numpy as np
from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
# bag of words (bow)
from sklearn.feature_extraction.text import CountVectorizer
# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import itertools
import matplotlib.pyplot as plt


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, text):
        return [self.wnl.lemmatize(t) for t in word_tokenize(text)]

def onehot_to_int(onehot):
    y = np.zeros(len(onehot))
    i = 0
    for o in onehot:
        max = np.argmax(o)
        y[i] = max
        i += 1
    return y


stopwordset = set(stopwords.words("english"))

# read json data from file utf-8 encoded
with open('data/cards.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

mycards = []
colors = []
i = 0
for card in data:
    if sum(card['color_identity']) > 1:
        continue
    rules_text = card['type_line']  # add type_line field
    #rules_text = card["rules_text"]
    rules_text += " " + card["rules_text"]
    rules_text = rules_text.strip()
    rules_text = rules_text.replace("\n", " ")
    rules_text = rules_text.replace("•", "")
    if card['power'] is not None:  # add power and toughness field
        rules_text += ' ' + card['power'] + ' ' + card['toughness']
    for c in rules_text:
        if c in punctuation:
            rules_text = rules_text.replace(c, ' ')
    rules_text = re.sub('\d+', '', rules_text)  # remove digits
    mycards.append(rules_text)
    colors.append(card["colors"])
    i += 1

def multinomial_naive_bayes(X, y, vect_name):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # multinomial naive bayes
    nb = MultinomialNB()  # model
    nb.fit(X_train, y_train)  # train
    y_pred_class_nb = nb.predict(X_test)  # make class predictions for X_test
    # output
    #print("Accuracy: ", accuracy_score(y_test, y_pred_class_nb))
    print(f'Accuracy NB {vect_name}: {nb.score(X_test, y_test)}')
    # plot confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred_class_nb)
    # print("cnf_matrix: ", cnf_matrix)
    plot_confusion_matrix(cnf_matrix, classes=['white', 'blue', 'black', 'red', 'green', 'colorless'], normalize=True,
                          title=f'Confusion matrix all features NB {vect_name}')

def logistic_regression(X, y, vect_name):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred_class_lr = clf.predict(X_test)  # make class predictions for X_test
    # output
    print(f'Accuracy LR {vect_name}: {clf.score(X_test, y_test)}')
    # plot confusion matrix
    cnf_matrix_lr = confusion_matrix(y_test, y_pred_class_lr)
    #print("cnf_matrix: ", cnf_matrix_lr)
    plot_confusion_matrix(cnf_matrix_lr, classes=['white', 'blue', 'black', 'red', 'green', 'colorless'],
                          normalize=True,
                          title=f'Confusion matrix all features LR {vect_name}')

# plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()

# bag of words (vectorize data)
# vectorizer_bow = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')  # lemmatization doesn't help
vectorizer_bow = CountVectorizer(stop_words='english')
X_bow = vectorizer_bow.fit_transform(mycards)
print('X_bow.shape', X_bow.shape)

# tfidf (vectorize data)
vectorizer_tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer_tfidf.fit_transform(mycards)
print('X_tfidf.shape', X_tfidf.shape)

y = onehot_to_int(colors)
print('y.shape', y.shape)

multinomial_naive_bayes(X_bow, y, "BOW")
multinomial_naive_bayes(X_tfidf, y, "TF-IDF")

logistic_regression(X_bow, y, "BOW")
logistic_regression(X_tfidf, y, "TF-IDF")
