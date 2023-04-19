import json
import re

import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
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
with open('../data/cards.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

mycards = []
colors = []
i = 0
for card in data:
    rules_text = card["rules_text"]
    rules_text = rules_text.strip()
    rules_text = rules_text.replace("\n", " ")
    rules_text = rules_text.replace("•", "")
    for c in rules_text:
        if c in punctuation:
            rules_text = rules_text.replace(c, ' ')
    rules_text = re.sub('\d+', '', rules_text)  # for digits
    mycards.append(rules_text)
    colors.append(card["colors"])
    i += 1

print(mycards)

for crd in mycards:
    print(crd)

# bag of words (vectorize data)
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
X = vectorizer.fit_transform(mycards)
print('X.shape', X.shape)

y = onehot_to_int(colors)
print('y.shape', y.shape)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# multinomial naive bayes
nb = MultinomialNB()        # model
nb.fit(X_train, y_train)    # train
y_pred_class = nb.predict(X_test) # make class predictions for X_test

print("Accuracy: ", accuracy_score(y_test, y_pred_class))

# confusion matrix
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
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)
    plt.show()

cnf_matrix = confusion_matrix(y_test, y_pred_class)
print("cnf_matrix: ", cnf_matrix)

plot_confusion_matrix(cnf_matrix, classes=['white', 'blue', 'black', 'red', 'green', 'colorless'], normalize=True,
                          title='Confusion matrix with all features')
