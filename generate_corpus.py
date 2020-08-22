import os
from pathlib import Path
import numpy as np
import random
from sklearn.model_selection import train_test_split


# Preparing data
root_path = Path('psi_walk/')
folders = ['malware/', 'benign/']
corpus = list()
labels = list()

for folder in folders:
    for _, _, files in os.walk(root_path/folder):
        for name in files:
            try:
                with open(root_path/folder/name, 'r') as f:
                    doc = f.read().replace('\n', ' ')[:-1]
                words = doc.split(' ')
                enc = True
                for word in words:
                    if not 'sub' in word:
                        enc = False
                        break
                if enc:
                    continue
                corpus.append(doc)
                if 'benign' in folder:
                    labels.append(0)
                else:
                    labels.append(1)
            except Exception as e:
                print(e)

corpus, index = np.unique(corpus, axis=0, return_index=True)
labels = np.array(labels)[index]

X_train, X_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.3, random_state=2020)

with open('corpus/train_small.txt', 'w') as f:
    for i in range(len(X_train)):
        f.write(str(X_train[i]) + ' ' + str(y_train[i]) + '\n')

with open('corpus/test_small.txt', 'w') as f:
    for i in range(len(X_test)):
        f.write(str(X_test[i]) + ' ' + str(y_test[i]) + '\n')

print(len(X_train), len(X_test))
