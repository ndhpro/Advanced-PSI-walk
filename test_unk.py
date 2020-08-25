import os
import numpy as np
import pandas as pd
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


df = pd.read_csv("data.csv")
d2v = Doc2Vec.load("result/model/d2v.model")
fs = pickle.load(open("result/model/fs.pickle", "rb"))
scaler = pickle.load(open("result/model/scaler.pickle", "rb"))
clf = pickle.load(open("result/model/knn.pickle", "rb"))

true, total = 0, 0
root = "psi-walk/test_unk/"
for _, _, files in os.walk(root):
    for fname in files:
        fpath = root + fname
        with open(fpath, 'r') as f:
            x = f.read()
            c = 1
            # for w in x.split():
            #     if not "sub" in w:
            #         c = 1
            #         break
            if c:
                total += 1
                print(fname, end=' ', flush=True)
                x = TaggedDocument(words=word_tokenize(x), tags=[fname])
                x = d2v.infer_vector(x.words)
                x = np.array(x).reshape(1, -1)
                x = fs.transform(x)
                x = scaler.transform(x)
                y = clf.predict(x)[0]
                if y == 1:
                    true += 1
                print(y, end=' ', flush=True)
                print(df.loc[df["md5"]==fname[:fname.find(".txt")], "label"].values[0])

acc = 100 * float(true) / total
print("Total:", total)
print("Correctly predicte:", true)
print("Accuracy (Unknown botnet): %0.2f" % acc)