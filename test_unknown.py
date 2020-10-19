import numpy as np
import pandas as pd
import pickle
from glob import glob
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


df = pd.read_csv("elf_v2.csv")
d2v = Doc2Vec.load("log/Full/d2v.model")
fs = pickle.load(open("log/Full/fs.pickle", "rb"))
scaler = pickle.load(open("log/Full/scaler.pickle", "rb"))
clf = pickle.load(open("log/Full/knn.pickle", "rb"))
total = true =  0

fpaths = glob("psi_walk_v2.5/unknown_malware/*")
for fpath in fpaths:
    fname = fpath.split('/')[-1].replace(".txt", "")
    with open(fpath, 'r') as f:
        data = f.read()
    enc = True
    for word in data.split():
        if not "sub" in word:
            enc = False
            break
    if enc:
        # print(fname, "encoded")
        continue

    total += 1
    print(fname, end=' ', flush=True)

    # Test
    x = TaggedDocument(words=word_tokenize(data), tags=[fname])
    x = d2v.infer_vector(x.words)
    x = np.array(x).reshape(1, -1)
    x = fs.transform(x)
    x = scaler.transform(x)
    y = clf.predict(x)[0]

    if y == 1:
        true += 1
    print(y, end=' ', flush=True)
    print(df.loc[df["md5"]==fname, "label"].values[0])

acc = 100 * float(true) / total
print(f"Accuracy: %.2f ({true}/{total})" % acc)