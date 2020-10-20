from run import run_file
from time import time
from glob import glob
import random
import pickle
import numpy as np
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

d2v = Doc2Vec.load("log/Full/d2v.model")
fs = pickle.load(open("log/Full/fs.pickle", "rb"))
scaler = pickle.load(open("log/Full/scaler.pickle", "rb"))
clf = pickle.load(open("log/Full/knn.pickle", "rb"))

graph_paths = glob("/home/server1/Downloads/psi_graph_v2/*/*")
graph_paths = random.choices(graph_paths, k=20)
ext_time = pred_time = n = 0
for gpath in graph_paths:
    t = time()
    run_file(gpath)
    fname = gpath.split('/')[-1].replace(".txt", "")
    fpath = "test/walk/" + fname + ".txt"
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r') as f:
        data = f.read()
    enc = True
    for word in data.split():
        if not "sub" in word:
            enc = False
            break
    if enc:
        continue
    ext_time += (time()-t)
    t = time()
    x = TaggedDocument(words=word_tokenize(data), tags=[fname])
    x = d2v.infer_vector(x.words)
    x = np.array(x).reshape(1, -1)
    x = fs.transform(x)
    x = scaler.transform(x)
    y = clf.predict(x)[0]
    print("Label:", y)
    n += 1
    pred_time += (time()-t)

print("Avg extr time: %.2f seconds (%d samples)" % ((ext_time / n), n))
print("Avg pred time: %.2f seconds (%d samples)" % ((pred_time / n), n))