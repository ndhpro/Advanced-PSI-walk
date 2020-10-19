import os
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from doc2vec import doc2vec
from classify import classify


psi_paths = [
    "psi_walk_v2.5/malware/",
    "psi_walk_v2.5/benign/"
]

def run_dataset(dataset_name):
    elf_data = pd.read_csv("elf_v2.csv")

    X = list()
    tmp = list()
    for root in psi_paths:
        for _, _, files in os.walk(root):
            for fname in files:
                with open(root + fname, 'r') as f:
                    data = f.read()
                if data in tmp:
                    continue
                else:
                    tmp.append(data)
                enc = True
                for word in data.split():
                    if not "sub" in word:
                        enc = False
                        break
                if enc:
                    continue

                md5 = fname.split('.')[0]
                if dataset_name == "Full":
                    X.append(root+fname)
                else:
                    dataset = elf_data.loc[elf_data["md5"]==md5, "Dataset"].values
                    if not len(dataset) or dataset == dataset_name:
                        X.append(root+fname)
    print(len(X))
    X_train_list, X_test_list= train_test_split(X, random_state=2020, test_size=.3)

    X_train_aug = X_train_list.copy()
    for fpath in X_train_list:
        for i in range(3):
            aug_path = fpath.replace("psi_walk_v2.5", "aug") + str(i)
            if os.path.exists(aug_path):
                X_train_aug.append(aug_path)

    
    X_train = list()
    y_train = list()
    for fpath in X_train_aug:
        with open(fpath, "r") as f:
            data = f.read()
        X_train.append(data)
        if "benign" in fpath:
            y_train.append(0)
        else:
            y_train.append(1)

    X_test = list()
    y_test = list()
    for fpath in X_test_list:
        with open(fpath, "r") as f:
            data = f.read()
        X_test.append(data)
        if "benign" in fpath:
            y_test.append(0)
        else:
            y_test.append(1)
    
    print(len(X_train), len(X_test))

    tagged_X_train = [TaggedDocument(words=word_tokenize(
        d), tags=[str(i)]) for i, d in enumerate(X_train)]
    tagged_X_test = [TaggedDocument(words=word_tokenize(
        d), tags=[str(i)]) for i, d in enumerate(X_test)]

    return tagged_X_train, tagged_X_test, y_train, y_test

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--cross-arch", type=bool, default=False)
    args.add_argument("--dataset", type=str, default="Full")

    args = args.parse_args()

    if not args.cross_arch:
        dataset_name = args.dataset
        if not os.path.exists("log/" + dataset_name):
            os.mkdir("log/" + dataset_name)
        X_train, X_test, y_train, y_test = run_dataset(dataset_name)
        X_train, X_test = doc2vec(dataset_name, X_train, X_test)
        classify(dataset_name, X_train, X_test, y_train, y_test)
