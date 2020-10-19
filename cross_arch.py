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
elf_arch = {
    "ARM": "arm",
    "MIPS R3000": "mips",
    "i386": "i386",
    "AMD x86-64": "x64",
    "PowerPC": "powerpc",
    "SPARC": "sparc",
    "Motorola 68000": "motorola",
    "SuperH": "superh",
    "ARC": "arc"
}
benign_arch = {
    "ARM": "arm",
    "MIPS": "mips",
    "Intel 80386": "i386",
    "x86-64": "x64",
    "PowerPC": "powerpc",
    "Tilera": "tilera"
}
def run_dataset(save_path, train, test):
    elf_data = pd.read_csv("elf_v2.csv")
    beg_data = pd.read_csv("benign.csv")

    X_train_list = list()
    X_test_list = list()
    train_tmp = list()
    test_tmp = list()
    for root in psi_paths:
        for _, _, files in os.walk(root):
            for fname in files:
                with open(root + fname, 'r') as f:
                    data = f.read()
                enc = True
                for word in data.split():
                    if not "sub" in word:
                        enc = False
                        break
                if enc:
                    continue

                md5 = fname.split('.')[0]

                if "malware" in root:
                    arch = elf_data.loc[elf_data["md5"]==md5, "Machine"].values
                    if elf_arch[arch[0]] == train:
                        if data not in train_tmp:
                            X_train_list.append(root+fname)
                            train_tmp.append(data)
                    elif elf_arch[arch[0]] == test:
                        if data not in test_tmp:
                            X_test_list.append(root+fname)
                            test_tmp.append(data)
                else:
                    arch = beg_data.loc[beg_data["md5"]==md5, "arch"].values
                    if benign_arch[arch[0]] == train:
                        if data not in train_tmp:
                            X_train_list.append(root+fname)
                            train_tmp.append(data)
                    elif benign_arch[arch[0]] == test:
                        if data not in test_tmp:
                            X_test_list.append(root+fname)
                            test_tmp.append(data)


    print(len(X_train_list), len(X_test_list))

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
    args.add_argument("--train", type=str)
    args.add_argument("--test", type=str)

    args = args.parse_args()

    save_path = f"{args.train}-{args.test}"
    if not os.path.exists(f"log/{save_path}"):
        os.mkdir(f"log/{save_path}")

    X_train, X_test, y_train, y_test = run_dataset(save_path, args.train, args.test)
    X_train, X_test = doc2vec(save_path, X_train, X_test)
    classify(save_path, X_train, X_test, y_train, y_test)
