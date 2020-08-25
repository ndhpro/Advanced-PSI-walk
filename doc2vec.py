import os
import multiprocessing
from tqdm import tqdm
from sklearn import utils
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from classify import classify


def load_data():
    file_list = []
    for root in ["psi-walk/malware/", "psi-walk/benign/"]:
        for _, _, files in os.walk(root):
            for file_name in files:
                file_list.append(root + file_name)

    print("Loading data...")
    data = dict()
    for file_name in tqdm(file_list):
        with open(file_name, 'r') as f:
            x = f.read()
            c = 1
            # for w in x.split():
            #     if not "sub" in w:
            #         c = 1
            #         break
            if c:
                data[x] = 0 if "benign" in file_name else 1
    X, y = list(), list()
    for k, v in data.items():
        X.append(k)
        y.append(v)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=2020, test_size=.3)

    tagged_X_train = [TaggedDocument(words=word_tokenize(
        d), tags=[str(i)]) for i, d in enumerate(X_train)]
    tagged_X_test = [TaggedDocument(words=word_tokenize(
        d), tags=[str(i)]) for i, d in enumerate(X_test)]

    return tagged_X_train, tagged_X_test, y_train, y_test


def doc2vec(X_train, X_test):
    print(f"Performing Doc2vec on {len(X_train)} documents...",)
    num_cores = multiprocessing.cpu_count()
    try:
        model = Doc2Vec(X_train,
                        vector_size=64,
                        dm=1,
                        window=2,
                        negative=5,
                        workers=num_cores,
                        epochs=100,
                        alpha=0.025)
    except Exception as e:
        print(e)
    model.save("result/model/d2v.model")

    print("Generating embeddings...")
    X_train_emb = [model.infer_vector(doc.words) for doc in X_train]
    X_test_emb = [model.infer_vector(doc.words) for doc in X_test]
    return X_train_emb, X_test_emb


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test = doc2vec(X_train, X_test)
    classify(X_train, X_test, y_train, y_test)
