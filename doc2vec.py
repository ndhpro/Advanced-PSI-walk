import os
import multiprocessing
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


def doc2vec(save_path, X_train, X_test):
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
    model.save(f"log/{save_path}/d2v.model")

    print("Generating embeddings...")
    X_train_emb = [model.infer_vector(doc.words) for doc in X_train]
    X_test_emb = [model.infer_vector(doc.words) for doc in X_test]
    return X_train_emb, X_test_emb
