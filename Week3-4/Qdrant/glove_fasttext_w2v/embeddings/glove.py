from gensim.models import KeyedVectors
import numpy as np

def load_glove_model(glove_path):
    print(f"Loading GloVe model from {glove_path}...")
    return KeyedVectors.load_word2vec_format(glove_path, no_header=True, binary=False)

def get_glove_embedding(model, text):
    words = text.split()
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)  # Average word vectors for sentence embedding
    return np.zeros(100)  # Return zero vector if no words found (100d version)
