from gensim.models import Word2Vec
import numpy as np

def train_word2vec(sentences):
    if not sentences:
        raise ValueError("No sentences provided for training Word2Vec!")

    tokenized_sentences = [sentence.split() for sentence in sentences]  # Ensure tokenization

    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_word2vec_embedding(model, packet):
    words = packet.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)
