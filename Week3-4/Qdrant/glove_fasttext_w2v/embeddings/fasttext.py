import fasttext
import fasttext.util
import numpy as np

def load_fasttext():
    fasttext.util.download_model('en', if_exists='ignore')
    model = fasttext.load_model('cc.en.300.bin')
    return model

def get_fasttext_embedding(model, packet):
    words = packet.split()
    vectors = [model[word] for word in words]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)
