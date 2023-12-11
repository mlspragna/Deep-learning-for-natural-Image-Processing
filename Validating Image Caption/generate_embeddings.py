from collections import Counter
import pandas as pd
import ast
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def get_vocab(dataset_df):
    #dataset_df['positive_captions'] = dataset_df['positive_captions'].apply(ast.literal_eval)
    word_counts = Counter()
    for sentences in dataset_df['positive_captions']:
        words = ' '.join(sentences).split()
        word_counts.update(words)
    vocab = [w for w in word_counts if word_counts[w]>=10]
    return vocab        

def max_sentence_length(dataset_df):
    #dataset_df['positive_captions'] = dataset_df['positive_captions'].apply(ast.literal_eval)
    fun = lambda captions: max([len(sentence.split(' ')) for sentence in captions])
    return dataset_df['positive_captions'].apply(fun).max()

def get_word_indices(vocab):
    wordtoix = {}
    idx = 1
    for w in vocab:
        wordtoix[w] = idx
        idx += 1
    return wordtoix

def get_embedding_matrix(glove_path, vocab_size, wordtoix):
    file = open(glove_path, encoding="utf-8")
    embeddings_glove = {}
    for line in file:
        word_embed = line.split()
        word = word_embed[0]
        embed = np.asarray(word_embed[1:], dtype='float32')
        embeddings_glove[word] = embed

    embedding_dim = 200
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in wordtoix.items():
        embedding_vector = embeddings_glove.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_indices_of_captions(cap, word_to_indices, max_sen_length):
    cap_indices = [word_to_indices[w] for w in cap.split(' ') if w in word_to_indices]
    cap_indices_padded = pad_sequences([cap_indices], maxlen=max_sen_length, padding='post')[0]
    return cap_indices_padded

