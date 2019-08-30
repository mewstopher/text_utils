import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

def data_process(dataframe, text_col):
    """
    removes punctuation, tokenizes from a
    pandas dataframe

    PARAMS
    ---------------------------
    dat: pd dataframe
    """
    dataframe[text_col] = dataframe[text_col].apply(lambda x: text_to_word_sequence(x))
    return dataframe

def get_indices(X, word_to_index, maxlen):
    """
    get indices from comments, then padd comments

    PARAMS
    ------------------------
    X: values from dataset (comments)
    word_to_index: dictionary mapping words to
    indices
    maxlen: length to truncate comments at
    """
    max_sentence_length = len(max(X, key=len))
    indices = sentences_to_indices(X, word_to_index,
                                   max_sentence_length)
    X_padded = pad_sequences(indices, maxlen=maxlen,
                             truncating='post')
    return X_padded

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    """
    converts sentences in an array to an array of incdices

    PARAMS
    -------------------
    X = array of sentences
    word_to_index:  dictionary containing each word mapped to an index
    max_len: length of longest sentence in data
    """
    m = X.shape[0]

    X_indices = np.zeros([m, max_len])
    for i in range(m):
        sentence_words = X[i]
        j = 0
        for w in sentence_words:
            if w in word_to_index.keys():
                X_indices[i, j] = word_to_index[w]
                j = j + 1
    return X_indices

def build_embedding_matrix(word_to_vec_map, word_to_index):
    """
    Creates an embedding matrix from word_to_vec_map, and word_to_index
    Not framework specific

    PARAMS
    ------------------------------
    word_to_vec_map: a dictionary mapping words to their glove vector representation
    word_to_index: dictionary mapping words to their indices in the vocab
    """
    # add 1 for keras embedding requirement
    vocab_len = len(word_to_vec_map) + 1
    emb_dim = word_to_vec_map['cucumber'].shape[0]
    emb_matrix = np.zeros([vocab_len, emb_dim])
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    return emb_matrix

def pretrained_embedding_layer(word_to_vec_map, word_to_index,trainable=False):
    """
    creates a keras Embedding() layer and loads in pre-trained
    Glove 50 dimensional vectors

    PARAMS
    ------------------------------
    word_to_vec_map: a dictionary mapping words to their glove vector representation
    word_to_index: dictionary mapping words to their indices in the vocab
    """
    # add 1 for keras embedding requirement
    vocab_len = len(word_to_vec_map) + 1
    emb_dim = word_to_vec_map['cucumber'].shape[0]
    emb_matrix = np.zeros([vocab_len, emb_dim])
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    embedding_layer = Embedding(input_dim=vocab_len,
                                     output_dim=emb_dim,trainable=trainable)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def get_test_accuracy(X_test, y_test, model):
    """
    uses trained model to get test accuracy

    PARAMS
    ------------------------
    X_test = test set
    """
    loss, acc = model.evaluate(X_test, y_test)
    print("test accuracy = ", acc)


