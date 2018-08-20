import csv
import random
import string

import os
import re
import matplotlib.pyplot as plt
import logging
import itertools
import numpy as np
import pandas as pd
import gensim as gs
from pprint import pprint
from collections import Counter
import nltk
nltk.data.path.append('/Users/German/tensorflow/venv/lib/nltk_data')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from keras.preprocessing.text import Tokenizer

logging.getLogger().setLevel(logging.INFO)


def clean_text(text, remove_stopwords=False, stem_words=False):

    # Remove punctuation
    text = text.translate(None, punctuation)

    # Convert sentences to lower case and split into individual words
    text = text.lower().split()

    # Remove stop words (Optional)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = strip_links(text)
    text = strip_all_entities(text)

    # Stemming (Optional)
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text


# Removing URL's from tweets
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text

# Removing Tags and Hashtags
def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


def clean_str(s):
    """NOT NEEDED ANYMORE"""
    s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
    s = re.sub(r" : ", ":", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower()


def gloveVec(filename):
    embeddings = {}
    with open(filename, 'r') as data:
        # f = data.read().encode("utf-8")
        i = 0
        for line in data:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings[word] = coefs
            except ValueError:
                i += 1
    data.close()
    return embeddings


def load_embedding_vectors_word2vec(vocabulary, filename, vector_size):
    # load embedding_vectors from the word2vec

    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if True:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:

                    f.seek(binary_len, 1)
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors


def load_embeddings(vocabulary, dimension):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, dimension)
    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """Pad setences during training or prediction"""
    if forced_sequence_length is None:  # Train
        sequence_length = max(len(x) for x in sentences)
    else:  # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0:  # Prediction: cut off the sentence if it is longer than the sequence length
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def count_words(filename):
    sentences, labels, vocabulary, df = load(filename)

    num_words = []
    for line in sentences:
        counter = len(line.split())
        num_words.append(counter)

    num_files = len(num_words)

    return num_files, num_words


def load_data(filename):

    sentences = []
    labels = []

    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            sentences.append(row[1])
            labels.append(row[0])

    # Get all classes
    classes = sorted(list(set(labels)))

    # Encode labels into one-hot vectors
    labels = pd.get_dummies(labels, columns=classes).values.tolist()

    x_raw = map(clean_str, sentences)
    y_raw = labels

    # x_raw = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(x_raw)

    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])

    x_raw = np.array(x_raw)
    y = np.array(y_raw)

    return x_raw, y


def load(filename):

    df = pd.read_csv(filename)
    selected = ['label', 'text']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))
    df = df[0:100000]

    # Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_clean = df[selected[1]].apply(lambda x: clean_text(x, True, False)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

    # x_pad = pad_sentences(x_clean)
    vocabulary, vocabulary_inv = build_vocab(x_clean)

    # x = np.array([[vocabulary[word] for word in sentence] for sentence in x_pad])
    y = np.array(y_raw)

    return x_clean, y, vocabulary, df


def load4(path):

    tweets = []
    affect = []

    os.chdir(path)

    for filename in os.listdir(path):

        f = open(filename, 'r')
        lines = f.readlines()[1:]
        for x in lines:
            tweets.append(x.split('\t')[1])
            affect.append(x.split('\t')[2])
        f.close()

    return tweets, affect


def split_train_test_data(filename):

    classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

    df = pd.read_csv(filename)
    selected = ['label', 'text']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)  # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected)  # Drop null rows
    df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe
    df = df[0:100000]

    sentences = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    labels = df[selected[0]].tolist()

    joy = []
    fear = []
    anger = []
    sadness = []
    disgust = []
    shame = []
    guilt = []

    for i, e in enumerate(labels):
        if e == classes[0]:
            joy.append(sentences[i])
        elif e == classes[1]:
            fear.append(sentences[i])
        elif e == classes[2]:
            anger.append(sentences[i])
        elif e == classes[3]:
            sadness.append(sentences[i])
        elif e == classes[4]:
            disgust.append(sentences[i])
        elif e == classes[5]:
            shame.append(sentences[i])
        elif e == classes[6]:
            guilt.append(sentences[i])

    train_sentences = []
    train_labels = []
    test_labels = []

    count = 0

    joy_count = len(joy)
    fear_count = len(fear)
    anger_count = len(anger)
    sadness_count = len(sadness)
    disgust_count = len(disgust)
    shame_count = len(shame)
    guilt_count = len(guilt)

    while count < int(0.8 * joy_count):
        i = random.choice(range(len(joy)))
        train_sentences.append(joy[i])
        train_labels.append('joy')
        del joy[i]
        count = count + 1
    count = 0
    while count < int(0.8 * fear_count):
        i = random.choice(range(len(fear)))
        train_sentences.append(fear[i])
        train_labels.append('fear')
        del fear[i]
        count = count + 1
    count = 0
    while count < int(0.8 * anger_count):
        i = random.choice(range(len(anger)))
        train_sentences.append(anger[i])
        train_labels.append('anger')
        del anger[i]
        count = count + 1
    count = 0

    while count < int(0.8 * sadness_count):
        i = random.choice(range(len(sadness)))
        train_sentences.append(sadness[i])
        train_labels.append('sadness')
        del sadness[i]
        count = count + 1
    count = 0
    while count < int(0.8 * disgust_count):
        i = random.choice(range(len(disgust)))
        train_sentences.append(disgust[i])
        train_labels.append('disgust')
        del disgust[i]
        count = count + 1
    count = 0
    while count < int(0.8 * shame_count):
        i = random.choice(range(len(shame)))
        train_sentences.append(shame[i])
        train_labels.append('shame')
        del shame[i]
        count = count + 1
    count = 0
    while count < int(0.8 * guilt_count):
        i = random.choice(range(len(guilt)))
        train_sentences.append(guilt[i])
        train_labels.append('guilt')
        del guilt[i]
        count = count + 1

    test_sentences = joy + fear + anger + sadness + disgust + shame + guilt
    for x in joy:
        test_labels.append('joy')
    for x in fear:
        test_labels.append('fear')
    for x in anger:
        test_labels.append('anger')
    for x in sadness:
        test_labels.append('sadness')
    for x in disgust:
        test_labels.append('disgust')
    for x in shame:
        test_labels.append('shame')
    for x in guilt:
        test_labels.append('guilt')

    with open('./data/isear_train.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(itertools.izip(train_labels, train_sentences))

    with open('./data/isear_test.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(itertools.izip(test_labels, test_sentences))

    return


def build_word_embedding_mat(word_vecs, vocabulary_inv, k=300):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_mat = np.zeros(shape=(9000, k), dtype='float32')
    for idx in range(len(vocabulary_inv)):
        embedding_mat[idx + 1] = word_vecs[vocabulary_inv[idx]]
    print "Embedding matrix of size " + str(np.shape(embedding_mat))
    # initialize the first row,
    embedding_mat[0] = np.random.uniform(-0.25, 0.25, k)
    return embedding_mat


def vocab_to_word2vec(fname, vocab, vector_size):
    """
    Load word2vec from Mikolov
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    print str(len(word_vecs)) + " words found in word2vec."

    # add unknown words by generating random word vectors
    count_missing = 0
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, vector_size)
            count_missing += 1
    print str(count_missing) + " words not found, generated by random."
    return word_vecs


def load_data_and_labels(filename):
    """Load sentences and labels"""
    df = pd.read_csv(filename)
    selected = ['label', 'text']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)  # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected)  # Drop null rows
    df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe
    df = df[0:100000]
    print len(df)
    # Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

    vocabulary, vocabulary_inv = build_vocab(x_raw)

    word2vec = vocab_to_word2vec("../GoogleNews-vectors-negative300.bin", vocabulary, 300)
    # word2vec = vocab_to_word2vec("../embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt", vocabulary, 200)

    embedding_mat = build_word_embedding_mat(word2vec, vocabulary_inv)

    return x_raw, y_raw, df, labels, embedding_mat


def plot_confusion_matrix(cm, labels,
                          normalize=True,
                          title='Confusion Matrix (Validation Set)',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":

    split_train_test_data('./data/iseardataset.csv')
    # load_data('./data/isear_train.csv')
    # load4('./data/SemEval-2017/train/EI-reg-En-anger-train.txt')