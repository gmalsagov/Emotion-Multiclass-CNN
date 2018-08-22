import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_helper
from sklearn.manifold import TSNE
from gensim.models import word2vec, Word2Vec
import os

os.chdir("../")


def load(filename):

    df = pd.read_csv(filename)
    selected = ['text']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))
    df = df[0:100000]

    x_clean = df[selected[0]].apply(lambda x: data_helper.clean_text(x, True, False)).tolist()

    return x_clean


def build_corpus(sentences):
    "Creates a list of lists containing words from each sentence"

    corpus = []

    for sentence in sentences:

        words = sentence.split(' ')
        corpus.append(words)

    return corpus


def count_words(sentences):

    num_words = []
    max_num_words = 0

    for sentence in sentences:
        counter = len(sentence.split())

        if counter > max_num_words:
            max_num_words = counter
        num_words.append(counter)

    tokens = len(num_words)

    return num_words, max_num_words, tokens


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def display_closest_words(model, word):

    arr = np.empty((0, 100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


# Path to training data
filename = './data/imdb_tr.csv'
trained_model = './embeddings/IMDB/imdb.100d.model'

# Check if model already exists
if os.path.isfile(trained_model):
    model = Word2Vec.load(trained_model)

# If not, load data and create new model
else:
    time_str = datetime.datetime.now().isoformat()
    print(time_str + " Loading Data...")

    print('\n')

    # Load data from file
    sentences = load(filename)

    # Some statistics about loaded sentences
    num_words, max_num_words, tokens = count_words(sentences)

    print('The total number of reviews: ', tokens)
    print('The total number of words in corpus: ', sum(num_words))
    print('The maximum number of words in review: ', max_num_words)
    print('The average number of words per review: ', sum(num_words) / len(num_words))

    # Split sentences into list of words and pre-process them
    corpus = build_corpus(sentences)

    print('\n')

    time_str = datetime.datetime.now().isoformat()
    print(time_str + " Building a Word2Vec vocabulary...")

    # Build a word2vec matrix
    model = word2vec.Word2Vec(corpus, size=100, window=5, min_count=5, workers=4)

    print(time_str + " Training Neural Network...")

    # Train single layer neural network
    model.train(corpus, total_examples=len(corpus), epochs=100)
    print(time_str + " Network Trained!")

    # Check if path exists
    if not os.path.exists('./embeddings/IMDB'):
        os.makedirs('./embeddings/IMDB/')

    print("Saving trained model")
    model.save('./embeddings/IMDB/imdb.100d.model')
    model.wv.save_word2vec_format('./embeddings/IMDB/imdb.100d.txt', binary=False)

    print("Model Saved!")


# tsne_plot(model)

print(model.most_similar("interest"))