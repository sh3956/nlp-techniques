"""
Similar to the neural network:
Skipgram: an unsupervised learning algorithm that predict
the most related words for a given word [related words]
trained weight matrix can be used for word embedding
CBOW: given context, predict word
"""
import numpy as np


def softmax(x):
    """ take a normalized softmax
    params:
        x(np.array): word probability vector
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


class word2vec(object):
    def __init__(self):
        """
        Attributes:
            N (int): size of the hidden layers
            the number of columns in embedding matrix
            lr (float): learning rate of gradient descent
            in the backpropagation
        """
        self.N = 10
        self.lr = 0.001
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.window_size = 2
        self.words = []
        self.word_index = []
        self.predict = None
        self.V = None
        self.W = None
        self.W1 = None

    def initialize(self, V, data):
        """ randomly pick two W matrices
        params:
            V(int): number of unique words
            data(list/ array): words
        """
        self.V = V
        self.W = np.random.uniform(-0.8, 0.8, (self.V, self.N))
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V))
        self.words = data
        for i in range(len(data)):
            self.word_index[data[i]] = i

    def feed_forward(self, X):
        """ the skip-gram model structure
        params:
            X: data, can be train set or test set
        """
        self.h = np.dot(self.W.T, X).reshape(self.N, 1)
        self.u = np.dot(self.W1.T, self.h)
        self.predict = softmax(self.u)
        return self.predict

    def backpropagate(self, x, t):
        """ do backpropagation to recompute weights
        gradient descent on weight matrices
        params:
            x: training data
            t: label data
        """
        e = self.predict - np.array(t).reshape(self.V, 1)
        dedW1 = np.dot(self.h, e.T)
        X = np.array(x).reshape(self.V, 1)
        dedW = np.dot(X, np.dot(self.W1, e).T)
        self.W1 = self.W1 - self.lr * dedW1
        self.W = self.W - self.lr * dedW

    def train(self, epochs):
        """ train the skip gram model
        params:
            epochs (int): run this many times model
        """
        for x in range(1, epochs):
            self.loss = 0
            for j in range(len(self.X_train)):
                self.feed_forward(self.X_train[j])
                self.backpropagate(self.X_train[j], self.y_train[j])
                C = 0
                for m in range(self.V):
                    if self.y_train[j][m]:
                        self.loss += -1 * self.u[m][0]
                        C += 1
                self.loss += C * np.log(np.sum(np.exp(self.u)))
            print("epoch ", x, " loss = ", self.loss)
            # learning rate adjusts with current epoch
            self.lr *= 1 / (1 + self.lr * x)

    def predict(self, word, number_of_predictions):
        if word in self.words:
            index = self.word_index[word]
            X = [0 for i in range(self.V)]
            X[index] = 1
            prediction = self.feed_forward(X)
            output = {}
            for i in range(self.V):
                output[prediction[i][0]] = i

            top_context_words = []
            for k in sorted(output, reverse=True):
                top_context_words.append(self.words[output[k]])
                if len(top_context_words) >= number_of_predictions:
                    break

            return top_context_words
        else:
            print("Word not found in dicitonary")


def prepare_data_for_training(sentences, w2v):
    """
    params:
        sentences (string): a string of sentences
        w2v(word2vec instance) : w2v model
    """
    data = {}
    # count the occurrence of words
    for sentence in sentences:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    V = len(data)
    # word sort by occurrence
    data = sorted(list(data.keys()))
    # vocab records word and its index
    vocab = {}
    for i in range(len(data)):
        vocab[data[i]] = i
    # count surrounding words
    for sentence in sentences:
        for i in range(len(sentence)):
            # there is a context and center word list for every sentences
            center_word = [0 for x in range(V)]
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)]
            # need to count every center word's context words
            for j in range(i - w2v.window_size, i + w2v.window_size):
                if i != j and 0 <= j < len(sentence):
                    context[vocab[sentence[j]]] += 1
            w2v.X_train.append(center_word)
            w2v.y_train.append(context)
    w2v.initialize(V, data)

    return w2v.X_train, w2v.y_train


def main(text_data):
    """ main function for skipgram models
    params:
        text_data(string): a string of training data
    """
    epochs = 1000
    w2v = word2vec()
    prepare_data_for_training(text_data, w2v)
    w2v.train(epochs)

    print(w2v.predict("around", 3))

