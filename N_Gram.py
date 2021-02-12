"""
Ngram: by subdividing texts into n-word tokens, it can be used
to predict the probability of having the next word.
"""
from Text_Preprocess import preprocess_data
n = 3

def ngram(prev_words):
    """ run a ngram model and predict the next words
    params:
        prev_words(tuple): a tuple of previous words
    """
    ngrams = generate_ngram(n)
    model_dict = counter(ngrams, n)
    predict(model_dict, prev_words, n)


def generate_ngram(n):
    """ generate ngram
    params:
        n: the number of words in each gram
    returns:
        a list of ngrams
    """
    s = preprocess_data()
    tokens = [token for token in s.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatenate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def counter(ngrams, n):
    """ build a n-gram model
    n = 3
    """
    model = {}
    for w1, w2, w3 in ngrams:
        model[(w1, w2)][w3] += 1
    for w1_w2 in model:
        total_counts = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_counts
            model[w1_w2] = dict(sorted(model[w1_w2].items(), key=lambda x: x[1]))
    return model

def predict(model, words, n):
    """ predict the probability of
        next word by given first n-1 words.
    params:
    words (tuple): a tuple of words with length n-1
    n (int): the gram number
    """
    if len(words) != n-1:
        raise Exception('The length of words provided should be n-1')
    print(model[words][:10])
