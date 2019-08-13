import time
import codecs
import numpy as np
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer


def get_trimmed_w2v_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data['embeddings']


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok
    return d


def load_vocab_utf8(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with codecs.open(filename, encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok
    return d


def load_wordnet_superset():
    """
    Returns:
        d: dict[word] = index
        embeddings: list of list embedding
    """
    d = dict()
    embeddings = [np.zeros(45, dtype=float)]
    with open('data/knowledge_base/wordnet_superset.txt') as f:
        for idx, line in enumerate(f):
            word, vec = line.strip().split('\t', 1)
            d[word] = idx + 1  # preserve idx 0 for pad_tok

            embedding = list(map(float, vec.split()))
            embeddings.append(np.array(embedding))

    return d, np.array(embeddings)


def load_wordnet_node2vec():
    """
    Returns:
        d: dict[word] = index
        embeddings: list of list embedding
    """
    d = dict()
    embeddings = [np.zeros(100, dtype=float)]
    with open('data/knowledge_base/node2vec_wordnet.txt') as f:
        for idx, line in enumerate(f):
            word, vec = line.strip().split('\t', 1)
            d[word] = idx + 1  # preserve idx 0 for pad_tok

            embedding = list(map(float, vec.split()))
            embeddings.append(np.array(embedding))

    return d, np.array(embeddings)


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))

        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, 100)

    return sequence_padded, sequence_length


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.job = None

    def start(self, job):
        if job is None:
            return None
        self.start_time = time.time()
        self.job = job
        print("[INFO] {job} started.".format(job=self.job))

    def stop(self):
        if self.job is None:
            return None
        elapsed_time = time.time() - self.start_time
        print("[INFO] {job} finished in {elapsed_time:0.3f} s."
              .format(job=self.job, elapsed_time=elapsed_time))
        self.job = None


class Log:
    verbose = True
    @staticmethod
    def log(text):
        if Log.verbose:
            print(text)


class WordNet:
    lemmer = None

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN

    @staticmethod
    def lemmatize(word, pos):
        if WordNet.lemmer is None:
            WordNet.lemmer = WordNetLemmatizer()
        return WordNet.lemmer.lemmatize(word, WordNet.get_wordnet_pos(pos))
