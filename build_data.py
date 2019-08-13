import pickle

from data.sem2010 import constants as sem2010_constants
from dataset.dataset_sem2010 import Sem2010Dataset
from utils import Timer, load_vocab, load_wordnet_superset

import os


def build_sem2010():
    t = Timer()
    t.start('build data sem2010')
    # load vocabs
    vocab_words = load_vocab(sem2010_constants.ALL_WORDS)
    vocab_chars = load_vocab(sem2010_constants.ALL_CHARS)
    vocab_poses = load_vocab(sem2010_constants.ALL_POSES)
    vocab_relations = load_vocab(sem2010_constants.ALL_RELATIONS)
    vocab_wordnet_supersets, _ = load_wordnet_superset()

    for f_name in os.listdir(sem2010_constants.RAW_DATA):
        print('build file', f_name)
        name, _ = f_name.rsplit('.', 1)

        ds = Sem2010Dataset(
            sem2010_constants.RAW_DATA + '{}.txt'.format(name),
            vocab_words, vocab_chars, vocab_poses, vocab_relations, vocab_wordnet_supersets,
        )
        pickle.dump(ds, open(sem2010_constants.PICKLE_DATA + '{}.pickle'.format(name), 'wb'), pickle.HIGHEST_PROTOCOL)

    t.stop()


if __name__ == '__main__':
    build_sem2010()
