import numpy as np
from collections import Counter
import itertools

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

from utils import WordNet

np.random.seed(13)

wno = WordNet()


def merge_dataset(d1, d2):
    """
    merge 2 datasets into only one dataset
    :param Dataset d1:
    :param Dataset d2:
    :return Dataset:
    """
    r = Dataset(data_name='merged', init=False)

    r.labels = list(itertools.chain(d1.labels, d2.labels))
    r.identities = list(itertools.chain(d1.identities, d2.identities))

    r.words = list(itertools.chain(d1.words, d2.words))
    r.poses = list(itertools.chain(d1.poses, d2.poses))
    r.chars = list(itertools.chain(d1.chars, d2.chars))
    r.s_relations = list(itertools.chain(d1.s_relations, d2.s_relations))
    r.wordnet_supersets = list(itertools.chain(d1.wordnet_supersets, d2.wordnet_supersets))

    r.indexes = list(itertools.chain(d1.indexes, d2.indexes))
    r.child_indexes = list(itertools.chain(d1.child_indexes, d2.child_indexes))
    r.positions = list(itertools.chain(d1.positions, d2.positions))
    r.child_positions = list(itertools.chain(d1.child_positions, d2.child_positions))

    r.relations = list(itertools.chain(d1.relations, d2.relations))
    r.directions = list(itertools.chain(d1.directions, d2.directions))

    return r


class Dataset:
    def __init__(self, data_name, vocab_words=None, vocab_chars=None, vocab_poses=None, vocab_relations=None, vocab_wordnet_supersets=None, init=True):
        self.data_name = data_name
        if not init:
            return

        if self.all_labels is None:
            self.all_labels = []
            raise AttributeError('all_labels is not defined')

        self.labels = None
        self.identities = None

        # original sent
        self.words = None
        self.poses = None
        self.chars = None
        self.wordnet_supersets = None

        self.s_relations = None

        # sdp data
        self.indexes = None
        self.child_indexes = None
        self.positions = None
        self.child_positions = None

        self.relations = None
        self.directions = None

        self.vocab_words = vocab_words
        self.vocab_chars = vocab_chars
        self.vocab_poses = vocab_poses
        self.vocab_relations = vocab_relations
        self.vocab_wordnet_supersets = vocab_wordnet_supersets

        self._process_data()
        self._clean_data()

    def _clean_data(self):
        del self.vocab_words
        del self.vocab_chars
        del self.vocab_poses
        del self.vocab_relations
        del self.vocab_wordnet_supersets

    def _process_data(self):
        with open(self.data_name, 'r') as f:
            raw_data = f.readlines()

        all_labels, self.identities, all_words, all_poses, all_s_relations, self.indexes, self.child_indexes, all_relations, all_directions = self.parse_raw(raw_data)

        labels = []

        words = []
        chars = []
        poses = []
        s_relations = []
        wordnet_supersets = []

        positions = []
        child_positions = []

        relations = []
        directions = []

        for i in range(len(all_labels)):
            labels.append(self.all_labels.index(all_labels[i]))

            # process original sentence
            ws, cs, ps, wns, srs = [], [], [], [], []

            for w, p in zip(all_words[i], all_poses[i]):
                lemma_word = wno.lemmatize(w, p)
                wn_pos = wno.get_wordnet_pos(p)
                wn_key = '{}.{}'.format(lemma_word, wn_pos)
                wn_id = self.vocab_wordnet_supersets[wn_key] if wn_key in self.vocab_wordnet_supersets else 0
                wns.append(wn_id)

                pw, pc = self._process_word(w)
                ws.append(pw)
                cs.append(pc)

                ps.append(self.vocab_poses[p])

            for r in all_s_relations[i]:
                srs.append(self.vocab_relations[r])

            words.append(ws)
            chars.append(cs)
            poses.append(ps)
            s_relations.append(srs)
            wordnet_supersets.append(wns)

            # process sdp
            # process indexes
            e1_index = self.indexes[i][0]
            e2_index = self.indexes[i][-1]
            ps = []
            cps = []
            for j, index in enumerate(self.indexes[i]):
                ps.append((float(index - e1_index), float(index - e2_index)))
                cps.append([float(ci - index) for ci in self.child_indexes[i][j]])

            positions.append(ps)
            child_positions.append(cps)

            # dependencies
            rs, ds = [], []
            for d in all_directions[i]:
                ds.append(1 if d == 'l' else 2)

            for r in all_relations[i]:
                rs.append(self.vocab_relations[r])

            directions.append(ds)
            relations.append(rs)

        self.labels = labels

        self.words = words
        self.poses = poses
        self.chars = chars
        self.s_relations = s_relations
        self.wordnet_supersets = wordnet_supersets

        self.positions = positions
        self.child_positions = child_positions

        self.relations = relations
        self.directions = directions

    def _process_word(self, word):
        """

        :param str word:
        :return:
        """
        char_ids = []
        # 0. get chars of words
        if self.vocab_chars is not None:
            for char in word:
                # ignore chars out of vocabulary
                if char in self.vocab_chars:
                    char_ids += [self.vocab_chars[char]]

        # 2. get id of word
        word = word.lower()
        if word in self.vocab_words:
            word_id = self.vocab_words[word]
        else:
            word_id = self.vocab_words['$UNK$']

        # 3. return tuple word id, char ids
        return word_id, char_ids

    def parse_raw(self, raw_data):
        all_labels = []
        all_identities = []

        all_words = []
        all_poses = []
        all_s_relations = []

        all_indexes = []
        all_child_indexes = []
        all_relations = []
        all_directions = []
        return all_labels, all_identities, all_words, all_poses, all_s_relations, all_indexes, all_child_indexes, all_relations, all_directions

    @staticmethod
    def reverse_sdp(sdp):
        if sdp:
            nodes = sdp.split()
            if len(nodes) % 2:
                ret = []
                for i, node in enumerate(nodes[::-1]):
                    if i % 2:
                        rev_dep = '({}_{})'.format(
                            'r' if node[1] == 'l' else 'l',
                            node[3:-1]
                        )
                        ret.append(rev_dep)
                    else:
                        ret.append(node)

                return ' '.join(ret)
            else:
                raise ValueError('Invalid sdp')
        else:
            return ''

    def __apply_indicates(self, indicates):
        self.labels = [self.labels[i] for i in indicates]
        self.identities = [self.identities[i] for i in indicates]

        self.words = [self.words[i] for i in indicates]
        self.poses = [self.poses[i] for i in indicates]
        self.chars = [self.chars[i] for i in indicates]
        self.s_relations = [self.s_relations[i] for i in indicates]
        self.wordnet_supersets = [self.wordnet_supersets[i] for i in indicates]

        self.indexes = [self.indexes[i] for i in indicates]
        self.child_indexes = [self.child_indexes[i] for i in indicates]
        self.positions = [self.positions[i] for i in indicates]
        self.child_positions = [self.child_positions[i] for i in indicates]

        self.relations = [self.relations[i] for i in indicates]
        self.directions = [self.directions[i] for i in indicates]

    def under_sample(self, n, seed):
        c = Counter(self.labels)
        print('training shape before under sampling: {}'.format({k: c[k] for k in c}))

        d = {k: n for k in c if c[k] > n}
        rus = RandomUnderSampler(ratio=d, random_state=seed, return_indices=True)

        sample_data = [[0]] * len(self.labels)
        _, _, indicates = rus.fit_sample(sample_data, self.labels)
        self.__apply_indicates(indicates)

        c = Counter(self.labels)
        print('training shape after under sampling: {}'.format({k: c[k] for k in c}))

    def over_sample(self, n, seed):
        c = Counter(self.labels)
        print('training shape before over sampling: {}'.format({k: c[k] for k in c}))

        d = {k: n for k in c if c[k] < n}
        ros = RandomOverSampler(ratio=d, random_state=seed)

        sample_data = [[i] for i in range(len(self.labels))]
        data, _ = ros.fit_sample(sample_data, self.labels)
        indicates = [i[0] for i in data]
        self.__apply_indicates(indicates)

        c = Counter(self.labels)
        print('training shape before over sampling: {}'.format({k: c[k] for k in c}))

    def shuffle(self):
        (
            self.labels,
            self.identities,
            self.words,
            self.poses,
            self.chars,
            self.s_relations,
            self.wordnet_supersets,
            self.indexes,
            self.child_indexes,
            self.positions,
            self.child_positions,
            self.relations,
            self.directions
        ) = shuffle(
            self.labels,
            self.identities,
            self.words,
            self.poses,
            self.chars,
            self.s_relations,
            self.wordnet_supersets,
            self.indexes,
            self.child_indexes,
            self.positions,
            self.child_positions,
            self.relations,
            self.directions,
        )

    def clone(self, ratio=100, replace=False, seed=None):
        np.random.seed(seed)
        num_of_example = len(self.labels)
        indicates = np.random.choice(num_of_example, (num_of_example*ratio)//100, replace=replace)
        # print('indicates', [i for i in indicates])

        cloned_data = Dataset('{}_percents'.format(ratio), init=False)

        cloned_data.labels = [self.labels[i] for i in indicates]
        cloned_data.identities = [self.identities[i] for i in indicates]

        cloned_data.words = [self.words[i] for i in indicates]
        cloned_data.poses = [self.poses[i] for i in indicates]
        cloned_data.chars = [self.chars[i] for i in indicates]
        cloned_data.s_relations = [self.s_relations[i] for i in indicates]
        cloned_data.wordnet_supersets = [self.wordnet_supersets[i] for i in indicates]

        cloned_data.indexes = [self.indexes[i] for i in indicates]
        cloned_data.child_indexes = [self.child_indexes[i] for i in indicates]
        cloned_data.positions = [self.positions[i] for i in indicates]
        cloned_data.child_positions = [self.child_positions[i] for i in indicates]

        cloned_data.relations = [self.relations[i] for i in indicates]
        cloned_data.directions = [self.directions[i] for i in indicates]

        return cloned_data

