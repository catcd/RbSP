from dataset.dataset import Dataset
from data.sem2010.constants import ALL_LABELS


class Sem2010Dataset(Dataset):
    def __init__(self, data_name, vocab_words=None, vocab_chars=None, vocab_poses=None, vocab_relations=None, vocab_wordnet_supersets=None):
        self.all_labels = ALL_LABELS
        super().__init__(data_name, vocab_words, vocab_chars, vocab_poses, vocab_relations, vocab_wordnet_supersets)

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

        for line in raw_data:
            id, pair, label, sent, sdp = line.strip().split('\t')

            all_labels.append(label)
            all_identities.append((id, pair))

            # parse original sent
            words = []
            poses = []
            s_relations = []

            for a_word in sent.strip().split():
                w, p, r = a_word.split('|')
                words.append(w)
                poses.append(p)
                s_relations.append(r)

            all_words.append(words)
            all_poses.append(poses)
            all_s_relations.append(s_relations)

            # parse sdp
            indexes = []
            child_indexes = []
            relations = []
            directions = []

            nodes = sdp.split()
            for a_word_node in nodes[::2]:
                index, children = a_word_node.split('|')

                indexes.append(int(index))
                child_indexes.append(list(map(int, children.split(','))) if children != '' else [])

            all_indexes.append(indexes)
            all_child_indexes.append(child_indexes)

            for a_dep_node in nodes[1::2]:
                r = '(' + a_dep_node[3:]
                r = r.split(':', 1)[0] + ')' if ':' in r else r
                d = a_dep_node[1]
                relations.append(r)
                directions.append(d)

            all_relations.append(relations)
            all_directions.append(directions)

        return all_labels, all_identities, all_words, all_poses, all_s_relations, all_indexes, all_child_indexes, all_relations, all_directions
